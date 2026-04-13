# ──────────────────────────────────────────────────────────────────────────────
# Circuit: the main user-facing type for circuit quantization
#
# Wraps the full pipeline: graph → symbolic → modes → numerical Hamiltonian.
# ──────────────────────────────────────────────────────────────────────────────

"""
    Circuit <: AbstractQuantumSystem

A quantized superconducting circuit derived from a graph description.

# Construction
```julia
circ = Circuit(\"\"\"
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [C,  0, 1, EC=0.5]
\"\"\"; ncut=30)
```

# Fields
- `symbolic_circuit` — symbolic analysis results
- `transformation_matrix` — T: node → mode variables
- `var_categories` — mode classification (periodic/extended/free/frozen)
- `cutoffs` — Hilbert space truncation per mode
- `ext_basis` — basis type for extended variables (:harmonic or :grid)
- `phi_grid_ranges` — grid range for discretized extended variables
"""
mutable struct Circuit <: AbstractQuantumSystem
    symbolic_circuit::SymbolicCircuit
    transformation_matrix::Matrix{Float64}
    var_categories::VarCategories
    mode_hamiltonian_symbolic::Num

    # Hilbert space configuration
    cutoffs::Dict{Int, Int}       # mode_index => cutoff dimension
    ext_basis::Symbol             # :harmonic or :grid
    phi_grid_ranges::Dict{Int, Tuple{Float64, Float64}}

    # Oscillator lengths for harmonic basis (computed from LC parameters)
    osc_lengths::Dict{Int, Float64}

    # External flux and offset charge values
    external_flux_values::Vector{Float64}
    offset_charge_values::Dict{Int, Float64}

    # Branch parameter overrides for parameter sweeps (EJ, EL, EC)
    # Key: (branch_index, param_name) => override_value
    branch_param_overrides::Dict{Tuple{Int, Symbol}, Float64}

    # Cache
    _hamiltonian_cache::Union{Nothing, QuantumObject}

    # Hierarchical diagonalization configuration (set by configure!)
    # Types use Any/Vector to avoid forward-reference issues (SubCircuit, HilbertSpace
    # are defined in files loaded after quantization.jl)
    _system_hierarchy::Union{Nothing, Vector}
    _subsystem_trunc_dims::Any                          # nothing or scqubits-style truncation list
    _subsystems::Union{Nothing, Vector}                 # Vector{SubCircuit} when set
    _hilbert_space::Any                                 # HilbertSpace when set
    _hd_cache::Any                                      # internal HD cache tree when set
    _subsystem_sym_hamiltonians::Union{Nothing, Dict{Int, Num}}
    _subsystem_interactions_sym::Union{Nothing, Dict{Set{Int}, Num}}
    _hierarchical_diagonalization::Bool
end

"""
    Circuit(description::String; ncut=30, cutoff_ext=20, ext_basis=:harmonic,
            phi_range=(-6π, 6π), external_fluxes=Float64[], offset_charges=Float64[])

Construct a `Circuit` from a YAML-style description string.
"""
function Circuit(description::String;
                 ncut::Int=30,
                 cutoff_ext::Int=20,
                 ext_basis::Symbol=:harmonic,
                 phi_range::Tuple{Float64, Float64}=(-6π, 6π),
                 external_fluxes::Vector{Float64}=Float64[],
                 offset_charges::Vector{Float64}=Float64[])
    # Parse and analyze
    graph = parse_circuit(description)
    sc = build_symbolic_circuit(graph)
    T, vc = compute_variable_transformation(sc)
    H_mode = transform_hamiltonian(sc, T, vc)

    # Set up cutoffs
    cutoffs = Dict{Int, Int}()
    for i in vc.periodic
        cutoffs[i] = 2 * ncut + 1
    end
    for i in vc.extended
        cutoffs[i] = cutoff_ext
    end

    # Grid ranges for extended variables
    phi_ranges = Dict{Int, Tuple{Float64, Float64}}()
    for i in vc.extended
        phi_ranges[i] = phi_range
    end

    # External fluxes: default to 0
    n_ext = length(sc.external_fluxes)
    ext_vals = length(external_fluxes) == n_ext ? external_fluxes : zeros(n_ext)

    # Offset charges: keyed by actual periodic mode index
    ng_vals = _build_offset_charge_values(vc, offset_charges)

    circ = Circuit(sc, T, vc, H_mode,
                   cutoffs, ext_basis, phi_ranges, Dict{Int, Float64}(),
                   ext_vals, ng_vals,
                   Dict{Tuple{Int, Symbol}, Float64}(), nothing,
                   # Hierarchical diagonalization config (initially unconfigured)
                   nothing, nothing, nothing, nothing, nothing, nothing, nothing, false)

    # Compute oscillator lengths (needs the Circuit to read branch params)
    circ.osc_lengths = _compute_osc_lengths(circ)

    return circ
end

function _build_offset_charge_values(vc::VarCategories, offset_charges::Vector{Float64})
    periodic_modes = vc.periodic
    isempty(offset_charges) && return Dict{Int, Float64}(mode => 0.0 for mode in periodic_modes)
    length(offset_charges) == length(periodic_modes) || throw(ArgumentError(
        "Expected $(length(periodic_modes)) offset charges in periodic-mode order $(periodic_modes), got $(length(offset_charges))"
    ))
    return Dict(mode => Float64(offset_charges[idx]) for (idx, mode) in enumerate(periodic_modes))
end

# ── AbstractQuantumSystem interface ──────────────────────────────────────────

function hilbertdim(circ::Circuit)
    dims = _subsystem_dims(circ)
    return prod(dims)
end

function hamiltonian(circ::Circuit)
    circ._hamiltonian_cache !== nothing && return circ._hamiltonian_cache

    H = _build_numerical_hamiltonian(circ)
    circ._hamiltonian_cache = H
    return H
end

function _invalidate_hamiltonian_cache!(circ::Circuit)
    circ._hamiltonian_cache = nothing
    # Recompute oscillator lengths if branch parameters have been overridden
    if !isempty(circ.branch_param_overrides)
        circ.osc_lengths = _compute_osc_lengths(circ)
    end
    return circ
end

function _clear_hierarchical_cache!(circ::Circuit)
    # Clear numerical HD results (symbolic decomposition stays valid)
    if circ._hierarchical_diagonalization
        circ._hilbert_space = nothing
        circ._subsystems = nothing
        circ._hd_cache = nothing
    end
    return circ
end

function _refresh_configured_hierarchy_after_param!(circ::Circuit, param_name::Symbol)
    _invalidate_hamiltonian_cache!(circ)
    if circ._hierarchical_diagonalization
        if circ._hd_cache !== nothing
            _refresh_configured_hierarchical!(circ, param_name)
        else
            circ._hilbert_space = nothing
            circ._subsystems = nothing
        end
    end
    return circ
end

"""Invalidate cached Hamiltonian (call after changing parameters)."""
function invalidate_cache!(circ::Circuit)
    _invalidate_hamiltonian_cache!(circ)
    _clear_hierarchical_cache!(circ)
    return circ
end

# ── Hierarchical diagonalization configuration ─────────────────────────────

"""
    _decompose_sym_hamiltonian(circ::Circuit, system_hierarchy::Vector)

Decompose `circ.mode_hamiltonian_symbolic` into per-subsystem Hamiltonians
and per-subsystem interaction terms, matching the scqubits sequential
subtraction algorithm in `_sym_subsystem_hamiltonian_and_interactions`.

Returns `(subsys_H, interactions)` where:
- `subsys_H::Dict{Int, Num}` maps 1-based subsystem group index to its symbolic H
- `interactions::Dict{Int, Num}` maps 1-based subsystem group index to the
  interaction terms first encountered when processing that subsystem (scqubits
  stores interactions per-subsystem, not per-set).

Additionally populates `interactions_by_set::Dict{Set{Int}, Num}` which is
stored as a secondary index for the `sym_interaction` exact-set lookup.
"""
function _decompose_sym_hamiltonian(circ::Circuit, system_hierarchy::Vector)
    n = circ.symbolic_circuit.graph.num_nodes

    # Build mode_index → group_index mapping (1-based groups)
    mode_to_group = Dict{Int, Int}()
    flat_groups = _flatten_hierarchy_groups(system_hierarchy)
    for (gi, group) in enumerate(flat_groups)
        for mi in group
            mode_to_group[mi] = gi
        end
    end
    n_groups = length(flat_groups)

    # Build variable → mode_index lookup (operator symbols only: θ_i and nθ_i)
    var_to_mode = Dict{Any, Int}()
    for i in 1:n
        var_to_mode[Symbolics.unwrap(Symbolics.variable(:θ, i))] = i
        var_to_mode[Symbolics.unwrap(Symbolics.variable(:nθ, i))] = i
    end

    # ── Helper: classify operator-variable groups for a single term ───────
    # A variable is an "operator" if it maps to a mode via var_to_mode.
    # All other symbols (external fluxes, offset charges, branch parameters)
    # are ignored during classification.
    function _term_groups(raw_term)
        term = Symbolics.wrap(raw_term)
        vars = Symbolics.get_variables(term)
        groups = Set{Int}()
        for v in vars
            vu = Symbolics.unwrap(v)
            if haskey(var_to_mode, vu)
                mi = var_to_mode[vu]
                haskey(mode_to_group, mi) && push!(groups, mode_to_group[mi])
            end
        end
        return groups
    end

    # ── Step 1: separate constant terms (no operator variables) ───────────
    H_expanded = Symbolics.expand(circ.mode_hamiltonian_symbolic)
    H_unwrapped = Symbolics.unwrap(H_expanded)
    all_terms = if Symbolics.iscall(H_unwrapped) && Symbolics.operation(H_unwrapped) === (+)
        Symbolics.arguments(H_unwrapped)
    else
        [H_unwrapped]
    end

    constants = Num[]
    op_terms  = Num[]
    for raw_term in all_terms
        groups = _term_groups(raw_term)
        if isempty(groups)
            push!(constants, Symbolics.wrap(raw_term))
        else
            push!(op_terms, Symbolics.wrap(raw_term))
        end
    end

    # Running Hamiltonian (without constants — they are distributed later).
    running_H = isempty(op_terms) ? Num(0) : reduce(+, op_terms)

    # ── Step 2: sequential subsystem extraction (scqubits algorithm) ──────
    subsys_H   = Dict{Int, Num}(gi => Num(0) for gi in 1:n_groups)
    int_per_subsys = Dict{Int, Num}(gi => Num(0) for gi in 1:n_groups)

    for gi in 1:n_groups
        subsys_modes = Set(flat_groups[gi])

        # Expand current running Hamiltonian to additive terms
        rH = Symbolics.expand(running_H)
        rH_unwrapped = Symbolics.unwrap(rH)
        terms_now = if Symbolics.iscall(rH_unwrapped) && Symbolics.operation(rH_unwrapped) === (+)
            Symbolics.arguments(rH_unwrapped)
        else
            [rH_unwrapped]
        end

        H_sys = Num(0)
        H_int = Num(0)
        for raw_term in terms_now
            groups = _term_groups(raw_term)
            term = Symbolics.wrap(raw_term)
            # Pure subsystem term: all operator vars are within this subsystem
            if !isempty(groups) && issubset(groups, Set([gi]))
                H_sys = H_sys + term
            # Interaction term: has operators in this subsystem AND outside
            elseif !isempty(intersect(groups, Set([gi]))) && !issubset(groups, Set([gi]))
                H_int = H_int + term
            end
        end

        # Distribute constants that share symbolic variables with H_sys
        sys_vars = Set(Symbolics.unwrap.(Symbolics.get_variables(H_sys)))
        owned_consts = Num[]
        remaining_consts = Num[]
        for c in constants
            c_vars = Set(Symbolics.unwrap.(Symbolics.get_variables(c)))
            if !isempty(intersect(c_vars, sys_vars))
                push!(owned_consts, c)
            else
                push!(remaining_consts, c)
            end
        end
        constants = remaining_consts
        for c in owned_consts
            H_sys = H_sys + c
        end

        subsys_H[gi]       = H_sys
        int_per_subsys[gi] = H_int

        # Subtract processed terms from running Hamiltonian
        running_H = Symbolics.expand(running_H - H_sys - H_int)
    end

    # Leftover constants → first subsystem (scqubits convention)
    for c in constants
        subsys_H[1] = subsys_H[1] + c
    end

    # ── Step 3: build secondary index by subsystem-set for sym_interaction ─
    interactions_by_set = Dict{Set{Int}, Num}()
    for gi in 1:n_groups
        int_expr = Symbolics.expand(int_per_subsys[gi])
        int_unwrapped = Symbolics.unwrap(int_expr)
        int_terms = if Symbolics.iscall(int_unwrapped) && Symbolics.operation(int_unwrapped) === (+)
            Symbolics.arguments(int_unwrapped)
        else
            [int_unwrapped]
        end
        for raw_term in int_terms
            groups = _term_groups(raw_term)
            isempty(groups) && continue
            key = Set(groups)
            interactions_by_set[key] = get(interactions_by_set, key, Num(0)) + Symbolics.wrap(raw_term)
        end
    end

    return (subsys_H, interactions_by_set)
end

"""Extract flat list of leaf mode-index groups from a system_hierarchy specification."""
function _flatten_hierarchy_groups(hierarchy::Vector)
    groups = Vector{Int}[]
    for item in hierarchy
        if item isa Vector{Int}
            push!(groups, item)
        elseif item isa Vector
            # Nested group: recursively flatten all leaves within it into one group
            push!(groups, _collect_all_modes(item))
        else
            error("Invalid system_hierarchy element: $item")
        end
    end
    return groups
end

"""Recursively collect all mode indices from a nested hierarchy element."""
function _collect_all_modes(v::Vector{Int})
    return copy(v)
end

function _collect_all_modes(v::Vector)
    modes = Int[]
    for item in v
        append!(modes, _collect_all_modes(item))
    end
    return modes
end

"""
    configure!(circ::Circuit; system_hierarchy, subsystem_trunc_dims)

Configure hierarchical diagonalization for the circuit, storing hierarchy
metadata and computing both symbolic and numerical decompositions.

This is the Julia equivalent of `circ.configure(...)` in Python scqubits.
After calling, the circuit provides:
- `sym_hamiltonian(circ; subsystem_index=i)` for per-subsystem symbolic Hamiltonians
- `sym_interaction(circ; subsystem_indices=(i,j,...))` for symbolic interaction terms
- `circ._subsystems` for the diagonalized subsystem objects
- `circ._hilbert_space` for the diagonalized composite system

Uses 1-based indexing (Julia convention) for subsystem indices.

# Example
```julia
configure!(circ; system_hierarchy=[[1], [2]], subsystem_trunc_dims=[10, 10])
sym_hamiltonian(circ; subsystem_index=1)      # symbolic H for subsystem 1
sym_interaction(circ; subsystem_indices=(1, 2), return_expr=true) # coupling between subsystems 1 and 2
```
"""
function configure!(circ::Circuit; system_hierarchy, subsystem_trunc_dims=nothing)
    subsystem_trunc_dims === nothing && throw(ArgumentError(
        "subsystem_trunc_dims is required in strict parity mode. " *
        "Use truncation_template(system_hierarchy) as a starting point."
    ))
    # Validate truncation shape up front (throws on mismatch).
    _ = _subsystem_trunc_dims_to_path_dict(system_hierarchy, subsystem_trunc_dims)

    # Store configuration
    circ._system_hierarchy = system_hierarchy
    circ._subsystem_trunc_dims = deepcopy(subsystem_trunc_dims)
    circ._hierarchical_diagonalization = true

    # Symbolic decomposition
    subsys_H, interactions = _decompose_sym_hamiltonian(circ, system_hierarchy)
    circ._subsystem_sym_hamiltonians = subsys_H
    circ._subsystem_interactions_sym = interactions

    # Numerical hierarchical diagonalization with reusable cache
    hs = _configure_hierarchical_cache!(circ;
        system_hierarchy=system_hierarchy,
        subsystem_trunc_dims=subsystem_trunc_dims)
    circ._hilbert_space = hs
    circ._subsystems = SubCircuit[s for s in hs.subsystems]

    return circ
end

# ── Symbolic accessors ──────────────────────────────────────────────────────

"""
    _raw_sym_hamiltonian_expr(circ::Circuit; subsystem_index=nothing)

Return the unformatted symbolic Hamiltonian expression (in mode variables).

Without `subsystem_index`: returns the full-circuit symbolic Hamiltonian.
With `subsystem_index=i` (1-based): returns the symbolic Hamiltonian for
subsystem `i` only, as decomposed by [`configure!`](@ref).
"""
function _raw_sym_hamiltonian_expr(circ::Circuit; subsystem_index::Union{Nothing,Int}=nothing)
    if subsystem_index === nothing
        return circ.mode_hamiltonian_symbolic
    end
    circ._hierarchical_diagonalization ||
        error("subsystem_index requires configure!() to be called first")
    circ._subsystem_sym_hamiltonians === nothing &&
        error("Symbolic decomposition not available. Call configure!() first.")
    haskey(circ._subsystem_sym_hamiltonians, subsystem_index) ||
        error("subsystem_index=$subsystem_index is out of range. " *
              "Valid indices: $(sort(collect(keys(circ._subsystem_sym_hamiltonians))))")
    return circ._subsystem_sym_hamiltonians[subsystem_index]
end

"""
    sym_hamiltonian(circ::Circuit;
                    subsystem_index=nothing,
                    float_round=6,
                    print_latex=false,
                    return_expr=false)

Return or print the formatted symbolic Hamiltonian expression.

Julia uses 1-based subsystem indices. With `return_expr=true`, printing is
suppressed and the formatted `Symbolics.Num` is returned.
"""
function sym_hamiltonian(circ::Circuit;
                         subsystem_index::Union{Nothing,Int}=nothing,
                         float_round::Int=6,
                         print_latex::Bool=false,
                         return_expr::Bool=false)
    expr = _raw_sym_hamiltonian_expr(circ; subsystem_index=subsystem_index)
    display_expr = _format_symbolic_expr(circ, expr; float_round=float_round)

    if return_expr
        return display_expr
    end

    if print_latex
        println(latexify(display_expr))
    end
    println(display_expr)
    return nothing
end

"""Return the symbolic Hamiltonian expression (in node variables)."""
sym_hamiltonian_node(circ::Circuit) = circ.symbolic_circuit.hamiltonian_symbolic

"""
    sym_interaction(circ::Circuit;
                    subsystem_indices::Tuple{Vararg{Int}},
                    float_round=6,
                    print_latex=false,
                    return_expr=false)

Return or print symbolic interaction terms for an arbitrary set of subsystems.

`subsystem_indices` follows scqubits semantics: interaction terms are included
only when their involved subsystem-index set exactly matches the requested set.
Ordering is ignored, so `(1,2,3)` and `(3,1,2)` are equivalent.

# Keywords
- `float_round::Int=6` — round floating-point coefficients to this many
  decimal places in the returned/printed expression (matches scqubits
  `_make_expr_human_readable` behavior).
- `print_latex::Bool=false` — additionally print LaTeX representation.
- `return_expr::Bool=false` — if `true`, suppress printing and return the
  formatted expression. **Note:** the scqubits default is `False` (print mode);
  we match that here.

Requires [`configure!`](@ref) to have been called first.
"""
function sym_interaction(circ::Circuit;
                         subsystem_indices::Tuple{Vararg{Int}},
                         float_round::Int=6,
                         print_latex::Bool=false,
                         return_expr::Bool=false)
    expr = _raw_subsystem_interaction_expr(circ, subsystem_indices)
    display_expr = _format_interaction_expr(circ, expr; float_round=float_round)

    if return_expr
        return display_expr
    end

    if print_latex
        println(latexify(display_expr))
    end
    println(display_expr)
    return nothing
end

function _raw_subsystem_interaction_expr(circ::Circuit,
                                         subsystem_indices::Tuple{Vararg{Int}})
    circ._hierarchical_diagonalization ||
        error("sym_interaction requires configure!() to be called first")
    circ._subsystem_interactions_sym === nothing &&
        error("Symbolic decomposition not available. Call configure!() first.")
    unique_subsystems = sort(unique(collect(subsystem_indices)))
    length(unique_subsystems) >= 2 || throw(ArgumentError(
        "subsystem_indices must contain at least two subsystem indices."
    ))
    key = Set(unique_subsystems)
    return get(circ._subsystem_interactions_sym, key, Num(0))
end

function _format_interaction_expr(circ::Circuit, expr::Num; float_round::Int=6)
    return _format_symbolic_expr(circ, expr; float_round=float_round)
end

function _format_symbolic_expr(circ::Circuit, expr::Num; float_round::Int=6)
    display_expr = _round_symbolic_floats(expr, float_round)
    return _humanize_external_fluxes(circ, display_expr)
end

"""Round floating-point literal coefficients in a Symbolics expression."""
function _round_symbolic_floats(expr::Num, digits::Int)
    subs = Pair{Num, Num}[]
    _find_float_leaves!(subs, Symbolics.unwrap(expr), digits, Set{UInt}())
    isempty(subs) && return expr
    result = expr
    for (old, new) in subs
        result = Symbolics.substitute(result, Dict(old => new))
    end
    return result
end

function _find_float_leaves!(subs, t, digits, visited)
    id = objectid(t)
    id in visited && return
    push!(visited, id)
    if !Symbolics.iscall(t)
        v = Symbolics.value(t)
        if v isa AbstractFloat
            rounded = round(v; digits=digits)
            if rounded != v
                push!(subs, Symbolics.wrap(t) => Num(rounded))
            end
        end
        return
    end
    for a in Symbolics.arguments(t)
        _find_float_leaves!(subs, a, digits, visited)
    end
end

"""Replace external flux symbols with `Φi/(2π)` form in a symbolic expression."""
function _humanize_external_fluxes(circ::Circuit, expr::Num)
    sc = circ.symbolic_circuit
    isempty(sc.external_fluxes) && return expr
    subs = Dict{Num, Num}()
    for flux in sc.external_fluxes
        subs[flux] = flux / (2π)
    end
    return Symbolics.substitute(expr, subs)
end

"""Return `(T, var_categories)` — the variable transformation from node to mode basis."""
variable_transformation(circ::Circuit) = (circ.transformation_matrix, circ.var_categories)

"""Return the symbolic external flux variables."""
external_fluxes(circ::Circuit) = circ.symbolic_circuit.external_fluxes

"""
    sym_external_fluxes(circ::Circuit)

Return a scqubits-style loop-to-flux mapping for the superconducting loops
identified during symbolic circuit construction.

The returned dictionary is keyed by the symbolic flux variables `Φ1`, `Φ2`, ...
and each value is a named tuple with:
- `closure_branch::Int` — the closure branch carrying that flux variable
- `loop::Vector{Tuple{Int,Int}}` — the corresponding fundamental loop as
  `(branch_index, sign)` pairs
"""
function sym_external_fluxes(circ::Circuit)
    sc = circ.symbolic_circuit
    mapping = Dict{Num, NamedTuple{(:closure_branch, :loop), Tuple{Int, Vector{Tuple{Int, Int}}}}}()
    for (flux, closure_branch, loop) in zip(sc.external_fluxes,
                                            sc.superconducting_closure_branches,
                                            sc.superconducting_loops)
        mapping[flux] = (closure_branch=closure_branch, loop=copy(loop))
    end
    return mapping
end

"""Return the symbolic offset charge variables."""
offset_charges(circ::Circuit) = circ.symbolic_circuit.offset_charges

"""
    offset_charge_transformation(circ::Circuit)

Return symbolic equations mapping periodic-mode offset charges `ng<i>` to node
charge placeholders `q_n1, q_n2, ...` using `inv(Tᵀ)`.
"""
function offset_charge_transformation(circ::Circuit)
    Tt_inv = inv(circ.transformation_matrix')
    n = circ.symbolic_circuit.graph.num_nodes
    node_charge_vars = Num[Num(Symbolics.variable(Symbol("q_n$(j)"))) for j in 1:n]

    eqs = Any[]
    for mode in circ.var_categories.periodic
        lhs = Num(Symbolics.variable(Symbol("ng$(mode)")))
        rhs = sum(Tt_inv[mode, j] * node_charge_vars[j] for j in 1:n; init=Num(0))
        push!(eqs, lhs ~ rhs)
    end

    display(eqs)
    return eqs
end

# ── Dynamic Circuit properties ───────────────────────────────────────────────

function _cutoff_names(circ::Circuit)
    vc = getfield(circ, :var_categories)
    names = Symbol[]
    append!(names, Symbol("cutoff_n_$(mode)") for mode in vc.periodic)
    append!(names, Symbol("cutoff_ext_$(mode)") for mode in vc.extended)
    return names
end

function _cutoff_property_info(circ::Circuit, name::Symbol)
    s = string(name)
    if (m = match(r"^cutoff_n_(\d+)$", s)) !== nothing
        mode = parse(Int, m.captures[1])
        mode in getfield(circ, :var_categories).periodic ||
            throw(ArgumentError("cutoff_n_$mode is only defined for periodic modes $(getfield(circ, :var_categories).periodic)"))
        return (:periodic, mode)
    elseif (m = match(r"^cutoff_ext_(\d+)$", s)) !== nothing
        mode = parse(Int, m.captures[1])
        mode in getfield(circ, :var_categories).extended ||
            throw(ArgumentError("cutoff_ext_$mode is only defined for extended modes $(getfield(circ, :var_categories).extended)"))
        return (:extended, mode)
    end
    return nothing
end

function Base.getproperty(circ::Circuit, name::Symbol)
    if name === :cutoff_names
        return _cutoff_names(circ)
    end
    info = _cutoff_property_info(circ, name)
    if info === nothing
        return getfield(circ, name)
    end

    kind, mode = info
    dim = getfield(circ, :cutoffs)[mode]
    return kind === :periodic ? (dim - 1) ÷ 2 : dim
end

function Base.setproperty!(circ::Circuit, name::Symbol, value)
    info = _cutoff_property_info(circ, name)
    if info === nothing
        return setfield!(circ, name, value)
    end

    kind, mode = info
    value isa Integer || throw(ArgumentError("$(name) must be set to an integer"))
    intval = Int(value)

    if kind === :periodic
        intval >= 0 || throw(ArgumentError("$(name) must be a nonnegative integer"))
        getfield(circ, :cutoffs)[mode] = 2 * intval + 1
    else
        intval >= 1 || throw(ArgumentError("$(name) must be a positive integer"))
        getfield(circ, :cutoffs)[mode] = intval
    end

    invalidate_cache!(circ)
    return intval
end

function Base.propertynames(circ::Circuit, private::Bool=false)
    names = collect(fieldnames(typeof(circ)))
    push!(names, :cutoff_names)
    append!(names, _cutoff_names(circ))
    return Tuple(names)
end

# ── Symbolic Lagrangian ────────────────────────────────────────────────────────

"""
    sym_lagrangian(circ::Circuit; vars_type::Symbol=:node)

Return the symbolic Lagrangian of the circuit as a `Symbolics.Num` expression.

# Keyword arguments
- `vars_type=:node`: Lagrangian in node flux variables φᵢ, φ̇ᵢ
- `vars_type=:new`: Lagrangian in transformed mode variables θᵢ, θ̇ᵢ

The Lagrangian is L = T - V where T is the capacitive kinetic energy
and V contains inductive and Josephson potential terms.
"""
function sym_lagrangian(circ::Circuit; vars_type::Symbol=:node)
    vars_type in (:node, :new) || throw(ArgumentError(
        "vars_type must be :node or :new, got :$vars_type"))

    sc = circ.symbolic_circuit
    n = sc.graph.num_nodes
    cg = sc.graph

    if vars_type == :node
        return _sym_lagrangian_node(sc, cg, n)
    else
        return _sym_lagrangian_new(sc, cg, n, circ.transformation_matrix)
    end
end

function _sym_lagrangian_node(sc::SymbolicCircuit, cg::CircuitGraph, n::Int)
    # Kinetic energy: T = (1/2) φ̇ᵀ C φ̇
    kinetic = Num(0)
    for i in 1:n, j in 1:n
        kinetic += sc.capacitance_matrix[i, j] * sc.node_dot_vars[i] * sc.node_dot_vars[j]
    end
    kinetic = kinetic / 2

    # Inductive potential: Σ (EL/2)(branch_flux + Φext)²
    V_inductive = _build_inductive_terms(cg, sc.node_vars, sc.branch_flux_allocations)

    # Josephson potential: Σ -EJ cos(phase + Φext)
    V_JJ = sum(-ej * cos(phase) for (ej, phase) in sc.josephson_terms; init=Num(0))

    return Symbolics.simplify(kinetic - V_inductive - V_JJ)
end

function _sym_lagrangian_new(sc::SymbolicCircuit, cg::CircuitGraph, n::Int,
                              T::Matrix{Float64})
    T_inv = inv(T)

    # Mode variables
    θ_vars = [Symbolics.variable(:θ, i) for i in 1:n]
    θ̇_vars = [Symbolics.variable(:θ̇, i) for i in 1:n]

    # Kinetic energy: T = (1/2) θ̇ᵀ Cθ θ̇  where Cθ = T⁻ᵀ C T⁻¹
    C_float = Float64.(Symbolics.value.(sc.capacitance_matrix))
    C_transformed = T_inv' * C_float * T_inv

    kinetic = Num(0)
    for i in 1:n, j in 1:n
        abs(C_transformed[i, j]) < 1e-15 && continue
        kinetic += C_transformed[i, j] * θ̇_vars[i] * θ̇_vars[j]
    end
    kinetic = kinetic / 2

    # Substitution map: φᵢ → Σⱼ T⁻¹[i,j] θⱼ
    node_subs = Dict(sc.node_vars[i] => sum(T_inv[i, j] * θ_vars[j] for j in 1:n)
                      for i in 1:n)

    # Inductive potential in mode basis
    V_inductive = Num(0)
    for (bi, b) in enumerate(cg.branches)
        b.branch_type == L_branch || continue
        el = b.parameters[:EL]
        branch_flux_node = _branch_phase(b, sc.node_vars) + sc.branch_flux_allocations[bi]
        branch_flux_mode = Symbolics.substitute(branch_flux_node, node_subs)
        V_inductive += el / 2 * branch_flux_mode^2
    end

    # Josephson potential in mode basis
    V_JJ = Num(0)
    for (ej, phase_expr) in sc.josephson_terms
        new_phase = Symbolics.substitute(phase_expr, node_subs)
        V_JJ += -ej * cos(new_phase)
    end

    return Symbolics.simplify(kinetic - V_inductive - V_JJ)
end

# ── Parameter setters ────────────────────────────────────────────────────────

"""Set external flux value. `index` is 1-based."""
function set_external_flux!(circ::Circuit, index::Int, value::Float64)
    old = circ.external_flux_values[index]
    isequal(old, value) && return circ
    circ.external_flux_values[index] = value
    _refresh_configured_hierarchy_after_param!(circ, Symbol("Φ$(index)"))
    return circ
end

"""Set offset charge value for periodic mode `index`."""
function set_offset_charge!(circ::Circuit, index::Int, value::Float64)
    index in circ.var_categories.periodic ||
        throw(ArgumentError("Offset charge ng$index is only defined for periodic modes $(circ.var_categories.periodic)"))
    old = get(circ.offset_charge_values, index, 0.0)
    isequal(old, value) && return circ
    circ.offset_charge_values[index] = value
    _refresh_configured_hierarchy_after_param!(circ, Symbol("ng$(index)"))
    return circ
end

# ── set_param! / get_param for parameter sweeps ─────────────────────────────

function _circuit_flux_index(circ::Circuit, param_name::Symbol)
    s = string(param_name)
    m = match(r"^Φ(\d+)$", s)
    m === nothing && return nothing
    idx = parse(Int, m.captures[1])
    1 <= idx <= length(circ.external_flux_values) ||
        error("External flux index $idx out of range for circuit with $(length(circ.external_flux_values)) flux parameters")
    return idx
end

function _circuit_charge_index(circ::Circuit, param_name::Symbol)
    s = string(param_name)
    m = match(r"^ng(\d+)$", s)
    m === nothing && return nothing
    idx = parse(Int, m.captures[1])
    idx in circ.var_categories.periodic ||
        error("Offset charge ng$idx is only defined for periodic modes $(circ.var_categories.periodic)")
    return idx
end

function set_param!(circ::Circuit, param_name::Symbol, val)
    if (idx = _circuit_flux_index(circ, param_name)) !== nothing
        set_external_flux!(circ, idx, Float64(val))
    elseif (idx = _circuit_charge_index(circ, param_name)) !== nothing
        set_offset_charge!(circ, idx, Float64(val))
    elseif param_name in (:EJ, :EC, :EL)
        _set_branch_param_first!(circ, param_name, Float64(val))
        invalidate_cache!(circ)
        return
    elseif (m = match(r"^(EJ|EC|EL)_(\d+)$", string(param_name))) !== nothing
        pname = Symbol(m.captures[1])
        branch_idx = parse(Int, m.captures[2])
        1 <= branch_idx <= length(circ.symbolic_circuit.graph.branches) ||
            error("Branch index $branch_idx out of range")
        circ.branch_param_overrides[(branch_idx, pname)] = Float64(val)
        invalidate_cache!(circ)
    else
        error("Circuit parameter :$param_name not recognized. " *
              "Use :Φ1, :Φ2, ..., :ng1, :ng2, ..., :EJ, :EC, :EL, :EJ_N, :EC_N, :EL_N")
    end
end

function get_param(circ::Circuit, param_name::Symbol)
    if (idx = _circuit_flux_index(circ, param_name)) !== nothing
        return circ.external_flux_values[idx]
    elseif (idx = _circuit_charge_index(circ, param_name)) !== nothing
        return get(circ.offset_charge_values, idx, 0.0)
    elseif param_name in (:EJ, :EC, :EL)
        return _get_branch_param_first(circ, param_name)
    elseif (m = match(r"^(EJ|EC|EL)_(\d+)$", string(param_name))) !== nothing
        pname = Symbol(m.captures[1])
        branch_idx = parse(Int, m.captures[2])
        return _get_branch_param(circ, branch_idx, pname)
    else
        error("Circuit parameter :$param_name not recognized. " *
              "Use :Φ1, :Φ2, ..., :ng1, :ng2, ..., :EJ, :EC, :EL, :EJ_N, :EC_N, :EL_N")
    end
end

"""Set parameter on the first branch that has it."""
function _set_branch_param_first!(circ::Circuit, param::Symbol, val::Float64)
    for (bi, b) in enumerate(circ.symbolic_circuit.graph.branches)
        if haskey(b.parameters, param)
            circ.branch_param_overrides[(bi, param)] = val
            return
        end
    end
    error("No branch in circuit has parameter :$param")
end

"""Get parameter from the first branch that has it."""
function _get_branch_param_first(circ::Circuit, param::Symbol)
    for (bi, b) in enumerate(circ.symbolic_circuit.graph.branches)
        if haskey(b.parameters, param)
            return _get_branch_param(circ, bi, param)
        end
    end
    error("No branch in circuit has parameter :$param")
end

# ── Branch parameter helpers ────────────────────────────────────────────────

"""Return effective value of `param` for branch `branch_idx`, using override if set."""
function _get_branch_param(circ::Circuit, branch_idx::Int, param::Symbol)
    key = (branch_idx, param)
    return get(circ.branch_param_overrides, key,
               circ.symbolic_circuit.graph.branches[branch_idx].parameters[param])
end

"""Build N×N capacitance matrix using current (overridden) branch parameters."""
function _build_capacitance_matrix_numeric(circ::Circuit)
    cg = circ.symbolic_circuit.graph
    n = cg.num_nodes
    C = zeros(Float64, n, n)

    for (bi, b) in enumerate(cg.branches)
        haskey(b.parameters, :EC) || continue
        ec = _get_branch_param(circ, bi, :EC)
        cap = 1.0 / (8.0 * ec)

        i, j = b.node_i, b.node_j
        if i != 0 && j != 0
            C[i, i] += cap; C[j, j] += cap; C[i, j] -= cap; C[j, i] -= cap
        elseif i == 0 && j != 0
            C[j, j] += cap
        elseif j == 0 && i != 0
            C[i, i] += cap
        end
    end
    return C
end

"""Build N×N inverse inductance matrix using current (overridden) branch parameters."""
function _build_inv_inductance_matrix_numeric(circ::Circuit)
    cg = circ.symbolic_circuit.graph
    n = cg.num_nodes
    L_inv = zeros(Float64, n, n)

    for (bi, b) in enumerate(cg.branches)
        b.branch_type == L_branch || continue
        el = _get_branch_param(circ, bi, :EL)

        i, j = b.node_i, b.node_j
        if i != 0 && j != 0
            L_inv[i, i] += el; L_inv[j, j] += el; L_inv[i, j] -= el; L_inv[j, i] -= el
        elseif i == 0 && j != 0
            L_inv[j, j] += el
        elseif j == 0 && i != 0
            L_inv[i, i] += el
        end
    end
    return L_inv
end

"""Evaluate the numerical external flux for a given branch from its symbolic allocation."""
function _eval_branch_ext_flux(circ::Circuit, bi::Int)
    sc = circ.symbolic_circuit
    alloc = sc.branch_flux_allocations[bi]
    subs = Dict{Num, Float64}()
    for (fi, ef) in enumerate(sc.external_fluxes)
        subs[ef] = fi <= length(circ.external_flux_values) ?
                   circ.external_flux_values[fi] : 0.0
    end
    return Float64(Symbolics.value(Symbolics.substitute(alloc, subs)))
end

"""Coefficient of mode `mode_idx` in a branch's phase (φ_j - φ_i) after transformation."""
function _branch_mode_coeff(b::Branch, T_inv::Matrix{Float64}, mode_idx::Int)
    w = 0.0
    if b.node_j != 0
        w += T_inv[b.node_j, mode_idx]
    end
    if b.node_i != 0
        w -= T_inv[b.node_i, mode_idx]
    end
    return w
end

"""Return current EJ values for each Josephson term, respecting overrides.
Order matches `sc.josephson_terms`."""
function _get_josephson_ej_values(circ::Circuit)
    cg = circ.symbolic_circuit.graph
    ej_vals = Float64[]
    for (bi, b) in enumerate(cg.branches)
        b.branch_type == JJ_branch || continue
        push!(ej_vals, _get_branch_param(circ, bi, :EJ))
    end
    return ej_vals
end

"""
    list_branch_params(circ::Circuit)

Print all branch parameters with their current (effective) values.
"""
function list_branch_params(circ::Circuit)
    cg = circ.symbolic_circuit.graph
    for (bi, b) in enumerate(cg.branches)
        type_str = string(b.branch_type)
        for (pname, _) in b.parameters
            val = _get_branch_param(circ, bi, pname)
            println("  Branch $bi ($type_str, $(b.node_i)->$(b.node_j)): $pname = $val")
        end
    end
end

# ── Numerical Hamiltonian construction ───────────────────────────────────────

function _subsystem_dims(circ::Circuit)
    vc = circ.var_categories
    dims = Int[]
    for i in vcat(vc.periodic, vc.extended)
        push!(dims, circ.cutoffs[i])
    end
    return dims
end

function _build_numerical_hamiltonian(circ::Circuit)
    sc = circ.symbolic_circuit
    vc = circ.var_categories
    T = circ.transformation_matrix
    T_inv = inv(T)
    n = sc.graph.num_nodes

    active_modes = vcat(vc.periodic, vc.extended)
    n_active = length(active_modes)
    dims = _subsystem_dims(circ)
    total_dim = prod(dims)

    # Build operators for each active mode
    mode_ops = _build_mode_operators(circ, active_modes, dims)

    # 1. Charging energy: H_charge = Σ 4*EC_θ[i,j] * (n_i - ng_i)(n_j - ng_j)
    C_numeric = _build_capacitance_matrix_numeric(circ)
    ec_float = inv(C_numeric) ./ 2
    ec_transformed = T_inv' * ec_float * T_inv

    H = spzeros(ComplexF64, total_dim, total_dim)

    for (ai, mi) in enumerate(active_modes)
        for (aj, mj) in enumerate(active_modes)
            ec_val = ec_transformed[mi, mj]
            abs(ec_val) < 1e-15 && continue

            ni_op = mode_ops[ai].n_op
            nj_op = mode_ops[aj].n_op

            # Apply offset charges for periodic variables
            if mi in vc.periodic
                ng = get(circ.offset_charge_values, mi, 0.0)
                if abs(ng) > 1e-15
                    I_i = _eye_like(ni_op)
                    ni_op = ni_op - ng * I_i
                end
            end
            if mj in vc.periodic
                ng = get(circ.offset_charge_values, mj, 0.0)
                if abs(ng) > 1e-15
                    I_j = _eye_like(nj_op)
                    nj_op = nj_op - ng * I_j
                end
            end

            # Wrap to full space and add
            ni_full = _identity_wrap_sparse(ni_op, ai, dims)
            nj_full = _identity_wrap_sparse(nj_op, aj, dims)
            H .+= 4 * ec_val * ni_full * nj_full
        end
    end

    # 2. Inductive energy: (EL/2)(branch_flux + Φext)²
    #    = (1/2) φ^T L_inv φ  +  EL·Φext·(φ_j - φ_i)  +  (EL/2)·Φext²
    # Part 2a: quadratic (flux-independent) part via L_inv
    L_inv_float = _build_inv_inductance_matrix_numeric(circ)
    L_inv_transformed = T' * L_inv_float * T

    for (ai, mi) in enumerate(active_modes)
        for (aj, mj) in enumerate(active_modes)
            l_val = L_inv_transformed[mi, mj]
            abs(l_val) < 1e-15 && continue

            phi_i = _identity_wrap_sparse(mode_ops[ai].phi_op, ai, dims)
            phi_j = _identity_wrap_sparse(mode_ops[aj].phi_op, aj, dims)
            H .+= 0.5 * l_val * phi_i * phi_j
        end
    end

    # Part 2b: linear and constant terms from inductive external flux
    cg = sc.graph
    for (bi, b) in enumerate(cg.branches)
        b.branch_type == L_branch || continue
        el = _get_branch_param(circ, bi, :EL)
        phi_ext_val = _eval_branch_ext_flux(circ, bi)
        abs(phi_ext_val) < 1e-15 && continue

        # Linear term: EL * Φext * (φ_j - φ_i) in mode-transformed space
        for (ai, mi) in enumerate(active_modes)
            w_k = _branch_mode_coeff(b, T_inv, mi)
            abs(w_k) < 1e-15 && continue
            phi_op = _identity_wrap_sparse(mode_ops[ai].phi_op, ai, dims)
            H .+= el * phi_ext_val * w_k * phi_op
        end

        # Constant term: (EL/2) * Φext²
        I_op = sparse(ComplexF64(1.0) * I, total_dim, total_dim)
        H .+= (el / 2) * phi_ext_val^2 * I_op
    end

    # 3. Josephson terms: -EJ * cos(phase)
    ej_current_vals = _get_josephson_ej_values(circ)
    for (jj_idx, (ej_sym, phase_sym)) in enumerate(sc.josephson_terms)
        ej_val = ej_current_vals[jj_idx]

        # Extract mode coefficients and constant (ext flux) part from symbolic phase
        phase_coeffs, ext_phase = _extract_phase_info(circ, phase_sym, T_inv, n, active_modes)

        cos_op = _build_cos_operator(circ, phase_coeffs, ext_phase,
                                     active_modes, dims, mode_ops)
        H .-= ej_val * cos_op
    end

    return QuantumObject(sparse(H))
end

# ── Mode operator storage ────────────────────────────────────────────────────

struct ModeOperators
    n_op::SparseMatrixCSC{ComplexF64, Int}
    phi_op::SparseMatrixCSC{ComplexF64, Int}
    cos_op::SparseMatrixCSC{ComplexF64, Int}
    sin_op::SparseMatrixCSC{ComplexF64, Int}
    exp_ip_op::SparseMatrixCSC{ComplexF64, Int}   # e^{iθ} or e^{iφ}
end

function _build_mode_operators(circ::Circuit, active_modes::Vector{Int},
                                dims::Vector{Int})
    vc = circ.var_categories
    ops = ModeOperators[]

    for (idx, mode) in enumerate(active_modes)
        dim = dims[idx]
        if mode in vc.periodic
            ncut = (dim - 1) ÷ 2
            n_op = sparse(n_operator_periodic(ncut).data)
            exp_op = sparse(exp_i_theta_operator(ncut).data)
            cos_op = (exp_op + exp_op') / 2
            sin_op = (exp_op - exp_op') / (2im)
            # φ operator not directly available in charge basis; use exp_i_theta
            phi_op = spzeros(ComplexF64, dim, dim)  # placeholder
            push!(ops, ModeOperators(n_op, phi_op, cos_op, sin_op, exp_op))
        else
            # Extended mode
            if circ.ext_basis == :harmonic
                osc_len = get(circ.osc_lengths, mode, 1.0)
                cutoff = dim
                a = sparse(destroy(cutoff).data)
                ad = sparse(a')
                phi_op = osc_len * (a + ad) / sqrt(2)
                n_op = 1im * (ad - a) / (sqrt(2) * osc_len)
                # cos/sin via matrix exponential
                phi_dense = Matrix(phi_op)
                exp_ip = sparse(exp(1im * phi_dense))
                cos_op = (exp_ip + exp_ip') / 2
                sin_op = (exp_ip - exp_ip') / (2im)
                push!(ops, ModeOperators(n_op, phi_op, cos_op, sin_op, exp_ip))
            else
                # Grid basis
                phi_range = circ.phi_grid_ranges[mode]
                grid = Grid1d(phi_range[1], phi_range[2], dim)
                phi_op = sparse(phi_operator_grid(grid).data)
                d2 = sparse(d2_dphi2_operator_grid(grid).data)
                n_op = -d2  # n² ~ -d²/dφ² (up to constants)
                cos_op = sparse(cos_phi_operator_grid(grid).data)
                sin_op = sparse(sin_phi_operator_grid(grid).data)
                exp_ip = cos_op + 1im * sin_op
                push!(ops, ModeOperators(n_op, phi_op, cos_op, sin_op, exp_ip))
            end
        end
    end

    return ops
end

# ── Josephson term evaluation ────────────────────────────────────────────────

"""
Extract mode coefficients and constant (external flux) part of a symbolic phase.

Returns `(coeffs, const_phase)` where:
- `coeffs::Dict{Int,Float64}` maps active mode index → coefficient
- `const_phase::Float64` is the constant (flux-dependent) part of the phase
"""
function _extract_phase_info(circ::Circuit, phase_sym::Num,
                              T_inv::Matrix{Float64}, n::Int,
                              active_modes::Vector{Int})
    sc = circ.symbolic_circuit
    coeffs = Dict{Int, Float64}()

    # Extract coefficient of each node variable by substitution
    for (ai, mi) in enumerate(active_modes)
        coeff = 0.0
        for k in 1:n
            test_vals = Dict(sc.node_vars[j] => (j == k ? 1.0 : 0.0) for j in 1:n)
            for ef in sc.external_fluxes
                test_vals[ef] = 0.0
            end
            c_k = Float64(Symbolics.value(Symbolics.substitute(phase_sym, test_vals)))
            coeff += c_k * T_inv[k, mi]
        end
        if abs(coeff) > 1e-15
            coeffs[ai] = coeff
        end
    end

    # Extract constant (external flux) part: substitute all node vars = 0,
    # external fluxes = their numerical values
    const_vals = Dict{Num, Float64}()
    for j in 1:n
        const_vals[sc.node_vars[j]] = 0.0
    end
    for (fi, ef) in enumerate(sc.external_fluxes)
        const_vals[ef] = fi <= length(circ.external_flux_values) ?
                         circ.external_flux_values[fi] : 0.0
    end
    const_phase = Float64(Symbolics.value(Symbolics.substitute(phase_sym, const_vals)))

    return coeffs, const_phase
end

"""
Build `exp(i * c * θ)` for a single mode, using the best available method.
Returns a sparse matrix in the mode's Hilbert space.
"""
function _build_single_mode_exp(circ::Circuit, mop::ModeOperators,
                                 mode::Int, c::Float64)
    if abs(c) < 1e-15
        dim = size(mop.n_op, 1)
        return sparse(ComplexF64(1.0) * I, dim, dim)
    elseif mode in circ.var_categories.periodic && abs(c) ≈ 1.0
        return c > 0 ? mop.exp_ip_op : sparse(mop.exp_ip_op')
    else
        phi = mop.phi_op
        if iszero(phi)
            return c > 0 ? mop.exp_ip_op : sparse(mop.exp_ip_op')
        else
            return sparse(exp(Matrix(1im * c * phi)))
        end
    end
end

function _build_cos_operator(circ::Circuit, phase_coeffs::Dict{Int, Float64},
                              ext_phase::Float64,
                              active_modes::Vector{Int}, dims::Vector{Int},
                              mode_ops::Vector{ModeOperators})
    n_active = length(active_modes)
    total_dim = prod(dims)

    if length(phase_coeffs) == 0
        # No mode dependence, just constant phase
        return cos(ext_phase) * sparse(I, total_dim, total_dim)
    end

    if length(phase_coeffs) == 1
        # Single-mode: cos(c * θ_i + ext_phase)
        ai, c = first(phase_coeffs)
        mode = active_modes[ai]

        if mode in circ.var_categories.periodic && abs(c) ≈ 1.0
            # Exact: cos(c*θ + Φ) = cos(Φ)cos(θ) - c*sin(Φ)sin(θ)
            cos_θ = mode_ops[ai].cos_op
            sin_θ = mode_ops[ai].sin_op
            op_local = cos(ext_phase) * cos_θ - c * sin(ext_phase) * sin_θ
            return _identity_wrap_sparse(op_local, ai, dims)
        else
            # General case: matrix exponentiation
            phi_op = mode_ops[ai].phi_op
            if iszero(phi_op)
                exp_op = mode_ops[ai].exp_ip_op
                if c > 0
                    full_exp = exp_op * exp(1im * ext_phase)
                else
                    full_exp = exp_op' * exp(-1im * ext_phase)
                end
                op_local = (full_exp + full_exp') / 2
            else
                phase_op = c * phi_op
                exp_op = sparse(exp(Matrix(1im * phase_op))) * exp(1im * ext_phase)
                op_local = (exp_op + exp_op') / 2
            end
            return _identity_wrap_sparse(op_local, ai, dims)
        end
    end

    # Multi-mode: cos(Σ c_i θ_i + ext_phase)
    # e^{i(c1*θ1 + c2*θ2 + ...)} = e^{ic1*θ1} ⊗ e^{ic2*θ2} ⊗ ...
    exp_ops = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, n_active)
    for ai in 1:n_active
        dim = dims[ai]
        c = get(phase_coeffs, ai, 0.0)
        if abs(c) < 1e-15
            exp_ops[ai] = sparse(ComplexF64(1.0) * I, dim, dim)
        elseif active_modes[ai] in circ.var_categories.periodic && abs(c) ≈ 1.0
            exp_ops[ai] = c > 0 ? mode_ops[ai].exp_ip_op : sparse(mode_ops[ai].exp_ip_op')
        else
            phi = mode_ops[ai].phi_op
            if iszero(phi)
                exp_ops[ai] = c > 0 ? mode_ops[ai].exp_ip_op :
                              sparse(mode_ops[ai].exp_ip_op')
            else
                exp_ops[ai] = sparse(exp(Matrix(1im * c * phi)))
            end
        end
    end

    full_exp = reduce(kron, exp_ops) * exp(1im * ext_phase)
    return (full_exp + full_exp') / 2
end

# ── Helper functions ─────────────────────────────────────────────────────────

function _identity_wrap_sparse(op::SparseMatrixCSC{ComplexF64, Int},
                                subsys_idx::Int, dims::Vector{Int})
    n = length(dims)
    result = subsys_idx == 1 ? op : sparse(ComplexF64(1.0) * I, dims[1], dims[1])
    for i in 2:n
        if i == subsys_idx
            result = kron(result, op)
        else
            result = kron(result, sparse(ComplexF64(1.0) * I, dims[i], dims[i]))
        end
    end
    return result
end

function _eye_like(op::SparseMatrixCSC)
    n = size(op, 1)
    return sparse(ComplexF64(1.0) * I, n, n)
end

function _compute_osc_lengths(circ::Circuit)
    T = circ.transformation_matrix
    vc = circ.var_categories
    lengths = Dict{Int, Float64}()

    C_numeric = _build_capacitance_matrix_numeric(circ)
    ec_float = inv(C_numeric) ./ 2
    L_inv_float = _build_inv_inductance_matrix_numeric(circ)
    T_inv = inv(T)

    ec_transformed = T_inv' * ec_float * T_inv
    L_inv_transformed = T' * L_inv_float * T

    for i in vc.extended
        ec_ii = ec_transformed[i, i]
        el_ii = L_inv_transformed[i, i]
        if ec_ii > 0 && el_ii > 0
            # Oscillator length: l_osc = (8 EC / EL)^{1/4}
            lengths[i] = (8 * ec_ii / el_ii)^0.25
        else
            lengths[i] = 1.0
        end
    end

    return lengths
end
