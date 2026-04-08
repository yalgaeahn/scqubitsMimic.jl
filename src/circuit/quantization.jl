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
    offset_charge_values::Vector{Float64}

    # Cache
    _hamiltonian_cache::Union{Nothing, QuantumObject}
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

    # Compute oscillator lengths for harmonic basis
    osc_lengths = _compute_osc_lengths(sc, T, vc)

    # External fluxes: default to 0
    n_ext = length(sc.external_fluxes)
    ext_vals = length(external_fluxes) == n_ext ? external_fluxes : zeros(n_ext)

    # Offset charges: default to 0
    n_ng = length(sc.offset_charges)
    ng_vals = length(offset_charges) == n_ng ? offset_charges : zeros(n_ng)

    return Circuit(sc, T, vc, H_mode,
                   cutoffs, ext_basis, phi_ranges, osc_lengths,
                   ext_vals, ng_vals, nothing)
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

"""Invalidate cached Hamiltonian (call after changing parameters)."""
function invalidate_cache!(circ::Circuit)
    circ._hamiltonian_cache = nothing
end

# ── Symbolic accessors ──────────────────────────────────────────────────────

"""Return the symbolic Hamiltonian expression (in mode variables)."""
sym_hamiltonian(circ::Circuit) = circ.mode_hamiltonian_symbolic

"""Return the symbolic Hamiltonian expression (in node variables)."""
sym_hamiltonian_node(circ::Circuit) = circ.symbolic_circuit.hamiltonian_symbolic

"""Return `(T, var_categories)` — the variable transformation from node to mode basis."""
variable_transformation(circ::Circuit) = (circ.transformation_matrix, circ.var_categories)

"""Return the symbolic external flux variables."""
external_fluxes(circ::Circuit) = circ.symbolic_circuit.external_fluxes

"""Return the symbolic offset charge variables."""
offset_charges(circ::Circuit) = circ.symbolic_circuit.offset_charges

# ── Parameter setters ────────────────────────────────────────────────────────

"""Set external flux value. `index` is 1-based."""
function set_external_flux!(circ::Circuit, index::Int, value::Float64)
    circ.external_flux_values[index] = value
    invalidate_cache!(circ)
end

"""Set offset charge value for mode `index`."""
function set_offset_charge!(circ::Circuit, index::Int, value::Float64)
    circ.offset_charge_values[index] = value
    invalidate_cache!(circ)
end

# ── set_param! / get_param for parameter sweeps ─────────────────────────────

function set_param!(circ::Circuit, param_name::Symbol, val)
    if param_name == :flux || param_name == :Φext
        # Shortcut: set first external flux (common for single-loop circuits)
        set_external_flux!(circ, 1, Float64(val))
    elseif startswith(string(param_name), "Φext_")
        idx = parse(Int, string(param_name)[6:end])
        set_external_flux!(circ, idx, Float64(val))
    elseif param_name == :ng
        set_offset_charge!(circ, 1, Float64(val))
    elseif startswith(string(param_name), "ng_")
        idx = parse(Int, string(param_name)[4:end])
        set_offset_charge!(circ, idx, Float64(val))
    else
        error("Circuit parameter sweep for :$param_name not yet supported. " *
              "Use :flux, :Φext, :Φext_N, :ng, :ng_N")
    end
end

function get_param(circ::Circuit, param_name::Symbol)
    if param_name == :flux || param_name == :Φext
        return circ.external_flux_values[1]
    elseif startswith(string(param_name), "Φext_")
        idx = parse(Int, string(param_name)[6:end])
        return circ.external_flux_values[idx]
    elseif param_name == :ng
        return circ.offset_charge_values[1]
    elseif startswith(string(param_name), "ng_")
        idx = parse(Int, string(param_name)[4:end])
        return circ.offset_charge_values[idx]
    else
        error("Circuit parameter :$param_name not recognized")
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
    ec_float = Float64.(Symbolics.value.(sc.ec_matrix))
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
                idx_p = findfirst(==(mi), vc.periodic)
                if idx_p !== nothing && idx_p <= length(circ.offset_charge_values)
                    ng = circ.offset_charge_values[idx_p]
                    if abs(ng) > 1e-15
                        I_i = _eye_like(ni_op)
                        ni_op = ni_op - ng * I_i
                    end
                end
            end
            if mj in vc.periodic
                idx_p = findfirst(==(mj), vc.periodic)
                if idx_p !== nothing && idx_p <= length(circ.offset_charge_values)
                    ng = circ.offset_charge_values[idx_p]
                    if abs(ng) > 1e-15
                        I_j = _eye_like(nj_op)
                        nj_op = nj_op - ng * I_j
                    end
                end
            end

            # Wrap to full space and add
            ni_full = _identity_wrap_sparse(ni_op, ai, dims)
            nj_full = _identity_wrap_sparse(nj_op, aj, dims)
            H .+= 4 * ec_val * ni_full * nj_full
        end
    end

    # 2. Inductive energy: H_ind = (1/2) Σ L_inv_θ[i,j] * φ_i * φ_j
    L_inv_float = Float64.(Symbolics.value.(sc.inv_inductance_matrix))
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

    # 3. Josephson terms: -EJ * cos(phase)
    for (ej_sym, phase_sym) in sc.josephson_terms
        ej_val = Float64(Symbolics.value(ej_sym))

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

function _compute_osc_lengths(sc::SymbolicCircuit, T::Matrix{Float64},
                               vc::VarCategories)
    lengths = Dict{Int, Float64}()
    n = sc.graph.num_nodes

    ec_float = Float64.(Symbolics.value.(sc.ec_matrix))
    L_inv_float = Float64.(Symbolics.value.(sc.inv_inductance_matrix))
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
