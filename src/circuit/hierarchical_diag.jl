# ──────────────────────────────────────────────────────────────────────────────
# Hierarchical diagonalization for multi-mode circuits
#
# Supports both flat and recursive (nested) hierarchy specifications.
# Partitions circuit modes into subsystems, diagonalizes each independently,
# then builds a HilbertSpace with cross-subsystem couplings.
# ──────────────────────────────────────────────────────────────────────────────

# ── Hierarchy specification types ─────────────────────────────────────────────

"""
    HierarchyLeaf(mode_indices)

A leaf node in a hierarchy specification: a group of actual variable indices
(matching scqubits convention) to be diagonalized together.
"""
struct HierarchyLeaf
    mode_indices::Vector{Int}
end

"""
    HierarchyGroup(children)

A group node in a hierarchy specification. Its children (leaves or sub-groups)
are diagonalized independently, then coupled and re-diagonalized at this level.

For the *top-level* group, the final result is returned as a `HilbertSpace`
(not diagonalized again).  Intermediate groups are diagonalized and truncated.
"""
struct HierarchyGroup
    children::Vector{Union{HierarchyLeaf, HierarchyGroup}}
end

const HierarchyNode = Union{HierarchyLeaf, HierarchyGroup}

# ── HD truncation defaults ───────────────────────────────────────────────────

"""
    truncation_template(system_hierarchy; individual_trunc_dim=6,
                        combined_trunc_dim=30)

Return a scqubits-style hierarchical truncation template for `system_hierarchy`.

For a flat hierarchy such as `[[1], [2], [3]]`, this returns a Julia-friendly
`Dict{Int,Int}` like `Dict(1 => 6, 2 => 6, 3 => 6)`.

For a nested hierarchy such as `[[[1], [2]], [3]]`, this returns a
path-keyed `Dict{Tuple{Vararg{Int}},Int}` such as
`Dict((1,) => 30, (1,1) => 6, (1,2) => 6, (2,) => 6)`.

These defaults mirror scqubits semantics:
- leaf subsystems default to `individual_trunc_dim = 6`
- combined groups default to `combined_trunc_dim = 30`
"""
function truncation_template(system_hierarchy;
                             individual_trunc_dim::Int=6,
                             combined_trunc_dim::Int=30)
    individual_trunc_dim >= 1 || throw(ArgumentError(
        "individual_trunc_dim must be positive, got $individual_trunc_dim"))
    combined_trunc_dim >= 1 || throw(ArgumentError(
        "combined_trunc_dim must be positive, got $combined_trunc_dim"))

    hier = system_hierarchy isa HierarchyGroup ? system_hierarchy :
           system_hierarchy isa HierarchyLeaf ? HierarchyGroup([system_hierarchy]) :
           begin
               node = _to_hierarchy_node(system_hierarchy)
               node isa HierarchyLeaf ? HierarchyGroup([node]) : node
           end

    if all(child -> child isa HierarchyLeaf, hier.children)
        return Dict(idx => individual_trunc_dim for idx in eachindex(hier.children))
    end

    template = Dict{Tuple{Vararg{Int}}, Int}()
    for (idx, child) in enumerate(hier.children)
        _fill_truncation_template!(template, child, (idx,),
                                   individual_trunc_dim, combined_trunc_dim)
    end
    return template
end

function _fill_truncation_template!(template::Dict{Tuple{Vararg{Int}}, Int},
                                    node::HierarchyLeaf,
                                    path::Tuple{Vararg{Int}},
                                    individual_trunc_dim::Int,
                                    combined_trunc_dim::Int)
    template[path] = individual_trunc_dim
    return template
end

function _fill_truncation_template!(template::Dict{Tuple{Vararg{Int}}, Int},
                                    node::HierarchyGroup,
                                    path::Tuple{Vararg{Int}},
                                    individual_trunc_dim::Int,
                                    combined_trunc_dim::Int)
    template[path] = combined_trunc_dim
    for (idx, child) in enumerate(node.children)
        _fill_truncation_template!(template, child, (path..., idx),
                                   individual_trunc_dim, combined_trunc_dim)
    end
    return template
end

# ── SubCircuit type ──────────────────────────────────────────────────────────

"""
    SubCircuit <: AbstractQuantumSystem

A partition of a Circuit's modes treated as an independent quantum system.
Created by [`hierarchical_diag`](@ref) — not intended for direct construction.

The Hamiltonian is represented in the **truncated eigenbasis** of the subsystem,
i.e., it is a diagonal matrix of size `truncated_dim × truncated_dim`.
"""
struct SubCircuit <: AbstractQuantumSystem
    parent::Circuit
    mode_indices::Vector{Int}       # actual variable indices (scqubits convention)
    _hamiltonian::QuantumObject
    _dim::Int                       # truncated_dim
    _eigvals::Vector{Float64}
    _eigvecs::Matrix{ComplexF64}    # full-dim eigenvectors for operator transforms
end

hilbertdim(sc::SubCircuit) = sc._dim
hamiltonian(sc::SubCircuit) = sc._hamiltonian

function eigenvals(sc::SubCircuit; evals_count::Int=sc._dim)
    n = min(evals_count, sc._dim)
    return sc._eigvals[1:n]
end

function eigensys(sc::SubCircuit; evals_count::Int=sc._dim)
    n = min(evals_count, sc._dim)
    return sc._eigvals[1:n], Matrix{ComplexF64}(I, sc._dim, sc._dim)[:, 1:n]
end

# ── Result from diagonalizing one hierarchy level ─────────────────────────────

"""Internal result returned by `_process_node`."""
struct LevelResult
    subcircuit::SubCircuit
    all_mode_indices::Vector{Int}                              # actual variable indices in this subtree
    dressed_n_ops::Dict{Int, SparseMatrixCSC{ComplexF64, Int}}   # ai => dressed n̂
    dressed_phi_ops::Dict{Int, SparseMatrixCSC{ComplexF64, Int}} # ai => dressed φ̂
    dressed_exp_builders::Dict{Int, Function}                    # ai => (c -> dressed exp(icθ))
end

# ── Conversion helpers ───────────────────────────────────────────────────────

"""Convert a nested `Vector` structure to `HierarchyNode`."""
function _to_hierarchy_node(v::Vector{Int})
    return HierarchyLeaf(v)
end

function _to_hierarchy_node(v::Vector{Vector{Int}})
    # Flat: each element is a leaf
    return HierarchyGroup([HierarchyLeaf(g) for g in v])
end

function _to_hierarchy_node(v::Vector)
    # General nested case: each element may be Vector{Int} (leaf) or nested (group)
    children = Union{HierarchyLeaf, HierarchyGroup}[]
    for item in v
        push!(children, _to_hierarchy_node(item))
    end
    return HierarchyGroup(children)
end

"""Collect all mode indices from a hierarchy node."""
function _collect_modes(leaf::HierarchyLeaf)
    return copy(leaf.mode_indices)
end

function _collect_modes(group::HierarchyGroup)
    return reduce(vcat, [_collect_modes(c) for c in group.children])
end

# ── Shared context for all hierarchy levels ──────────────────────────────────

"""Immutable context shared across recursive hierarchy processing."""
struct HierarchyContext
    circ::Circuit
    active_modes::Vector{Int}
    dims::Vector{Int}
    mode_ops::Vector{ModeOperators}
    ec_transformed::Matrix{Float64}
    L_inv_transformed::Matrix{Float64}
    T_inv::Matrix{Float64}
    n_nodes::Int
    var_to_pos::Dict{Int,Int}           # variable index → position in active_modes/dims/mode_ops
end

# ── Leaf diagonalization ─────────────────────────────────────────────────────

function _process_leaf(ctx::HierarchyContext, leaf::HierarchyLeaf, trunc_dim::Int)
    circ = ctx.circ
    sc = circ.symbolic_circuit
    vc = circ.var_categories
    group = leaf.mode_indices           # actual variable indices (scqubits convention)
    sub_dims = [ctx.dims[ctx.var_to_pos[vi]] for vi in group]
    sub_total = prod(sub_dims)

    # ── Build ng-shifted n operators for periodic modes ────────────────────
    shifted_n_ops = Dict{Int, SparseMatrixCSC{ComplexF64, Int}}()
    for vi in group
        pos = ctx.var_to_pos[vi]
        n_op = ctx.mode_ops[pos].n_op
        if vi in vc.periodic
            ng = get(circ.offset_charge_values, vi, 0.0)
            if abs(ng) > 1e-15
                n_op = n_op - ng * _eye_like(n_op)
            end
        end
        shifted_n_ops[vi] = n_op
    end

    # ── Build sub-Hamiltonian ──────────────────────────────────────────────
    H_sub = spzeros(ComplexF64, sub_total, sub_total)

    # Charging energy (intra-group): 4*EC*(n_i - ng_i)*(n_j - ng_j)
    for (li, vi) in enumerate(group)
        for (lj, vj) in enumerate(group)
            ec_val = ctx.ec_transformed[vi, vj]
            abs(ec_val) < 1e-15 && continue
            ni_full = _identity_wrap_sparse(shifted_n_ops[vi], li, sub_dims)
            nj_full = _identity_wrap_sparse(shifted_n_ops[vj], lj, sub_dims)
            H_sub .+= 4 * ec_val * ni_full * nj_full
        end
    end

    # Inductive energy (intra-group): quadratic part
    for (li, vi) in enumerate(group)
        for (lj, vj) in enumerate(group)
            l_val = ctx.L_inv_transformed[vi, vj]
            abs(l_val) < 1e-15 && continue
            phi_i = _identity_wrap_sparse(ctx.mode_ops[ctx.var_to_pos[vi]].phi_op, li, sub_dims)
            phi_j = _identity_wrap_sparse(ctx.mode_ops[ctx.var_to_pos[vj]].phi_op, lj, sub_dims)
            H_sub .+= 0.5 * l_val * phi_i * phi_j
        end
    end

    # Inductive external flux: linear terms for modes in this leaf
    cg = sc.graph
    for (bi, b) in enumerate(cg.branches)
        b.branch_type == L_branch || continue
        el = _get_branch_param(circ, bi, :EL)
        phi_ext_val = _eval_branch_ext_flux(circ, bi)
        abs(phi_ext_val) < 1e-15 && continue
        for (li, vi) in enumerate(group)
            w_k = _branch_mode_coeff(b, ctx.T_inv, vi)
            abs(w_k) < 1e-15 && continue
            pos = ctx.var_to_pos[vi]
            phi_op = _identity_wrap_sparse(ctx.mode_ops[pos].phi_op, li, sub_dims)
            H_sub .+= el * phi_ext_val * w_k * phi_op
        end
    end

    # Josephson terms (intra-group only)
    ej_current_vals = _get_josephson_ej_values(circ)
    for (jj_idx, (ej_sym, phase_sym)) in enumerate(sc.josephson_terms)
        ej_val = ej_current_vals[jj_idx]
        phase_coeffs, ext_phase = _extract_phase_info(circ, phase_sym, ctx.T_inv,
                                                       ctx.n_nodes, ctx.active_modes)
        # Re-key from active-mode position to variable index
        var_coeffs = Dict(ctx.active_modes[ai] => c for (ai, c) in phase_coeffs)
        involved_modes = collect(keys(var_coeffs))
        if all(m -> m in group, involved_modes)
            local_coeffs = Dict{Int,Float64}()
            for (vi, c) in var_coeffs
                li = findfirst(==(vi), group)
                local_coeffs[li] = c
            end
            local_mode_ops = [ctx.mode_ops[ctx.var_to_pos[vi]] for vi in group]
            cos_op = _build_cos_operator(circ, local_coeffs, ext_phase,
                                          group, sub_dims, local_mode_ops)
            H_sub .-= ej_val * cos_op
        end
    end

    # ── Diagonalize ────────────────────────────────────────────────────────
    H_qobj = QuantumObject(sparse(H_sub))
    result = eigenstates(H_qobj)
    n_states = min(trunc_dim, length(result.values))
    evals = real.(result.values[1:n_states])
    evecs = result.vectors[:, 1:n_states]

    # SubCircuit Hamiltonian: diagonal in truncated eigenbasis
    H_diag = QuantumObject(spdiagm(0 => ComplexF64.(evals)))
    subcircuit = SubCircuit(circ, group, H_diag, n_states, evals, evecs)

    # ── Dressed operators (ng-shifted n̂ for correct cross-coupling) ───────
    dressed_n = Dict{Int, SparseMatrixCSC{ComplexF64, Int}}()
    dressed_phi = Dict{Int, SparseMatrixCSC{ComplexF64, Int}}()
    dressed_exp = Dict{Int, Function}()

    for (li, vi) in enumerate(group)
        pos = ctx.var_to_pos[vi]
        n_full = _identity_wrap_sparse(shifted_n_ops[vi], li, sub_dims)
        phi_full = _identity_wrap_sparse(ctx.mode_ops[pos].phi_op, li, sub_dims)
        dressed_n[vi] = sparse(evecs' * n_full * evecs)
        dressed_phi[vi] = sparse(evecs' * phi_full * evecs)

        # Closure: build dressed exp(icθ) for arbitrary coefficient c
        let mop = ctx.mode_ops[pos], _vi = vi,
            _li = li, _sub_dims = sub_dims, _evecs = evecs
            dressed_exp[vi] = function(c)
                bare_exp = _build_single_mode_exp(circ, mop, _vi, c)
                exp_full = _identity_wrap_sparse(bare_exp, _li, _sub_dims)
                return sparse(_evecs' * exp_full * _evecs)
            end
        end
    end

    return LevelResult(subcircuit, group, dressed_n, dressed_phi, dressed_exp)
end

# ── Intermediate group diagonalization ───────────────────────────────────────

function _process_intermediate_group(ctx::HierarchyContext,
                                      group::HierarchyGroup,
                                      trunc_dim::Int,
                                      path::Tuple{Vararg{Int}},
                                      trunc_dims::Dict{Tuple{Vararg{Int}}, Int})
    # ── Recursively process children ───────────────────────────────────────
    child_results = LevelResult[]
    for (ci, child) in enumerate(group.children)
        child_path = (path..., ci)
        r = _process_node(ctx, child, child_path, trunc_dims)
        push!(child_results, r)
    end

    n_children = length(child_results)
    child_dims = [hilbertdim(r.subcircuit) for r in child_results]
    combined_dim = prod(child_dims)

    # ── Build combined Hamiltonian ─────────────────────────────────────────
    H_combined = spzeros(ComplexF64, combined_dim, combined_dim)

    # Bare sub-Hamiltonians
    for (ci, r) in enumerate(child_results)
        H_i = sparse(r.subcircuit._hamiltonian.data)
        H_combined .+= _identity_wrap_sparse(H_i, ci, child_dims)
    end

    # Cross-child capacitive couplings (factor 2 for symmetric sum)
    for cA in 1:n_children
        for cB in (cA+1):n_children
            for viA in child_results[cA].all_mode_indices
                for viB in child_results[cB].all_mode_indices
                    ec_val = ctx.ec_transformed[viA, viB]
                    abs(ec_val) < 1e-15 && continue

                    nA = _identity_wrap_sparse(child_results[cA].dressed_n_ops[viA],
                                               cA, child_dims)
                    nB = _identity_wrap_sparse(child_results[cB].dressed_n_ops[viB],
                                               cB, child_dims)
                    H_combined .+= 2 * 4 * ec_val * nA * nB
                end
            end
        end
    end

    # Cross-child inductive couplings (factor 2 for symmetric sum)
    for cA in 1:n_children
        for cB in (cA+1):n_children
            for viA in child_results[cA].all_mode_indices
                for viB in child_results[cB].all_mode_indices
                    l_val = ctx.L_inv_transformed[viA, viB]
                    abs(l_val) < 1e-15 && continue

                    phiA = _identity_wrap_sparse(child_results[cA].dressed_phi_ops[viA],
                                                  cA, child_dims)
                    phiB = _identity_wrap_sparse(child_results[cB].dressed_phi_ops[viB],
                                                  cB, child_dims)
                    H_combined .+= 2 * 0.5 * l_val * phiA * phiB
                end
            end
        end
    end

    # Cross-child Josephson couplings
    all_modes_here = reduce(vcat, [r.all_mode_indices for r in child_results])
    ej_current_vals = _get_josephson_ej_values(ctx.circ)
    sc_sym = ctx.circ.symbolic_circuit
    for (jj_idx, (ej_sym, phase_sym)) in enumerate(sc_sym.josephson_terms)
        ej_val = ej_current_vals[jj_idx]
        phase_coeffs, ext_phase = _extract_phase_info(ctx.circ, phase_sym, ctx.T_inv,
                                                       ctx.n_nodes, ctx.active_modes)
        # Re-key from active-mode position to variable index
        var_coeffs = Dict(ctx.active_modes[ai] => c for (ai, c) in phase_coeffs)
        # Map modes to children
        children_involved = Dict{Int, Vector{Tuple{Int, Float64}}}()
        for (vi, c) in var_coeffs
            vi in all_modes_here || continue
            ci = findfirst(r -> vi in r.all_mode_indices, child_results)
            ci === nothing && continue
            if !haskey(children_involved, ci)
                children_involved[ci] = Tuple{Int, Float64}[]
            end
            push!(children_involved[ci], (vi, c))
        end

        # Skip if intra-child (already handled) or involves modes outside this group
        length(children_involved) <= 1 && continue

        # Build per-child exp operators
        dressed_exp_per_child = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, n_children)
        for ci in 1:n_children
            d = child_dims[ci]
            if !haskey(children_involved, ci)
                dressed_exp_per_child[ci] = sparse(ComplexF64(1.0) * I, d, d)
            else
                child_exp = sparse(ComplexF64(1.0) * I, d, d)
                for (vi, c) in children_involved[ci]
                    child_exp = child_exp * child_results[ci].dressed_exp_builders[vi](c)
                end
                dressed_exp_per_child[ci] = child_exp
            end
        end

        full_exp = reduce(kron, dressed_exp_per_child) * exp(1im * ext_phase)
        cos_op_full = (full_exp + full_exp') / 2
        H_combined .-= ej_val * cos_op_full
    end

    # ── Diagonalize combined system ────────────────────────────────────────
    H_qobj = QuantumObject(sparse(H_combined))
    result = eigenstates(H_qobj)
    n_states = min(trunc_dim, length(result.values))
    evals = real.(result.values[1:n_states])
    evecs = result.vectors[:, 1:n_states]

    # SubCircuit
    H_diag = QuantumObject(spdiagm(0 => ComplexF64.(evals)))
    subcircuit = SubCircuit(ctx.circ, all_modes_here, H_diag, n_states, evals, evecs)

    # ── Compute new dressed operators ──────────────────────────────────────
    new_dressed_n = Dict{Int, SparseMatrixCSC{ComplexF64, Int}}()
    new_dressed_phi = Dict{Int, SparseMatrixCSC{ComplexF64, Int}}()
    new_dressed_exp = Dict{Int, Function}()

    for (ci, r) in enumerate(child_results)
        for vi in r.all_mode_indices
            n_wrapped = _identity_wrap_sparse(r.dressed_n_ops[vi], ci, child_dims)
            phi_wrapped = _identity_wrap_sparse(r.dressed_phi_ops[vi], ci, child_dims)
            new_dressed_n[vi] = sparse(evecs' * n_wrapped * evecs)
            new_dressed_phi[vi] = sparse(evecs' * phi_wrapped * evecs)

            let old_builder = r.dressed_exp_builders[vi],
                _ci = ci, _child_dims = child_dims, _evecs = evecs
                new_dressed_exp[vi] = function(c)
                    old_exp = old_builder(c)
                    exp_wrapped = _identity_wrap_sparse(old_exp, _ci, _child_dims)
                    return sparse(_evecs' * exp_wrapped * _evecs)
                end
            end
        end
    end

    return LevelResult(subcircuit, all_modes_here, new_dressed_n, new_dressed_phi,
                       new_dressed_exp)
end

# ── Node dispatcher ──────────────────────────────────────────────────────────

function _process_node(ctx::HierarchyContext, node::HierarchyLeaf,
                       path::Tuple{Vararg{Int}},
                       trunc_dims::Dict{Tuple{Vararg{Int}}, Int})
    td = trunc_dims[path]
    return _process_leaf(ctx, node, td)
end

function _process_node(ctx::HierarchyContext, node::HierarchyGroup,
                       path::Tuple{Vararg{Int}},
                       trunc_dims::Dict{Tuple{Vararg{Int}}, Int})
    td = trunc_dims[path]
    return _process_intermediate_group(ctx, node, td, path, trunc_dims)
end

# ── Top-level entry point (new recursive API) ────────────────────────────────

"""
    hierarchical_diag(circ::Circuit;
                      system_hierarchy,
                      subsystem_trunc_dims=nothing) -> HilbertSpace

Partition circuit modes into subsystems and build a HilbertSpace using
hierarchical diagonalization.

# Flat hierarchy (backward-compatible)

```julia
hs = hierarchical_diag(circ;
    system_hierarchy  = [[1], [2]],
    subsystem_trunc_dims = Dict(1=>10, 2=>10))
```

Each `Vector{Int}` is a leaf group of **actual variable indices** (scqubits
convention), not positions in the active-modes array.
`subsystem_trunc_dims::Dict{Int,Int}` maps group index → truncation dimension.

# Nested hierarchy

```julia
hs = hierarchical_diag(circ;
    system_hierarchy  = [[[1], [2]], [3]],
    subsystem_trunc_dims = Dict(
        (1,1) => 10,   # leaf [1]
        (1,2) => 10,   # leaf [2]
        (1,)  => 15,   # intermediate group [[1],[2]]
        (2,)  => 10))  # leaf [3]
```

Keys in `subsystem_trunc_dims` are path tuples identifying each node.
Intermediate groups are re-diagonalized and truncated; the top level is not.

# Returns
A `HilbertSpace` whose subsystems are truncated `SubCircuit`s with
cross-coupling interaction terms.
"""
function hierarchical_diag(circ::Circuit; system_hierarchy, subsystem_trunc_dims=nothing)
    # ── Normalize inputs ───────────────────────────────────────────────────
    if system_hierarchy isa HierarchyGroup
        hier = system_hierarchy
    else
        hier = _to_hierarchy_node(system_hierarchy)
        # Ensure we have a HierarchyGroup at the top level
        if hier isa HierarchyLeaf
            hier = HierarchyGroup([hier])
        end
    end

    resolved_trunc_dims = subsystem_trunc_dims === nothing ?
                          truncation_template(system_hierarchy) :
                          subsystem_trunc_dims

    if resolved_trunc_dims isa Dict{Int, Int} ||
       (resolved_trunc_dims isa Dict && keytype(resolved_trunc_dims) == Int)
        td = Dict{Tuple{Vararg{Int}}, Int}((k,) => v
                                            for (k, v) in resolved_trunc_dims)
    else
        td = Dict{Tuple{Vararg{Int}}, Int}(resolved_trunc_dims)
    end
    sc = circ.symbolic_circuit
    vc = circ.var_categories
    T = circ.transformation_matrix
    T_inv = inv(T)
    n_nodes = sc.graph.num_nodes
    active_modes = vcat(vc.periodic, vc.extended)
    n_active = length(active_modes)

    # ── Validate hierarchy ─────────────────────────────────────────────────
    all_modes = sort(_collect_modes(hier))
    expected = sort(active_modes)
    all_modes == expected ||
        error("system_hierarchy must partition active variable indices $expected exactly. " *
              "Got: $all_modes")

    # ── Transformed matrices ───────────────────────────────────────────────
    C_numeric = _build_capacitance_matrix_numeric(circ)
    ec_float = inv(C_numeric) ./ 2
    ec_transformed = T_inv' * ec_float * T_inv

    L_inv_float = _build_inv_inductance_matrix_numeric(circ)
    L_inv_transformed = T' * L_inv_float * T

    dims = Int[circ.cutoffs[m] for m in active_modes]
    mode_ops = _build_mode_operators(circ, active_modes, dims)

    var_to_pos = Dict(mi => ai for (ai, mi) in enumerate(active_modes))
    ctx = HierarchyContext(circ, active_modes, dims, mode_ops,
                           ec_transformed, L_inv_transformed, T_inv, n_nodes,
                           var_to_pos)

    # ── Process top-level children ─────────────────────────────────────────
    child_results = LevelResult[]
    for (ci, child) in enumerate(hier.children)
        child_path = (ci,)
        r = _process_node(ctx, child, child_path, td)
        push!(child_results, r)
    end

    n_children = length(child_results)
    subcircuits = [r.subcircuit for r in child_results]

    # ── Build HilbertSpace ─────────────────────────────────────────────────
    hs = HilbertSpace(collect(AbstractQuantumSystem, subcircuits))
    dims_trunc = [hilbertdim(sc_i) for sc_i in subcircuits]

    # ── Cross-child capacitive couplings (factor 2 for symmetric sum) ──────
    for cA in 1:n_children
        for cB in (cA+1):n_children
            for viA in child_results[cA].all_mode_indices
                for viB in child_results[cB].all_mode_indices
                    ec_val = ec_transformed[viA, viB]
                    abs(ec_val) < 1e-15 && continue

                    nA_qobj = QuantumObject(sparse(child_results[cA].dressed_n_ops[viA]))
                    nB_qobj = QuantumObject(sparse(child_results[cB].dressed_n_ops[viB]))
                    scA = subcircuits[cA]
                    scB = subcircuits[cB]

                    add_interaction!(hs, 2 * 4 * ec_val,
                        [scA, scB],
                        [_ -> nA_qobj, _ -> nB_qobj])
                end
            end
        end
    end

    # ── Cross-child inductive couplings (factor 2 for symmetric sum) ───────
    for cA in 1:n_children
        for cB in (cA+1):n_children
            for viA in child_results[cA].all_mode_indices
                for viB in child_results[cB].all_mode_indices
                    l_val = L_inv_transformed[viA, viB]
                    abs(l_val) < 1e-15 && continue

                    phiA_qobj = QuantumObject(sparse(child_results[cA].dressed_phi_ops[viA]))
                    phiB_qobj = QuantumObject(sparse(child_results[cB].dressed_phi_ops[viB]))
                    scA = subcircuits[cA]
                    scB = subcircuits[cB]

                    add_interaction!(hs, 2 * 0.5 * l_val,
                        [scA, scB],
                        [_ -> phiA_qobj, _ -> phiB_qobj])
                end
            end
        end
    end

    # ── Cross-child Josephson couplings ────────────────────────────────────
    ej_current_vals = _get_josephson_ej_values(circ)
    for (jj_idx, (ej_sym, phase_sym)) in enumerate(sc.josephson_terms)
        ej_val = ej_current_vals[jj_idx]
        phase_coeffs, ext_phase = _extract_phase_info(circ, phase_sym, T_inv,
                                                       n_nodes, active_modes)
        # Re-key from active-mode position to variable index
        var_coeffs = Dict(active_modes[ai] => c for (ai, c) in phase_coeffs)
        # Map modes to top-level children
        children_involved = Dict{Int, Vector{Tuple{Int, Float64}}}()
        for (vi, c) in var_coeffs
            ci = findfirst(r -> vi in r.all_mode_indices, child_results)
            ci === nothing && continue
            if !haskey(children_involved, ci)
                children_involved[ci] = Tuple{Int, Float64}[]
            end
            push!(children_involved[ci], (vi, c))
        end

        length(children_involved) <= 1 && continue

        # Build per-child exp operators
        dressed_exp_per_child = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, n_children)
        for ci in 1:n_children
            d = dims_trunc[ci]
            if !haskey(children_involved, ci)
                dressed_exp_per_child[ci] = sparse(ComplexF64(1.0) * I, d, d)
            else
                child_exp = sparse(ComplexF64(1.0) * I, d, d)
                for (vi, c) in children_involved[ci]
                    child_exp = child_exp * child_results[ci].dressed_exp_builders[vi](c)
                end
                dressed_exp_per_child[ci] = child_exp
            end
        end

        full_exp = reduce(kron, dressed_exp_per_child) * exp(1im * ext_phase)
        cos_op_full = (full_exp + full_exp') / 2

        add_operator!(hs, QuantumObject(sparse(-ej_val * cos_op_full),
                                        dims=Tuple(dims_trunc)))
    end

    # ── Inductive external flux: constant energy shift ─────────────────────
    cg = sc.graph
    total_hs_dim = prod(dims_trunc)
    for (bi, b) in enumerate(cg.branches)
        b.branch_type == L_branch || continue
        el = _get_branch_param(circ, bi, :EL)
        phi_ext_val = _eval_branch_ext_flux(circ, bi)
        abs(phi_ext_val) < 1e-15 && continue
        shift = (el / 2) * phi_ext_val^2
        I_full = sparse(ComplexF64(1.0) * I, total_hs_dim, total_hs_dim)
        add_operator!(hs, QuantumObject(shift * I_full, dims=Tuple(dims_trunc)))
    end

    return hs
end
