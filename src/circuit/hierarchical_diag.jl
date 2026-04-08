# ──────────────────────────────────────────────────────────────────────────────
# Hierarchical diagonalization for multi-mode circuits
#
# Partitions circuit modes into subsystems, diagonalizes each independently,
# then builds a HilbertSpace with cross-subsystem capacitive couplings.
# ──────────────────────────────────────────────────────────────────────────────

"""
    SubCircuit <: AbstractQuantumSystem

A partition of a Circuit's modes treated as an independent quantum system.
Created by [`hierarchical_diag`](@ref) — not intended for direct construction.

The Hamiltonian is represented in the **truncated eigenbasis** of the subsystem,
i.e., it is a diagonal matrix of size `truncated_dim × truncated_dim`.
"""
struct SubCircuit <: AbstractQuantumSystem
    parent::Circuit
    mode_indices::Vector{Int}       # indices into parent's active_modes list
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

"""
    hierarchical_diag(circ::Circuit;
                      system_hierarchy::Vector{Vector{Int}},
                      subsystem_trunc_dims::Dict{Int,Int}) -> HilbertSpace

Partition circuit modes into subsystems and build a HilbertSpace using
hierarchical diagonalization.

Each group of modes is diagonalized independently and truncated to the
specified number of eigenstates. Cross-subsystem capacitive couplings
are added as interaction terms.

# Arguments
- `circ` — the full Circuit
- `system_hierarchy` — groups of mode indices (1-based into active modes list).
  E.g., `[[1], [2]]` puts mode 1 in subsystem 1 and mode 2 in subsystem 2.
- `subsystem_trunc_dims` — truncation per group.
  E.g., `Dict(1=>10, 2=>5)` keeps 10 states for group 1 and 5 for group 2.

# Returns
A `HilbertSpace` whose subsystems are truncated `SubCircuit`s with
capacitive cross-coupling interaction terms.

# Limitations
Only capacitive (charging energy) cross-coupling is implemented.
Josephson terms spanning multiple subsystems generate a warning.

# Example
```julia
circ = Circuit(\"\"\"
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 2, EJ=8.0, EC=0.4]
  - [C, 1, 2, EC=0.1]
\"\"\"; ncut=15)
hs = hierarchical_diag(circ;
    system_hierarchy=[[1], [2]],
    subsystem_trunc_dims=Dict(1=>8, 2=>8))
```
"""
function hierarchical_diag(circ::Circuit;
                           system_hierarchy::Vector{Vector{Int}},
                           subsystem_trunc_dims::Dict{Int,Int})
    sc = circ.symbolic_circuit
    vc = circ.var_categories
    T = circ.transformation_matrix
    T_inv = inv(T)
    n = sc.graph.num_nodes
    active_modes = vcat(vc.periodic, vc.extended)
    n_active = length(active_modes)

    # ── Validate hierarchy ──────────────────────────────────────────────────
    all_modes = reduce(vcat, system_hierarchy)
    sort!(all_modes)
    expected = collect(1:n_active)
    all_modes == expected ||
        error("system_hierarchy must partition active modes 1:$n_active exactly. " *
              "Got: $all_modes, expected: $expected")

    n_groups = length(system_hierarchy)
    for k in 1:n_groups
        haskey(subsystem_trunc_dims, k) ||
            error("subsystem_trunc_dims must have an entry for group $k")
    end

    # ── Transformed matrices (using current branch parameters) ──────────────
    C_numeric = _build_capacitance_matrix_numeric(circ)
    ec_float = inv(C_numeric) ./ 2
    ec_transformed = T_inv' * ec_float * T_inv

    L_inv_float = _build_inv_inductance_matrix_numeric(circ)
    L_inv_transformed = T' * L_inv_float * T

    # ── Build sub-Hamiltonians and diagonalize ──────────────────────────────
    dims = Int[circ.cutoffs[m] for m in active_modes]
    mode_ops = _build_mode_operators(circ, active_modes, dims)

    subcircuits = SubCircuit[]
    group_eigvecs = Matrix{ComplexF64}[]   # full-space eigenvectors per group
    group_n_ops = Vector{Vector{SparseMatrixCSC{ComplexF64,Int}}}()  # n operators per group

    for (gk, group) in enumerate(system_hierarchy)
        trunc_dim = subsystem_trunc_dims[gk]
        sub_dims = [dims[ai] for ai in group]
        sub_total = prod(sub_dims)

        # Build sub-Hamiltonian
        H_sub = spzeros(ComplexF64, sub_total, sub_total)

        # Charging energy (intra-group)
        for (li, ai) in enumerate(group)
            mi = active_modes[ai]
            for (lj, aj) in enumerate(group)
                mj = active_modes[aj]
                ec_val = ec_transformed[mi, mj]
                abs(ec_val) < 1e-15 && continue

                ni_full = _identity_wrap_sparse(mode_ops[ai].n_op, li, sub_dims)
                nj_full = _identity_wrap_sparse(mode_ops[aj].n_op, lj, sub_dims)
                H_sub .+= 4 * ec_val * ni_full * nj_full
            end
        end

        # Inductive energy (intra-group)
        for (li, ai) in enumerate(group)
            mi = active_modes[ai]
            for (lj, aj) in enumerate(group)
                mj = active_modes[aj]
                l_val = L_inv_transformed[mi, mj]
                abs(l_val) < 1e-15 && continue

                phi_i = _identity_wrap_sparse(mode_ops[ai].phi_op, li, sub_dims)
                phi_j = _identity_wrap_sparse(mode_ops[aj].phi_op, lj, sub_dims)
                H_sub .+= 0.5 * l_val * phi_i * phi_j
            end
        end

        # Josephson terms (intra-group only)
        ej_current_vals = _get_josephson_ej_values(circ)
        for (jj_idx, (ej_sym, phase_sym)) in enumerate(sc.josephson_terms)
            ej_val = ej_current_vals[jj_idx]
            phase_coeffs, ext_phase = _extract_phase_info(circ, phase_sym, T_inv, n, active_modes)

            # Check if this Josephson term only involves modes in this group
            involved_modes = collect(keys(phase_coeffs))
            if all(m -> m in group, involved_modes)
                # Remap mode indices to local sub-indices
                local_coeffs = Dict{Int,Float64}()
                for (ai, c) in phase_coeffs
                    li = findfirst(==(ai), group)
                    local_coeffs[li] = c
                end
                local_mode_ops = [mode_ops[ai] for ai in group]
                cos_op = _build_cos_operator(circ, local_coeffs, ext_phase,
                                              [active_modes[ai] for ai in group],
                                              sub_dims, local_mode_ops)
                H_sub .-= ej_val * cos_op
            elseif !isempty(intersect(Set(involved_modes), Set(group)))
                @warn "Josephson term spans multiple subsystem groups — " *
                      "cross-group Josephson coupling is not yet supported in hierarchical_diag"
            end
        end

        # Diagonalize
        H_qobj = QuantumObject(sparse(H_sub))
        result = eigenstates(H_qobj)
        n_states = min(trunc_dim, length(result.values))
        evals = real.(result.values[1:n_states])
        evecs = result.vectors[:, 1:n_states]

        # SubCircuit Hamiltonian: diagonal in truncated eigenbasis
        H_diag = QuantumObject(spdiagm(0 => ComplexF64.(evals)))

        push!(subcircuits, SubCircuit(circ, group, H_diag, n_states, evals, evecs))
        push!(group_eigvecs, evecs)

        # Store n operators in the sub-Hilbert space for coupling
        n_ops_group = SparseMatrixCSC{ComplexF64,Int}[]
        for (li, ai) in enumerate(group)
            n_full = _identity_wrap_sparse(mode_ops[ai].n_op, li, sub_dims)
            push!(n_ops_group, n_full)
        end
        push!(group_n_ops, n_ops_group)
    end

    # ── Build HilbertSpace ──────────────────────────────────────────────────
    hs = HilbertSpace(collect(AbstractQuantumSystem, subcircuits))

    # ── Add cross-group capacitive couplings ────────────────────────────────
    for gA in 1:n_groups
        for gB in (gA+1):n_groups
            for (liA, aiA) in enumerate(system_hierarchy[gA])
                miA = active_modes[aiA]
                for (liB, aiB) in enumerate(system_hierarchy[gB])
                    miB = active_modes[aiB]
                    ec_val = ec_transformed[miA, miB]
                    abs(ec_val) < 1e-15 && continue

                    # Transform n operators to truncated eigenbasis
                    evA = group_eigvecs[gA]
                    evB = group_eigvecs[gB]
                    n_opA = group_n_ops[gA][liA]
                    n_opB = group_n_ops[gB][liB]

                    nA_dressed = evA' * n_opA * evA
                    nB_dressed = evB' * n_opB * evB

                    # Create QuantumObject closures for add_interaction!
                    nA_qobj = QuantumObject(sparse(nA_dressed))
                    nB_qobj = QuantumObject(sparse(nB_dressed))
                    scA = subcircuits[gA]
                    scB = subcircuits[gB]

                    add_interaction!(hs, 4 * ec_val,
                        [scA, scB],
                        [_ -> nA_qobj, _ -> nB_qobj])
                end
            end
        end
    end

    return hs
end
