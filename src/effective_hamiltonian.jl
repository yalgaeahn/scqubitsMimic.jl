# ──────────────────────────────────────────────────────────────────────────────
# Effective Hamiltonian analysis on top of configured hierarchical circuits
# ──────────────────────────────────────────────────────────────────────────────

"""
    effective_hamiltonian(circ::Circuit;
                          projection_dims=(2,2,2),
                          basis_labels=nothing,
                          evals_count=nothing,
                          decompose_pauli=true)

Build a projected effective Hamiltonian for a configured circuit.

The circuit must have been configured via [`configure!`](@ref). Internally, the
configured hierarchical `HilbertSpace` is reused (or rebuilt if parameter
changes invalidated the numerical cache).

The effective Hamiltonian is the restriction of the full dressed Hamiltonian to
the selected bare product-state subspace:

    H_eff[i,k] = sum_j  E_j  <bare_i|dressed_j> <dressed_j|bare_k>

where the sum runs over **all** computed dressed eigenstates, not only those
corresponding to the selected bare labels. This ensures that virtual
contributions from non-selected states are included.

Returns a named tuple containing:
- `basis_labels`
- `dressed_indices`  — dressed indices corresponding to the selected bare labels
- `dressed_energies` — dressed energies for those indices
- `overlap_matrix`   — rectangular (n_bare × n_dressed) projection matrix
- `H_full_eff`       — full effective Hamiltonian in the bare subspace
- `H_bare_eff`       — diagonal bare-energy Hamiltonian
- `H_int_eff`        — interaction piece: `H_full_eff - H_bare_eff`
- `projection_dims`
- `subsystem_trunc_dims`
- `pauli_terms` for all-2-level projections when `decompose_pauli=true`
  (decomposed from `H_int_eff`, not `H_full_eff`)
"""
function effective_hamiltonian(circ::Circuit;
                               projection_dims=(2, 2, 2),
                               basis_labels=nothing,
                               evals_count::Union{Nothing, Int}=nothing,
                               decompose_pauli::Bool=true)
    hs = _ensure_configured_hilbert_space!(circ)
    sub_dims = [hilbertdim(s) for s in hs.subsystems]
    labels, resolved_projection_dims = _resolve_basis_labels(basis_labels,
                                                             projection_dims,
                                                             sub_dims)

    lookup_evals = isnothing(evals_count) ? hilbertdim(hs) : evals_count
    generate_lookup!(hs; evals_count=lookup_evals, overlap_threshold=0.0)

    bare_index_map = _bare_basis_index_map(sub_dims)
    selected_bare_indices = Int[]
    for label in labels
        haskey(bare_index_map, label) || error(
            "Bare label $label is out of range for subsystem dimensions $sub_dims.")
        push!(selected_bare_indices, bare_index_map[label])
    end

    bare_evals = [eigenvals(s; evals_count=hilbertdim(s)) for s in hs.subsystems]
    bare_energies = ComplexF64[
        sum(bare_evals[subsys_idx][label[subsys_idx]]
            for subsys_idx in eachindex(label))
        for label in labels
    ]
    H_bare = Matrix(Diagonal(bare_energies))
    dressed_indices = Int[dressed_index(hs, label...) for label in labels]
    dressed_energies = Float64[energy_by_bare_index(hs, label...) for label in labels]

    # Build rectangular overlap matrix: P[i, j] = <bare_i | dressed_j>
    # where i runs over the selected bare indices and j over ALL dressed states.
    dressed_vecs = hs.lookup.dressed_evecs
    n_bare = length(labels)
    n_dressed = size(dressed_vecs, 2)
    P = Matrix{ComplexF64}(undef, n_bare, n_dressed)
    for j in 1:n_dressed
        P[:, j] = dressed_vecs[selected_bare_indices, j]
    end

    # H_full_eff = P * diag(all_dressed_energies) * P'
    # This sums over ALL dressed states, giving the exact restriction of H to
    # the selected bare subspace.
    all_dressed_energies = ComplexF64.(hs.lookup.dressed_evals)
    H_full = _hermitize(P * (all_dressed_energies .* P') )
    H_int = _hermitize(H_full - H_bare)

    pauli_terms = if decompose_pauli && all(==(2), resolved_projection_dims)
        _pauli_decomposition(H_int)
    else
        nothing
    end

    # Also provide the square sub-overlap for the selected dressed states only
    # (useful for diagnostics — when this is near-unitary, the projection is clean).
    overlap_selected = P[:, dressed_indices]

    return (
        basis_labels=labels,
        dressed_indices=dressed_indices,
        dressed_energies=dressed_energies,
        overlap_matrix=overlap_selected,
        H_full_eff=H_full,
        H_bare_eff=H_bare,
        H_int_eff=H_int,
        projection_dims=resolved_projection_dims,
        subsystem_trunc_dims=deepcopy(circ._subsystem_trunc_dims),
        pauli_terms=pauli_terms,
    )
end

"""
    exchange_coupling(H, basis_labels, state_a, state_b; absval=true)

Return the matrix element between `state_a` and `state_b` inside a projected
effective Hamiltonian matrix.
"""
function exchange_coupling(H::AbstractMatrix,
                           basis_labels,
                           state_a::Tuple,
                           state_b::Tuple;
                           absval::Bool=true)
    idx_a = findfirst(==(state_a), basis_labels)
    idx_b = findfirst(==(state_b), basis_labels)
    idx_a === nothing && error("State $state_a not found in basis_labels")
    idx_b === nothing && error("State $state_b not found in basis_labels")
    val = H[idx_a, idx_b]
    return absval ? abs(val) : val
end

"""
    avoided_crossing_coupling(circ::Circuit;
                              state_a,
                              state_b,
                              sweep_param,
                              sweep_vals,
                              evals_count=16)

Track two bare-labeled states through a parameter sweep and extract the
minimum avoided-crossing gap. The reported coupling is `g_half_gap = min_gap/2`.
"""
function avoided_crossing_coupling(circ::Circuit;
                                   state_a::Tuple,
                                   state_b::Tuple,
                                   sweep_param::Symbol,
                                   sweep_vals,
                                   evals_count::Int=16)
    circ._hierarchical_diagonalization ||
        error("avoided_crossing_coupling requires configure!() to be called first")

    n_sub = length(circ._subsystems === nothing ? _ensure_configured_hilbert_space!(circ).subsystems :
                   circ._subsystems)
    length(state_a) == n_sub || error("state_a must have length $n_sub, got $(length(state_a))")
    length(state_b) == n_sub || error("state_b must have length $n_sub, got $(length(state_b))")

    param_vals = collect(Float64, sweep_vals)
    tracked_a = Vector{Float64}(undef, length(param_vals))
    tracked_b = Vector{Float64}(undef, length(param_vals))
    original_val = get_param(circ, sweep_param)

    try
        for (idx, val) in enumerate(param_vals)
            set_param!(circ, sweep_param, val)
            hs = _ensure_configured_hilbert_space!(circ)
            generate_lookup!(hs; evals_count=evals_count, overlap_threshold=0.0)
            tracked_a[idx] = energy_by_bare_index(hs, state_a...)
            tracked_b[idx] = energy_by_bare_index(hs, state_b...)
        end
    finally
        set_param!(circ, sweep_param, original_val)
    end

    gap = abs.(tracked_a .- tracked_b)
    min_idx = argmin(gap)
    return (
        state_a=state_a,
        state_b=state_b,
        sweep_param=sweep_param,
        sweep_vals=param_vals,
        tracked_energy_a=tracked_a,
        tracked_energy_b=tracked_b,
        gap=gap,
        min_gap=gap[min_idx],
        param_at_min_gap=param_vals[min_idx],
        g_half_gap=gap[min_idx] / 2,
    )
end

    # ── Internal helpers ─────────────────────────────────────────────────────────

function _ensure_configured_hilbert_space!(circ::Circuit)
    circ._hierarchical_diagonalization ||
        error("This analysis requires configure!() to be called first")
    circ._system_hierarchy === nothing &&
        error("No stored system_hierarchy found. Call configure!() first.")
    circ._subsystem_trunc_dims === nothing &&
        error("No stored subsystem_trunc_dims found. Call configure!() with explicit truncation.")

    if circ._hilbert_space === nothing
        hs = hierarchical_diag(circ;
            system_hierarchy=circ._system_hierarchy,
            subsystem_trunc_dims=circ._subsystem_trunc_dims)
        circ._hilbert_space = hs
        circ._subsystems = SubCircuit[s for s in hs.subsystems]
    elseif circ._subsystems === nothing
        circ._subsystems = SubCircuit[s for s in circ._hilbert_space.subsystems]
    end

    return circ._hilbert_space
end

function _resolve_basis_labels(basis_labels,
                               projection_dims,
                               sub_dims::Vector{Int})
    n_sub = length(sub_dims)
    if basis_labels === nothing
        length(projection_dims) == n_sub || throw(ArgumentError(
            "projection_dims must have length $n_sub, got $(length(projection_dims))"))
        any(d < 1 for d in projection_dims) && throw(ArgumentError(
            "projection_dims must be positive, got $projection_dims"))
        for (dim, sub_dim) in zip(projection_dims, sub_dims)
            dim <= sub_dim || throw(ArgumentError(
                "projection_dims=$projection_dims exceeds subsystem dimensions $sub_dims"))
        end
        labels = Tuple.(_generate_bare_indices(Int[projection_dims...]))
        return labels, Tuple(projection_dims)
    end

    labels = Tuple{Vararg{Int}}[]
    for label in basis_labels
        tpl = Tuple(label)
        length(tpl) == n_sub || throw(ArgumentError(
            "Each basis label must have length $n_sub, got $tpl"))
        for (idx, entry) in enumerate(tpl)
            1 <= entry <= sub_dims[idx] || throw(ArgumentError(
                "Basis label $tpl exceeds subsystem dimensions $sub_dims"))
        end
        push!(labels, tpl)
    end
    length(unique(labels)) == length(labels) ||
        throw(ArgumentError("basis_labels must be unique"))
    resolved_projection_dims = ntuple(i -> maximum(label[i] for label in labels), n_sub)
    return labels, resolved_projection_dims
end

function _bare_basis_index_map(sub_dims::Vector{Int})
    labels = _generate_bare_indices(sub_dims)
    return Dict(Tuple(label) => idx for (idx, label) in enumerate(labels))
end

_hermitize(H::AbstractMatrix) = (Matrix{ComplexF64}(H) + Matrix{ComplexF64}(H)') / 2

function _pauli_decomposition(H::AbstractMatrix; tol::Float64=1e-12)
    size(H, 1) == size(H, 2) || throw(DimensionMismatch(
        "Effective Hamiltonian must be square, got size $(size(H))"))
    n = round(Int, log2(size(H, 1)))
    2^n == size(H, 1) || throw(ArgumentError(
        "Pauli decomposition requires a 2^n-dimensional matrix, got size $(size(H, 1))"))

    pauli = Dict(
        'I' => ComplexF64[1 0; 0 1],
        'X' => ComplexF64[0 1; 1 0],
        'Y' => ComplexF64[0 -im; im 0],
        'Z' => ComplexF64[1 0; 0 -1],
    )
    symbols = ('I', 'X', 'Y', 'Z')
    coeffs = Dict{String, ComplexF64}()

    for combo in Iterators.product(ntuple(_ -> symbols, n)...)
        label = String(collect(combo))
        op = reduce(kron, (pauli[sym] for sym in combo))
        coeff = tr(op' * H) / 2^n
        abs(coeff) > tol && (coeffs[label] = coeff)
    end

    return coeffs
end
