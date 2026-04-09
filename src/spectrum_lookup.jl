# ──────────────────────────────────────────────────────────────────────────────
# SpectrumLookup functions: bare ↔ dressed state mapping for composite systems
#
# The SpectrumLookup struct is defined in spectrum_data.jl (loaded early).
# ──────────────────────────────────────────────────────────────────────────────

"""Default overlap threshold (squared) for bare↔dressed assignment.

A dressed state is only assigned to a bare state when
`|⟨bare|dressed⟩|² > OVERLAP_THRESHOLD`.  Matches the scqubits default of 0.5.
"""
const OVERLAP_THRESHOLD = 0.5

"""
    generate_lookup!(hs::HilbertSpace; evals_count=10,
                     overlap_threshold=OVERLAP_THRESHOLD)

Diagonalize the full Hamiltonian and build bare ↔ dressed state mapping.
Stores result in `hs.lookup`. Returns the `SpectrumLookup`.

The assignment algorithm matches scqubits DE (Dressed Energy) ordering:
dressed states are processed in ascending index order; each dressed state is
assigned to the bare product state with the largest `|⟨bare|dressed⟩|`,
provided `|overlap|² > overlap_threshold`. Once a bare state is claimed it
cannot be reused.

Set `overlap_threshold=0` to force assignment even for low overlaps (the
previous default behavior).
"""
function generate_lookup!(hs::HilbertSpace; evals_count::Int=10,
                          overlap_threshold::Float64=OVERLAP_THRESHOLD)
    # Dressed eigensystem
    dressed_vals, dressed_vecs = eigensys(hs; evals_count=evals_count)
    n_dressed = length(dressed_vals)

    # Bare eigensystems for each subsystem
    n_sub = length(hs.subsystems)
    bare_evals = Vector{Vector{Float64}}(undef, n_sub)
    bare_evecs = Vector{Matrix{ComplexF64}}(undef, n_sub)
    sub_dims = Int[]
    for (i, s) in enumerate(hs.subsystems)
        dim = hilbertdim(s)
        push!(sub_dims, dim)
        vals, vecs = eigensys(s; evals_count=dim)
        bare_evals[i] = vals
        bare_evecs[i] = vecs
    end

    # Build bare product states in the full Hilbert space
    # Product state |n1, n2, ...⟩ = |n1⟩ ⊗ |n2⟩ ⊗ ...
    total_dim = prod(sub_dims)
    n_bare_states = total_dim

    # Generate all multi-indices for bare states
    bare_indices = _generate_bare_indices(sub_dims)

    # Build overlap matrix: overlap[i, j] = |⟨dressed_i|bare_j⟩|²
    # where bare_j is the j-th product state in lexicographic order
    overlap = zeros(Float64, n_dressed, n_bare_states)

    for (j, bare_idx) in enumerate(bare_indices)
        # Construct bare product state vector
        bare_state = bare_evecs[1][:, bare_idx[1]]
        for k in 2:n_sub
            bare_state = kron(bare_state, bare_evecs[k][:, bare_idx[k]])
        end

        for i in 1:n_dressed
            overlap[i, j] = abs2(dot(dressed_vecs[:, i], bare_state))
        end
    end

    # Greedy assignment matching scqubits DE ordering:
    # iterate dressed states in ascending index order (1, 2, 3, ...).
    bare_to_dressed = Dict{Tuple, Int}()
    dressed_to_bare = Dict{Int, Tuple}()

    # Work on a mutable copy so we can zero out assigned columns.
    ov_work = copy(overlap)

    for dressed_idx in 1:n_dressed
        max_pos = argmax(@view ov_work[dressed_idx, :])
        max_ov = sqrt(ov_work[dressed_idx, max_pos])  # |⟨bare|dressed⟩|

        if max_ov^2 > overlap_threshold
            # Claim this bare state (zero its column to prevent reuse)
            ov_work[:, max_pos] .= 0.0

            bare_tup = Tuple(bare_indices[max_pos])
            bare_to_dressed[bare_tup] = dressed_idx
            dressed_to_bare[dressed_idx] = bare_tup
        end
    end

    lookup = SpectrumLookup(dressed_vals, dressed_vecs, bare_evals,
                             overlap, bare_to_dressed, dressed_to_bare)
    hs.lookup = lookup
    return lookup
end

"""
    dressed_index(hs::HilbertSpace, bare_labels::Int...)

Find the dressed eigenstate index corresponding to bare state labels.
Labels are 1-based eigenstate indices for each subsystem.

Example: `dressed_index(hs, 1, 2)` finds the dressed state corresponding to
ground state of subsystem 1 and first excited state of subsystem 2.
"""
function dressed_index(hs::HilbertSpace, bare_labels::Int...)
    hs.lookup === nothing && error("Call generate_lookup!(hs) first")
    key = Tuple(bare_labels)
    haskey(hs.lookup.bare_to_dressed, key) ||
        error("Bare state $key not found in lookup table")
    return hs.lookup.bare_to_dressed[key]
end

"""
    bare_index(hs::HilbertSpace, dressed_idx::Int)

Find the bare state labels corresponding to a dressed eigenstate index.
Returns a tuple of 1-based eigenstate indices for each subsystem.
"""
function bare_index(hs::HilbertSpace, dressed_idx::Int)
    hs.lookup === nothing && error("Call generate_lookup!(hs) first")
    haskey(hs.lookup.dressed_to_bare, dressed_idx) ||
        error("Dressed index $dressed_idx not found in lookup table")
    return hs.lookup.dressed_to_bare[dressed_idx]
end

"""
    energy_by_dressed_index(hs::HilbertSpace, idx::Int)

Return the dressed eigenvalue at index `idx`.
"""
function energy_by_dressed_index(hs::HilbertSpace, idx::Int)
    hs.lookup === nothing && error("Call generate_lookup!(hs) first")
    return hs.lookup.dressed_evals[idx]
end

"""
    energy_by_bare_index(hs::HilbertSpace, bare_labels::Int...)

Return the dressed eigenvalue for the state labeled by bare quantum numbers.
"""
function energy_by_bare_index(hs::HilbertSpace, bare_labels::Int...)
    idx = dressed_index(hs, bare_labels...)
    return hs.lookup.dressed_evals[idx]
end

"""
    op_in_dressed_eigenbasis(hs::HilbertSpace, op; truncated_dim=nothing)

Transform operator `op` (in the bare product basis) to the dressed eigenbasis.
Returns `U' * op_mat * U` where U is the matrix of dressed eigenvectors.

If `truncated_dim` is specified, only keep that many dressed states.
"""
function op_in_dressed_eigenbasis(hs::HilbertSpace, op;
                                  truncated_dim::Union{Nothing, Int}=nothing)
    hs.lookup === nothing && error("Call generate_lookup!(hs) first")
    U = hs.lookup.dressed_evecs
    op_mat = op isa QuantumObject ? op.data : op
    n = truncated_dim === nothing ? size(U, 2) : min(truncated_dim, size(U, 2))
    U_trunc = U[:, 1:n]
    return U_trunc' * op_mat * U_trunc
end

# ── Helpers ──────────────────────────────────────────────────────────────────

function _generate_bare_indices(sub_dims::Vector{Int})
    # Generate all tuples (i1, i2, ...) where 1 <= ik <= sub_dims[k]
    n = length(sub_dims)
    total = prod(sub_dims)
    indices = Vector{Vector{Int}}(undef, total)
    idx = 1
    _fill_indices!(indices, idx, Int[], sub_dims, 1)
    return indices
end

function _fill_indices!(indices::Vector{Vector{Int}}, idx::Int,
                         current::Vector{Int}, dims::Vector{Int}, level::Int)
    if level > length(dims)
        indices[idx] = copy(current)
        return idx + 1
    end
    for i in 1:dims[level]
        push!(current, i)
        idx = _fill_indices!(indices, idx, current, dims, level + 1)
        pop!(current)
    end
    return idx
end
