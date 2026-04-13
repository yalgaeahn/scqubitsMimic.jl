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

"""Resolve the actual overlap threshold for a lookup build."""
function _lookup_threshold(hs::HilbertSpace;
                           overlap_threshold::Union{Nothing, Float64}=nothing,
                           ignore_low_overlap::Union{Nothing, Bool}=nothing)
    if overlap_threshold !== nothing
        overlap_threshold < 0 && throw(ArgumentError(
            "overlap_threshold must be nonnegative, got $overlap_threshold"))
        return overlap_threshold
    end
    use_ignore_low_overlap = isnothing(ignore_low_overlap) ? hs.ignore_low_overlap : ignore_low_overlap
    return use_ignore_low_overlap ? 0.0 : OVERLAP_THRESHOLD
end

function _normalize_lookup_ordering(ordering::Union{Symbol, AbstractString})
    normalized = Symbol(uppercase(String(ordering)))
    normalized in (:DE, :LX, :BE) || throw(ArgumentError(
        "ordering must be one of :DE, :LX, or :BE; got $(repr(ordering))"))
    return normalized
end

function _validate_subsys_priority(hs::HilbertSpace,
                                   subsys_priority::Union{Nothing, AbstractVector{<:Integer}})
    n_sub = length(hs.subsystems)
    if subsys_priority === nothing
        return collect(1:n_sub)
    end
    priority = Int.(collect(subsys_priority))
    sort(priority) == collect(1:n_sub) || throw(ArgumentError(
        "subsys_priority must be a 1-based permutation of 1:$n_sub; got $priority"))
    return priority
end

function _lookup_build_data_from_spectral_data(hs::HilbertSpace,
                                               dressed_vals::AbstractVector,
                                               dressed_vecs::AbstractMatrix,
                                               bare_evals::AbstractVector,
                                               bare_evecs::AbstractVector)
    dressed_vals_ref = dressed_vals isa Vector{Float64} ? dressed_vals : Float64.(collect(dressed_vals))
    dressed_vecs_ref = dressed_vecs isa Matrix{ComplexF64} ? dressed_vecs : Matrix{ComplexF64}(dressed_vecs)
    n_dressed = length(dressed_vals_ref)

    n_sub = length(hs.subsystems)
    bare_evals_ref = Vector{Vector{Float64}}(undef, n_sub)
    bare_evecs_ref = Vector{Matrix{ComplexF64}}(undef, n_sub)
    sub_dims = Int[]
    for i in 1:n_sub
        bare_evals_ref[i] = bare_evals[i] isa Vector{Float64} ? bare_evals[i] : Float64.(collect(bare_evals[i]))
        bare_evecs_ref[i] = bare_evecs[i] isa Matrix{ComplexF64} ? bare_evecs[i] : Matrix{ComplexF64}(bare_evecs[i])
        push!(sub_dims, length(bare_evals_ref[i]))
    end

    bare_indices = _generate_bare_indices(sub_dims)
    overlap = zeros(Float64, n_dressed, length(bare_indices))

    for (j, bare_idx) in enumerate(bare_indices)
        bare_state = bare_evecs_ref[1][:, bare_idx[1]]
        for k in 2:n_sub
            bare_state = kron(bare_state, bare_evecs_ref[k][:, bare_idx[k]])
        end

        for i in 1:n_dressed
            overlap[i, j] = abs2(dot(dressed_vecs_ref[:, i], bare_state))
        end
    end

    return (; dressed_vals=dressed_vals_ref, dressed_vecs=dressed_vecs_ref,
            bare_evals=bare_evals_ref, bare_evecs=bare_evecs_ref,
            sub_dims, bare_indices, overlap)
end

function _lookup_build_data(hs::HilbertSpace; evals_count::Int=10)
    dressed_vals, dressed_vecs = eigensys(hs; evals_count=evals_count)

    n_sub = length(hs.subsystems)
    bare_evals = Vector{Vector{Float64}}(undef, n_sub)
    bare_evecs = Vector{Matrix{ComplexF64}}(undef, n_sub)
    for (i, s) in enumerate(hs.subsystems)
        dim = hilbertdim(s)
        vals, vecs = eigensys(s; evals_count=dim)
        bare_evals[i] = vals
        bare_evecs[i] = vecs
    end

    return _lookup_build_data_from_spectral_data(hs, dressed_vals, dressed_vecs,
                                                 bare_evals, bare_evecs)
end

function _build_lookup_de(data, threshold::Float64)
    bare_to_dressed = Dict{Tuple, Int}()
    dressed_to_bare = Dict{Int, Tuple}()
    ov_work = copy(data.overlap)

    for dressed_idx in 1:length(data.dressed_vals)
        max_pos = argmax(@view ov_work[dressed_idx, :])
        ov_work[dressed_idx, max_pos] > threshold || continue
        ov_work[:, max_pos] .= 0.0
        bare_tup = Tuple(data.bare_indices[max_pos])
        bare_to_dressed[bare_tup] = dressed_idx
        dressed_to_bare[dressed_idx] = bare_tup
    end

    return bare_to_dressed, dressed_to_bare
end

function _bare_excitation_operator_native(subsys::AbstractQuantumSystem,
                                          bare_evecs::Matrix{ComplexF64})
    dim = hilbertdim(subsys)
    bare_excitation = Matrix{ComplexF64}(create(dim).data)
    return bare_evecs * bare_excitation * bare_evecs'
end

function _full_excitation_operators(hs::HilbertSpace, bare_evecs, sub_dims)
    return [
        identity_wrap(_bare_excitation_operator_native(subsys, bare_evecs[i]), i, sub_dims).data
        for (i, subsys) in enumerate(hs.subsystems)
    ]
end

function _branch_analysis_lx_step!(priority::Vector{Int},
                                   depth::Int,
                                   current_dressed_idx::Int,
                                   current_state::Vector{ComplexF64},
                                   remaining_dressed_indices::Vector{Int},
                                   remaining_states::Vector{Vector{ComplexF64}},
                                   excite_ops,
                                   sub_dims::Vector{Int})
    mode = priority[depth]
    branch_length = sub_dims[mode]
    branch = Any[]

    dressed_idx = current_dressed_idx
    state = current_state
    while true
        if depth == length(priority)
            push!(branch, dressed_idx)
        else
            push!(branch, _branch_analysis_lx_step!(priority, depth + 1,
                                                   dressed_idx, state,
                                                   remaining_dressed_indices,
                                                   remaining_states,
                                                   excite_ops, sub_dims))
        end

        length(branch) == branch_length && break
        isempty(remaining_states) && throw(ArgumentError(
            "Not enough dressed eigenstates to complete LX lookup labeling. " *
            "Increase evals_count to cover the full Hilbert space."
        ))

        excited_state = excite_ops[mode] * state
        norm_excited = norm(excited_state)
        norm_excited > 0 || throw(ArgumentError(
            "Could not excite branch state while building LX lookup for subsystem $mode."))
        excited_state ./= norm_excited

        overlaps = [abs(dot(excited_state, candidate)) for candidate in remaining_states]
        max_pos = argmax(overlaps)
        state = remaining_states[max_pos]
        dressed_idx = remaining_dressed_indices[max_pos]
        deleteat!(remaining_states, max_pos)
        deleteat!(remaining_dressed_indices, max_pos)
    end

    return branch
end

function _fill_nested_branch_array!(dest::Array{Int}, nested, prefix::Vector{Int}=Int[])
    if nested isa Integer
        dest[prefix...] = nested
        return
    end

    for (i, item) in enumerate(nested)
        push!(prefix, i)
        _fill_nested_branch_array!(dest, item, prefix)
        pop!(prefix)
    end
end

function _build_lookup_lx(hs::HilbertSpace, data;
                          subsys_priority::Union{Nothing, AbstractVector{<:Integer}}=nothing)
    prod(data.sub_dims) == length(data.dressed_vals) || throw(ArgumentError(
        "LX ordering requires a complete dressed eigensystem. " *
        "Set evals_count=$(prod(data.sub_dims)) for this Hilbert space."))

    priority = _validate_subsys_priority(hs, subsys_priority)
    excite_ops = _full_excitation_operators(hs, data.bare_evecs, data.sub_dims)
    remaining_indices = collect(2:length(data.dressed_vals))
    remaining_states = [data.dressed_vecs[:, i] for i in 2:length(data.dressed_vals)]

    nested = _branch_analysis_lx_step!(priority, 1, 1, data.dressed_vecs[:, 1],
                                       remaining_indices, remaining_states,
                                       excite_ops, data.sub_dims)
    arranged = Array{Int}(undef, Tuple(data.sub_dims[priority])...)
    _fill_nested_branch_array!(arranged, nested)
    original_order = permutedims(arranged, invperm(priority))

    bare_to_dressed = Dict{Tuple, Int}()
    dressed_to_bare = Dict{Int, Tuple}()
    for idx in CartesianIndices(original_order)
        bare_label = Tuple(idx.I)
        dressed_idx = original_order[idx]
        bare_to_dressed[bare_label] = dressed_idx
        dressed_to_bare[dressed_idx] = bare_label
    end

    return bare_to_dressed, dressed_to_bare
end

function _build_lookup_be(hs::HilbertSpace, data;
                          subsys_priority::Union{Nothing, AbstractVector{<:Integer}}=nothing,
                          BEs_count::Union{Nothing, Int}=nothing)
    priority = _validate_subsys_priority(hs, subsys_priority)
    BEs_count !== nothing && BEs_count < 1 && throw(ArgumentError(
        "BEs_count must be positive; got $BEs_count"))

    n_available = length(data.dressed_vals)
    n_assign = isnothing(BEs_count) ? n_available : min(BEs_count, n_available)
    if !isnothing(BEs_count) && BEs_count > n_available
        @warn "evals_count=$n_available is less than BEs_count=$BEs_count; using $n_assign assigned dressed states."
    end

    excite_ops = _full_excitation_operators(hs, data.bare_evecs, data.sub_dims)
    bare_energies = [
        sum(data.bare_evals[subsys_idx][bare_idx[subsys_idx]] for subsys_idx in eachindex(data.sub_dims))
        for bare_idx in data.bare_indices
    ]
    sorted_positions = sortperm(bare_energies)[1:n_assign]

    assigned = fill(0, Tuple(data.sub_dims)...)
    remaining_indices = collect(1:n_available)
    remaining_states = [data.dressed_vecs[:, i] for i in 1:n_available]

    bare_to_dressed = Dict{Tuple, Int}()
    dressed_to_bare = Dict{Int, Tuple}()
    ground_label = ntuple(_ -> 1, length(data.sub_dims))

    for pos in sorted_positions
        bare_label = Tuple(data.bare_indices[pos])

        if bare_label == ground_label
            assigned[bare_label...] = 1
            bare_to_dressed[bare_label] = 1
            dressed_to_bare[1] = bare_label
            ground_pos = findfirst(==(1), remaining_indices)
            ground_pos !== nothing && (deleteat!(remaining_indices, ground_pos); deleteat!(remaining_states, ground_pos))
            continue
        end

        isempty(remaining_states) && break

        chosen_pos = nothing
        for subsys_idx in reverse(priority)
            bare_label[subsys_idx] == 1 && continue
            prev_label = collect(bare_label)
            prev_label[subsys_idx] -= 1
            prev_dressed_idx = assigned[Tuple(prev_label)...]
            prev_dressed_idx == 0 && continue

            excited_state = excite_ops[subsys_idx] * data.dressed_vecs[:, prev_dressed_idx]
            norm_excited = norm(excited_state)
            norm_excited > 0 || continue
            excited_state ./= norm_excited

            overlaps = [abs(dot(excited_state, candidate)) for candidate in remaining_states]
            chosen_pos = argmax(overlaps)
            break
        end

        chosen_pos === nothing && throw(ArgumentError(
            "Could not identify a previous assigned state to continue BE lookup labeling for bare label $bare_label"))

        dressed_idx = remaining_indices[chosen_pos]
        assigned[bare_label...] = dressed_idx
        bare_to_dressed[bare_label] = dressed_idx
        dressed_to_bare[dressed_idx] = bare_label
        deleteat!(remaining_indices, chosen_pos)
        deleteat!(remaining_states, chosen_pos)
    end

    return bare_to_dressed, dressed_to_bare
end

"""
    _build_lookup(hs::HilbertSpace; evals_count=10, ordering=:DE,
                  subsys_priority=nothing, BEs_count=nothing,
                  overlap_threshold=nothing, ignore_low_overlap=nothing)

Build a `SpectrumLookup` for `hs` without mutating `hs.lookup`.

Julia bare labels are 1-based throughout this API: `(1, 1, ...)` denotes the
product ground state.
"""
function _build_lookup_from_data(hs::HilbertSpace, data;
                                 ordering::Union{Symbol, AbstractString}=:DE,
                                 subsys_priority::Union{Nothing, AbstractVector{<:Integer}}=nothing,
                                 BEs_count::Union{Nothing, Int}=nothing,
                                 overlap_threshold::Union{Nothing, Float64}=nothing,
                                 ignore_low_overlap::Union{Nothing, Bool}=nothing)
    normalized_ordering = _normalize_lookup_ordering(ordering)
    if normalized_ordering == :DE
        subsys_priority !== nothing &&
            @warn "subsys_priority is ignored for DE ordering."
        BEs_count !== nothing &&
            @warn "BEs_count is ignored for DE ordering."
        threshold = _lookup_threshold(hs;
                                      overlap_threshold=overlap_threshold,
                                      ignore_low_overlap=ignore_low_overlap)
        bare_to_dressed, dressed_to_bare = _build_lookup_de(data, threshold)
    elseif normalized_ordering == :LX
        BEs_count !== nothing &&
            @warn "BEs_count is ignored for LX ordering."
        overlap_threshold !== nothing &&
            @warn "overlap_threshold is ignored for LX ordering."
        bare_to_dressed, dressed_to_bare = _build_lookup_lx(hs, data;
                                                            subsys_priority=subsys_priority)
    else
        overlap_threshold !== nothing &&
            @warn "overlap_threshold is ignored for BE ordering."
        bare_to_dressed, dressed_to_bare = _build_lookup_be(hs, data;
                                                            subsys_priority=subsys_priority,
                                                            BEs_count=BEs_count)
    end

    return SpectrumLookup(data.dressed_vals, data.dressed_vecs, data.bare_evals,
                          data.overlap, bare_to_dressed, dressed_to_bare)
end

function _build_lookup_from_spectral_data(hs::HilbertSpace,
                                          dressed_vals::AbstractVector,
                                          dressed_vecs::AbstractMatrix,
                                          bare_evals::AbstractVector,
                                          bare_evecs::AbstractVector;
                                          ordering::Union{Symbol, AbstractString}=:DE,
                                          subsys_priority::Union{Nothing, AbstractVector{<:Integer}}=nothing,
                                          BEs_count::Union{Nothing, Int}=nothing,
                                          overlap_threshold::Union{Nothing, Float64}=nothing,
                                          ignore_low_overlap::Union{Nothing, Bool}=nothing)
    data = _lookup_build_data_from_spectral_data(hs, dressed_vals, dressed_vecs,
                                                 bare_evals, bare_evecs)
    return _build_lookup_from_data(hs, data;
                                   ordering=ordering,
                                   subsys_priority=subsys_priority,
                                   BEs_count=BEs_count,
                                   overlap_threshold=overlap_threshold,
                                   ignore_low_overlap=ignore_low_overlap)
end

function _build_lookup(hs::HilbertSpace; evals_count::Int=10,
                       ordering::Union{Symbol, AbstractString}=:DE,
                       subsys_priority::Union{Nothing, AbstractVector{<:Integer}}=nothing,
                       BEs_count::Union{Nothing, Int}=nothing,
                       overlap_threshold::Union{Nothing, Float64}=nothing,
                       ignore_low_overlap::Union{Nothing, Bool}=nothing)
    data = _lookup_build_data(hs; evals_count=evals_count)
    return _build_lookup_from_data(hs, data;
                                   ordering=ordering,
                                   subsys_priority=subsys_priority,
                                   BEs_count=BEs_count,
                                   overlap_threshold=overlap_threshold,
                                   ignore_low_overlap=ignore_low_overlap)
end

"""
    generate_lookup!(hs::HilbertSpace; evals_count=10, ordering=:DE,
                     subsys_priority=nothing, BEs_count=nothing)

Diagonalize the full Hamiltonian and build a bare ↔ dressed lookup table on
`hs.lookup`.

Supported ordering schemes follow scqubits naming:
- `:DE` / `"DE"` — dressed-energy overlap labeling
- `:LX` / `"LX"` — lexical branch analysis
- `:BE` / `"BE"` — bare-energy branch analysis

Julia bare labels remain 1-based.
"""
function generate_lookup!(hs::HilbertSpace; evals_count::Int=10,
                          ordering::Union{Symbol, AbstractString}=:DE,
                          subsys_priority::Union{Nothing, AbstractVector{<:Integer}}=nothing,
                          BEs_count::Union{Nothing, Int}=nothing,
                          overlap_threshold::Union{Nothing, Float64}=nothing)
    lookup = _build_lookup(hs; evals_count=evals_count,
                           ordering=ordering,
                           subsys_priority=subsys_priority,
                           BEs_count=BEs_count,
                           overlap_threshold=overlap_threshold)
    hs.lookup = lookup
    return lookup
end

"""Return `true` when `hs` already has a populated [`SpectrumLookup`](@ref)."""
lookup_exists(hs::HilbertSpace) = hs.lookup !== nothing

function _lookup_dressed_index(lookup::SpectrumLookup, bare_labels::Tuple)
    return get(lookup.bare_to_dressed, bare_labels, nothing)
end

_lookup_dressed_index(lookup::SpectrumLookup, bare_labels::Int...) =
    _lookup_dressed_index(lookup, Tuple(bare_labels))

function _lookup_dressed_index_strict(lookup::SpectrumLookup, bare_labels::Tuple)
    idx = _lookup_dressed_index(lookup, bare_labels)
    idx === nothing && error(
        "Bare state $bare_labels not found in lookup. " *
        "Increase evals_count or use ignore_low_overlap=true for labeling.")
    return idx
end

_lookup_dressed_index_strict(lookup::SpectrumLookup, bare_labels::Int...) =
    _lookup_dressed_index_strict(lookup, Tuple(bare_labels))

function _lookup_bare_index(lookup::SpectrumLookup, dressed_idx::Int)
    haskey(lookup.dressed_to_bare, dressed_idx) || throw(ArgumentError(
        "Dressed index $dressed_idx not found in lookup table"))
    return lookup.dressed_to_bare[dressed_idx]
end

function _lookup_energy_by_bare_index(lookup::SpectrumLookup, bare_labels::Tuple;
                                      subtract_ground::Bool=false)
    idx = _lookup_dressed_index(lookup, bare_labels)
    idx === nothing && return NaN
    energy = lookup.dressed_evals[idx]
    if subtract_ground
        ground_energy = _lookup_energy_by_bare_index(lookup,
            ntuple(_ -> 1, length(bare_labels)))
        return energy - ground_energy
    end
    return energy
end

_lookup_energy_by_bare_index(lookup::SpectrumLookup, bare_labels::Int...; kwargs...) =
    _lookup_energy_by_bare_index(lookup, Tuple(bare_labels); kwargs...)

function _lookup_energy_by_bare_index_strict(lookup::SpectrumLookup, bare_labels::Tuple)
    idx = _lookup_dressed_index_strict(lookup, bare_labels)
    return lookup.dressed_evals[idx]
end

_lookup_energy_by_bare_index_strict(lookup::SpectrumLookup, bare_labels::Int...) =
    _lookup_energy_by_bare_index_strict(lookup, Tuple(bare_labels))

"""
    dressed_index(hs::HilbertSpace, bare_labels...)

Find the dressed eigenstate index corresponding to bare state labels.
Labels are 1-based eigenstate indices for each subsystem.

Example: `dressed_index(hs, 1, 2)` finds the dressed state corresponding to
ground state of subsystem 1 and first excited state of subsystem 2. Returns
`nothing` when the requested bare label is not present in the lookup.
"""
function dressed_index(hs::HilbertSpace, bare_labels::Tuple{Vararg{Int}})
    hs.lookup === nothing && error("Call generate_lookup!(hs) first")
    return _lookup_dressed_index(hs.lookup, bare_labels)
end

function dressed_index(hs::HilbertSpace, bare_labels::Int...)
    return dressed_index(hs, Tuple(bare_labels))
end

"""
    bare_index(hs::HilbertSpace, dressed_idx::Int)

Find the bare state labels corresponding to a dressed eigenstate index.
Returns a tuple of 1-based eigenstate indices for each subsystem.
"""
function bare_index(hs::HilbertSpace, dressed_idx::Int)
    hs.lookup === nothing && error("Call generate_lookup!(hs) first")
    return _lookup_bare_index(hs.lookup, dressed_idx)
end

"""
    energy_by_dressed_index(hs::HilbertSpace, idx::Int; subtract_ground=false)

Return the dressed eigenvalue at index `idx`.
"""
function energy_by_dressed_index(hs::HilbertSpace, idx::Int; subtract_ground::Bool=false)
    hs.lookup === nothing && error("Call generate_lookup!(hs) first")
    energy = hs.lookup.dressed_evals[idx]
    return subtract_ground ? energy - hs.lookup.dressed_evals[1] : energy
end

"""
    energy_by_bare_index(hs::HilbertSpace, bare_labels...; subtract_ground=false)

Return the dressed eigenvalue for the state labeled by bare quantum numbers.
Returns `NaN` when the requested bare label is not present in the lookup.
"""
function energy_by_bare_index(hs::HilbertSpace, bare_labels::Tuple{Vararg{Int}};
                              subtract_ground::Bool=false)
    hs.lookup === nothing && error("Call generate_lookup!(hs) first")
    return _lookup_energy_by_bare_index(hs.lookup, bare_labels;
                                        subtract_ground=subtract_ground)
end

function energy_by_bare_index(hs::HilbertSpace, bare_labels::Int...;
                              subtract_ground::Bool=false)
    return energy_by_bare_index(hs, Tuple(bare_labels);
                                subtract_ground=subtract_ground)
end

"""
    op_in_dressed_eigenbasis(hs::HilbertSpace, op_callable_or_tuple;
                             truncated_dim=nothing, op_in_bare_eigenbasis=false)

Transform an operator to the dressed eigenbasis of the full Hilbert space.

Supported interfaces:
- full operator matrix or `QuantumObject` already defined on the full Hilbert space
- `(op, subsys)` where `op` acts on a single subsystem
- callable subsystem operator generator, resolved by applicability to a unique subsystem

Julia bare labels are 1-based. The returned object is a dense matrix in the
truncated dressed basis.
"""
function _operator_matrix(op::QuantumObject)
    return Matrix{ComplexF64}(op.data)
end

function _operator_matrix(op::AbstractMatrix)
    return Matrix{ComplexF64}(op)
end

function _resolve_subsystem_index(hs::HilbertSpace, subsys::AbstractQuantumSystem)
    idx = findfirst(candidate -> candidate === subsys, hs.subsystems)
    idx === nothing && throw(ArgumentError("Subsystem not found in HilbertSpace"))
    return idx
end

function _subsystem_operator_to_full_matrix(hs::HilbertSpace, op, subsys::AbstractQuantumSystem;
                                            op_in_bare_eigenbasis::Bool=false)
    subsys_index = _resolve_subsystem_index(hs, subsys)
    sub_dims = [hilbertdim(s) for s in hs.subsystems]
    local_op = _operator_matrix(op)

    if op_in_bare_eigenbasis
        _, bare_evecs = eigensys(subsys; evals_count=hilbertdim(subsys))
        local_op = bare_evecs * local_op * bare_evecs'
    end

    return identity_wrap(local_op, subsys_index, sub_dims).data
end

function _resolve_callable_operator(hs::HilbertSpace, op_callable::Function)
    matches = Tuple{Any, AbstractQuantumSystem}[]
    for subsys in hs.subsystems
        applicable(op_callable, subsys) || continue
        push!(matches, (op_callable(subsys), subsys))
    end

    isempty(matches) && throw(ArgumentError(
        "Callable operator did not apply to any subsystem in this HilbertSpace."))
    length(matches) == 1 || throw(ArgumentError(
        "Callable operator is ambiguous for this HilbertSpace; use the `(op, subsys)` interface instead."))
    return matches[1]
end

"""
    op_in_dressed_eigenbasis(hs::HilbertSpace, op_callable_or_tuple;
                             truncated_dim=nothing,
                             op_in_bare_eigenbasis=false)

Transform an operator into the dressed eigenbasis of the full
[`HilbertSpace`](@ref).

Supported interfaces are:
- a full operator matrix or `QuantumObject`
- `(op, subsys)` for a subsystem-local operator
- a callable resolved against a unique subsystem
"""
function op_in_dressed_eigenbasis(hs::HilbertSpace, op_callable_or_tuple;
                                  truncated_dim::Union{Nothing, Int}=nothing,
                                  op_in_bare_eigenbasis::Bool=false)
    hs.lookup === nothing && error("Call generate_lookup!(hs) first")
    U = hs.lookup.dressed_evecs
    op_mat = if op_callable_or_tuple isa Tuple
        length(op_callable_or_tuple) == 2 || throw(ArgumentError(
            "Tuple interface must be `(op, subsys)`"))
        op, subsys = op_callable_or_tuple
        subsys isa AbstractQuantumSystem || throw(ArgumentError(
            "Tuple interface must be `(op, subsys)` with a subsystem object"))
        _subsystem_operator_to_full_matrix(hs, op, subsys;
                                           op_in_bare_eigenbasis=op_in_bare_eigenbasis)
    elseif op_callable_or_tuple isa Function
        op, subsys = _resolve_callable_operator(hs, op_callable_or_tuple)
        _subsystem_operator_to_full_matrix(hs, op, subsys;
                                           op_in_bare_eigenbasis=false)
    else
        _operator_matrix(op_callable_or_tuple)
    end
    n = truncated_dim === nothing ? size(U, 2) : min(truncated_dim, size(U, 2))
    U_trunc = U[:, 1:n]
    return U_trunc' * op_mat * U_trunc
end

function _state_label_to_dressed_index(hs::HilbertSpace, state_label)
    if state_label isa Integer
        idx = Int(state_label)
        1 <= idx <= length(hs.lookup.dressed_evals) || throw(BoundsError(hs.lookup.dressed_evals, idx))
        return idx
    elseif state_label isa Tuple{Vararg{Int}}
        idx = dressed_index(hs, state_label)
        idx === nothing && throw(ArgumentError(
            "No dressed state is labeled by bare state $(state_label) in the current lookup"))
        return idx
    else
        throw(ArgumentError(
            "state_label must be a dressed index (Int) or bare-state tuple of Int"))
    end
end

function _bare_product_state_amplitudes(dressed_vec::AbstractVector,
                                        bare_evecs::AbstractVector)
    sub_dims = [size(vecs, 2) for vecs in bare_evecs]
    amplitudes = Vector{Pair{Tuple, ComplexF64}}()
    for bare_idx in _generate_bare_indices(sub_dims)
        bare_state = bare_evecs[1][:, bare_idx[1]]
        for k in 2:length(bare_evecs)
            bare_state = kron(bare_state, bare_evecs[k][:, bare_idx[k]])
        end
        bare_tup = Tuple(bare_idx)
        amp = dot(bare_state, dressed_vec)
        push!(amplitudes, bare_tup => amp)
    end
    return amplitudes
end

function _hilbertspace_bare_evecs(hs::HilbertSpace)
    bare_evecs = Matrix{ComplexF64}[]
    for subsys in hs.subsystems
        _, vecs = eigensys(subsys; evals_count=hilbertdim(subsys))
        push!(bare_evecs, vecs)
    end
    return bare_evecs
end

function _sort_component_pairs(components::Vector{Pair{Tuple, T}},
                               components_count::Union{Nothing, Int}=nothing) where {T}
    sorted = sort(components; by=pair -> abs2(pair.second), rev=true)
    if components_count !== nothing
        components_count >= 0 || throw(ArgumentError("components_count must be nonnegative"))
        resize!(sorted, min(components_count, length(sorted)))
    end
    return sorted
end

"""
    dressed_state_components(hs::HilbertSpace, state_label;
                             components_count=nothing,
                             return_probability=true)

Return the bare-state composition of a dressed state from an existing lookup on
`hs`.

`state_label` may be a dressed-state index or a bare-state tuple. When
`return_probability=true`, the result contains probabilities; otherwise it
contains complex amplitudes. The returned pairs are sorted in descending order
of weight.
"""
function dressed_state_components(hs::HilbertSpace, state_label;
                                  components_count::Union{Nothing, Int}=nothing,
                                  return_probability::Bool=true)
    hs.lookup === nothing && error("Call generate_lookup!(hs) first")
    dressed_idx = _state_label_to_dressed_index(hs, state_label)

    if return_probability
        probabilities = Pair{Tuple, Float64}[]
        for (bare_idx, prob) in zip(_generate_bare_indices([length(vals) for vals in hs.lookup.bare_evals]),
                                    hs.lookup.overlap_matrix[dressed_idx, :])
            push!(probabilities, Tuple(bare_idx) => prob)
        end
        return _sort_component_pairs(probabilities, components_count)
    end

    bare_evecs = _hilbertspace_bare_evecs(hs)
    amplitudes = _bare_product_state_amplitudes(hs.lookup.dressed_evecs[:, dressed_idx], bare_evecs)
    return _sort_component_pairs(amplitudes, components_count)
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

function _canonical_dressed_indices(lookup::SpectrumLookup)
    sub_dims = Int[length(vals) for vals in lookup.bare_evals]
    dressed_indices = Vector{Union{Nothing, Int}}(undef, prod(sub_dims))
    for (i, bare_idx) in enumerate(_generate_bare_indices(sub_dims))
        dressed_indices[i] = get(lookup.bare_to_dressed, Tuple(bare_idx), nothing)
    end
    return dressed_indices
end
