# ──────────────────────────────────────────────────────────────────────────────
# Sweep slicing and transition/state-label analysis
# ──────────────────────────────────────────────────────────────────────────────

"""
    SweepSlice

View of a [`ParameterSweep`](@ref) restricted to a subset of parameter points.

`SweepSlice` is produced by indexing a sweep, for example `sweep[:, 3]`, and is
used by transition-analysis helpers such as [`transitions`](@ref) and
[`dressed_state_components`](@ref).
"""
struct SweepSlice
    parent::ParameterSweep
    selectors::Tuple
    selected_indices::Vector{Vector{Int}}
    point_param_indices::Vector{Tuple}
    point_indices::Vector{Int}
    free_dims::Vector{Int}
    param_name::Union{Nothing, Symbol}
    param_vals::Union{Nothing, Vector{Float64}}
end

function _normalize_slice_selector(dim::Int, selector)
    if selector isa Colon
        return collect(1:dim)
    elseif selector isa Integer
        idx = Int(selector)
        1 <= idx <= dim || throw(BoundsError(1:dim, idx))
        return [idx]
    elseif selector isa AbstractRange{<:Integer} || selector isa AbstractVector{<:Integer}
        indices = Int.(collect(selector))
        isempty(indices) && throw(ArgumentError("slice selector must not be empty"))
        all(1 .<= indices .<= dim) || throw(BoundsError(1:dim, indices))
        return indices
    else
        throw(ArgumentError(
            "unsupported sweep selector $(repr(selector)); use Int, Colon(), integer range, or integer vector"))
    end
end

function _slice_param_index_tuples(selected_indices::Vector{Vector{Int}})
    tuples = Tuple[]
    function _recurse(level::Int, current::Vector{Int})
        if level > length(selected_indices)
            push!(tuples, Tuple(current))
            return
        end
        for idx in selected_indices[level]
            push!(current, idx)
            _recurse(level + 1, current)
            pop!(current)
        end
    end
    _recurse(1, Int[])
    return tuples
end

function SweepSlice(parent::ParameterSweep, selectors::Tuple)
    length(selectors) == length(parent.param_order) || throw(ArgumentError(
        "expected $(length(parent.param_order)) sweep selectors for parameters $(parent.param_order); got $(length(selectors))"))
    selected_indices = [
        _normalize_slice_selector(length(parent.param_vals[parent.param_order[i]]), selectors[i])
        for i in eachindex(parent.param_order)
    ]
    point_param_indices = _slice_param_index_tuples(selected_indices)
    point_indices = [_parameter_point_index(parent, param_idx) for param_idx in point_param_indices]
    free_dims = Int.(findall(indices -> length(indices) > 1, selected_indices))
    param_name = length(free_dims) == 1 ? parent.param_order[only(free_dims)] : nothing
    param_vals = isnothing(param_name) ? nothing : parent.param_vals[param_name][selected_indices[only(free_dims)]]
    return SweepSlice(parent, selectors, selected_indices, point_param_indices, point_indices,
                      free_dims, param_name, param_vals)
end

function Base.getindex(sweep::ParameterSweep, selectors...)
    if length(selectors) == 1 && length(sweep.param_order) == 1
        return SweepSlice(sweep, (selectors[1],))
    end
    return SweepSlice(sweep, selectors)
end

function full_slice(sweep::ParameterSweep)
    return SweepSlice(sweep, ntuple(_ -> Colon(), length(sweep.param_order)))
end

function slice_param_axis(slice::SweepSlice)
    length(slice.free_dims) == 1 || throw(ArgumentError(
        "transition extraction requires exactly one swept dimension; got selectors $(slice.selectors)"))
    slice.param_name !== nothing || throw(ArgumentError("slice parameter metadata are unavailable"))
    slice.param_vals !== nothing || throw(ArgumentError("slice parameter values are unavailable"))
    return slice.param_name, slice.param_vals
end

function _fixed_slice_param_indices(slice::SweepSlice)
    isempty(slice.free_dims) || throw(ArgumentError(
        "this operation requires a fixed-point slice; got selectors $(slice.selectors)"))
    length(slice.point_param_indices) == 1 || throw(ArgumentError(
        "fixed-point slice resolved to multiple parameter points"))
    return slice.point_param_indices[1]
end

function resolve_subsystem_indices(hs::HilbertSpace, subsystems)
    if subsystems === nothing
        return collect(1:length(hs.subsystems))
    elseif subsystems isa Integer
        idx = Int(subsystems)
        1 <= idx <= length(hs.subsystems) || throw(BoundsError(hs.subsystems, idx))
        return [idx]
    elseif subsystems isa AbstractQuantumSystem
        idx = findfirst(candidate -> candidate === subsystems, hs.subsystems)
        idx === nothing && throw(ArgumentError("Subsystem not found in the provided HilbertSpace"))
        return [idx]
    elseif subsystems isa AbstractVector
        indices = Int[]
        for subsys in subsystems
            append!(indices, resolve_subsystem_indices(hs, subsys))
        end
        return sort!(unique(indices))
    else
        throw(ArgumentError(
            "subsystems must be a subsystem object, subsystem index, or a vector of them"))
    end
end

function _complete_bare_state(hs::HilbertSpace, partial_state::Tuple{Vararg{Int}},
                              subsys_indices::Vector{Int})
    if length(partial_state) == length(hs.subsystems)
        return partial_state
    end
    length(partial_state) == length(subsys_indices) || throw(ArgumentError(
        "bare state tuple length $(length(partial_state)) is incompatible with active subsystems $subsys_indices"))
    full_state = fill(1, length(hs.subsystems))
    for (entry, subsys_idx) in zip(partial_state, subsys_indices)
        full_state[subsys_idx] = entry
    end
    return Tuple(full_state)
end

function _validate_bare_state(hs::HilbertSpace, bare_state::Tuple{Vararg{Int}})
    length(bare_state) == length(hs.subsystems) || throw(ArgumentError(
        "bare state tuple length $(length(bare_state)) does not match subsystem count $(length(hs.subsystems))"))
    for (subsys_idx, level) in enumerate(bare_state)
        1 <= level <= hilbertdim(hs.subsystems[subsys_idx]) || throw(BoundsError(
            1:hilbertdim(hs.subsystems[subsys_idx]), level))
    end
    return bare_state
end

function _normalize_state_spec(hs::HilbertSpace, state, subsys_indices::Vector{Int};
                               allow_default_ground::Bool=false)
    if state === nothing
        allow_default_ground || throw(ArgumentError("state specification cannot be `nothing` here"))
        return false, ntuple(_ -> 1, length(hs.subsystems))
    elseif state isa Integer
        return true, Int(state)
    elseif state isa Tuple{Vararg{Int}}
        return false, _validate_bare_state(hs, _complete_bare_state(hs, state, subsys_indices))
    else
        throw(ArgumentError(
            "state specifications must be dressed indices (Int) or bare-state tuples of Int"))
    end
end

function _normalize_state_specs(hs::HilbertSpace, state_specs, subsys_indices::Vector{Int};
                                allow_default_ground::Bool=false)
    if state_specs isa AbstractVector && !(state_specs isa Tuple)
        return [_normalize_state_spec(hs, state, subsys_indices;
                                      allow_default_ground=allow_default_ground) for state in state_specs]
    end
    return [_normalize_state_spec(hs, state_specs, subsys_indices;
                                  allow_default_ground=allow_default_ground)]
end

function _state_energies(slice::SweepSlice, state, dressed::Bool;
                         subtract_ground::Bool=false)
    parent = slice.parent
    energies = Vector{Float64}(undef, length(slice.point_param_indices))
    for (i, param_idx) in enumerate(slice.point_param_indices)
        energies[i] = dressed ?
            energy_by_dressed_index(parent, state; param_indices=param_idx,
                                    subtract_ground=subtract_ground) :
            energy_by_bare_index(parent, state; param_indices=param_idx,
                                 subtract_ground=subtract_ground)
    end
    return energies
end

function _state_label_string(state)
    return state isa Tuple ? string(state) : "d$(state)"
end

function _transition_label(initial_state, final_state)
    return string(_state_label_string(initial_state), "→", _state_label_string(final_state))
end

function _final_states_for_subsys_transition(hs::HilbertSpace, subsys_idx::Int,
                                             initial_state::Tuple{Vararg{Int}})
    final_states = Tuple[]
    for level in 1:hilbertdim(hs.subsystems[subsys_idx])
        candidate = collect(initial_state)
        candidate[subsys_idx] = level
        candidate_tuple = Tuple(candidate)
        candidate_tuple == initial_state && continue
        push!(final_states, candidate_tuple)
    end
    return final_states
end

function _get_final_states_list(hs::HilbertSpace, initial_state;
                                initial_dressed::Bool,
                                subsys_indices::Vector{Int},
                                sidebands::Bool,
                                evals_count::Int)
    if initial_dressed
        return [(true, idx) for idx in 1:evals_count]
    end

    if !sidebands
        final_states = Tuple[]
        for subsys_idx in subsys_indices
            append!(final_states, _final_states_for_subsys_transition(hs, subsys_idx, initial_state))
        end
        return [(false, state) for state in sort!(unique(final_states))]
    end

    range_list = [1:hilbertdim(subsys) for subsys in hs.subsystems]
    for subsys_idx in setdiff(collect(1:length(hs.subsystems)), subsys_indices)
        range_list[subsys_idx] = initial_state[subsys_idx]:initial_state[subsys_idx]
    end
    final_states = Tuple[]
    for state in Iterators.product(range_list...)
        state_tuple = Tuple(Int.(collect(state)))
        state_tuple == initial_state && continue
        push!(final_states, state_tuple)
    end
    return [(false, state) for state in final_states]
end

function _validate_initial_labeling(slice::SweepSlice, initial_state,
                                    initial_energies::Vector{Float64})
    if any(isnan, initial_energies)
        @warn "The initial bare state undergoes significant hybridization; some transition energies are undefined under bare-label tracking."
    elseif all(level == 1 for level in initial_state)
        ground_energies = [
            slice.parent.dressed_evals[slice.point_indices[i], 1]
            for i in eachindex(slice.point_indices)
        ]
        if any(abs.(initial_energies .- ground_energies) .> 1e-9)
            @warn "The nominal bare ground state is not perfectly aligned with the true dressed ground state across this slice."
        end
    end
    return nothing
end

function transition_background(slice::SweepSlice;
                               initial=nothing,
                               subsystems=nothing,
                               photon_number::Int=1,
                               make_positive::Bool=true)
    photon_number > 0 || throw(ArgumentError("photon_number must be positive"))
    _require_sweep_lookup(slice.parent)
    hs = slice.parent.hilbertspace
    subsys_indices = resolve_subsystem_indices(hs, subsystems)
    initial_dressed, initial_state = _normalize_state_spec(
        hs, initial, subsys_indices; allow_default_ground=true)
    initial_energies = _state_energies(slice, initial_state, initial_dressed)

    all_diffs = Matrix{Float64}(undef, length(slice.point_param_indices), slice.parent.evals_count)
    for dressed_idx in 1:slice.parent.evals_count
        final_energies = _state_energies(slice, dressed_idx, true)
        diff = (final_energies .- initial_energies) ./ photon_number
        make_positive && (diff = abs.(diff))
        all_diffs[:, dressed_idx] .= diff
    end
    return all_diffs
end

"""
    transitions(slice::SweepSlice; as_specdata=false, subsystems=nothing,
                initial=nothing, final=nothing, sidebands=false,
                photon_number=1, make_positive=false)

Compute transition energies across a one-dimensional [`SweepSlice`](@ref).

By default this returns `(transition_specs, transition_energies)`. With
`as_specdata=true`, it returns a named tuple containing the swept parameter
axis, a transition-energy table, and human-readable labels.
"""
function transitions(slice::SweepSlice;
                     as_specdata::Bool=false,
                     subsystems=nothing,
                     initial=nothing,
                     final=nothing,
                     sidebands::Bool=false,
                     photon_number::Int=1,
                     make_positive::Bool=false)
    photon_number > 0 || throw(ArgumentError("photon_number must be positive"))
    _require_sweep_lookup(slice.parent)
    param_name, param_vals = slice_param_axis(slice)
    hs = slice.parent.hilbertspace
    subsys_indices = resolve_subsystem_indices(hs, subsystems)
    transition_specs = Tuple[]
    transition_energies = Vector{Vector{Float64}}()

    for (initial_dressed, initial_state) in _normalize_state_specs(hs, initial, subsys_indices;
                                                                   allow_default_ground=true)
        initial_energies = _state_energies(slice, initial_state, initial_dressed)
        !initial_dressed && _validate_initial_labeling(slice, initial_state, initial_energies)

        final_specs = if final === nothing
            _get_final_states_list(hs, initial_state;
                                   initial_dressed=initial_dressed,
                                   subsys_indices=subsys_indices,
                                   sidebands=sidebands,
                                   evals_count=slice.parent.evals_count)
        elseif final == -1
            [(true, idx) for idx in 1:slice.parent.evals_count]
        else
            _normalize_state_specs(hs, final, subsys_indices)
        end

        for (final_dressed, final_state) in final_specs
            final_energies = _state_energies(slice, final_state, final_dressed)
            diff = (final_energies .- initial_energies) ./ photon_number
            make_positive && (diff = abs.(diff))
            all(isnan, diff) && continue
            push!(transition_specs, (initial_state, final_state))
            push!(transition_energies, diff)
        end
    end

    if !as_specdata
        return transition_specs, transition_energies
    end

    labels = [_transition_label(initial_state, final_state)
              for (initial_state, final_state) in transition_specs]
    energy_table = isempty(transition_energies) ?
        zeros(length(param_vals), 0) :
        reduce(hcat, transition_energies)
    return (param_name=param_name,
            param_vals=param_vals,
            energy_table=energy_table,
            labels=labels)
end

"""
    transitions(sweep::ParameterSweep; kwargs...)

Convenience wrapper for one-dimensional sweeps. Internally this applies the
full sweep slice and forwards to `transitions(::SweepSlice; ...)`.
"""
function transitions(sweep::ParameterSweep; kwargs...)
    length(sweep.param_order) == 1 || throw(ArgumentError(
        "transitions(::ParameterSweep) only supports one-dimensional sweeps; use `sweep[...]` first"))
    return transitions(full_slice(sweep); kwargs...)
end

function _sweep_state_label_to_dressed_index(sweep::ParameterSweep, state_label;
                                             param_indices::Tuple{Vararg{Int}})
    if state_label isa Integer
        idx = Int(state_label)
        n_evals = size(sweep.dressed_evals, 2)
        1 <= idx <= n_evals || throw(BoundsError(axes(sweep.dressed_evals, 2), idx))
        return idx
    elseif state_label isa Tuple{Vararg{Int}}
        idx = dressed_index(sweep, state_label; param_indices=param_indices)
        idx === nothing && throw(ArgumentError(
            "No dressed state is labeled by bare state $(state_label) at parameter indices $(param_indices)"))
        return idx
    else
        throw(ArgumentError(
            "state_label must be a dressed index (Int) or bare-state tuple of Int"))
    end
end

"""
    dressed_state_components(sweep::ParameterSweep, state_label; param_indices,
                             components_count=nothing,
                             return_probability=true)

Return the bare-state composition of `state_label` at a single parameter point
of `sweep`.

`param_indices` selects the parameter point in Cartesian-product order. The
result is sorted in descending order of probability or amplitude magnitude.
"""
function dressed_state_components(sweep::ParameterSweep, state_label;
                                  param_indices::Tuple{Vararg{Int}},
                                  components_count::Union{Nothing, Int}=nothing,
                                  return_probability::Bool=true)
    _require_sweep_lookup(sweep)
    point_index = _parameter_point_index(sweep, param_indices)
    dressed_idx = _sweep_state_label_to_dressed_index(sweep, state_label; param_indices=param_indices)

    if return_probability
        lookup = sweep.lookups[point_index]
        probabilities = Pair{Tuple, Float64}[]
        for (bare_idx, prob) in zip(_generate_bare_indices([length(vals) for vals in lookup.bare_evals]),
                                    lookup.overlap_matrix[dressed_idx, :])
            push!(probabilities, Tuple(bare_idx) => prob)
        end
        return _sort_component_pairs(probabilities, components_count)
    end

    amplitudes = _bare_product_state_amplitudes(sweep.dressed_evecs[point_index][:, dressed_idx],
                                                sweep.bare_evecs[point_index])
    return _sort_component_pairs(amplitudes, components_count)
end

"""
    dressed_state_components(slice::SweepSlice, state_label;
                             components_count=nothing,
                             return_probability=true)

Return bare-state components for a fixed-point [`SweepSlice`](@ref). The slice
must resolve to exactly one parameter point.
"""
function dressed_state_components(slice::SweepSlice, state_label;
                                  components_count::Union{Nothing, Int}=nothing,
                                  return_probability::Bool=true)
    param_indices = _fixed_slice_param_indices(slice)
    return dressed_state_components(slice.parent, state_label;
                                    param_indices=param_indices,
                                    components_count=components_count,
                                    return_probability=return_probability)
end
