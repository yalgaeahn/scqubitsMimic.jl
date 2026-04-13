# ──────────────────────────────────────────────────────────────────────────────
# Parameter sweeps
# ──────────────────────────────────────────────────────────────────────────────

"""
    SingleSystemSweep

Parameter sweep over a single quantum system, computing spectra at each
parameter value.

This is a Julia-local helper for bare single-system sweeps. The scqubits-style
composite-system sweep API is [`ParameterSweep`](@ref).
"""
struct SingleSystemSweep
    system::AbstractQuantumSystem
    param_name::Symbol
    param_vals::Vector{Float64}
    spectrum::SpectrumData
end

function SingleSystemSweep(sys::AbstractQuantumSystem, param_name::Symbol,
                           param_vals::AbstractVector;
                           evals_count::Int=6, store_eigvecs::Bool=false)
    spectrum = get_spectrum_vs_paramvals(sys, param_name, param_vals;
                                         evals_count=evals_count,
                                         store_eigvecs=store_eigvecs)
    return SingleSystemSweep(sys, param_name, collect(Float64, param_vals), spectrum)
end

function SingleSystemSweep(sys::AbstractQuantumSystem, param_name::AbstractString,
                           param_vals::AbstractVector;
                           evals_count::Int=6, store_eigvecs::Bool=false)
    SingleSystemSweep(sys, Symbol(param_name), param_vals;
                      evals_count=evals_count,
                      store_eigvecs=store_eigvecs)
end

"""
    ParameterSweep

Composite Hilbert-space parameter sweep with scqubits-style naming.

This is the public sweep object for dressed-spectrum, lookup, and dispersive
analysis on coupled systems. Julia keeps the lighter bare single-system helper
separate as [`SingleSystemSweep`](@ref).
"""
mutable struct ParameterSweep
    hilbertspace::HilbertSpace
    param_order::Vector{Symbol}
    param_vals::Dict{Symbol, Vector{Float64}}
    update_hilbertspace::Function
    evals_count::Int
    dressed_evals::Union{Nothing, Matrix{Float64}}
    bare_evals::Union{Nothing, Vector{Vector{Vector{Float64}}}}
    bare_evecs::Union{Nothing, Vector{Vector{Matrix{ComplexF64}}}}
    dressed_evecs::Union{Nothing, Vector{Matrix{ComplexF64}}}
    dressed_indices::Union{Nothing, Vector{Vector{Union{Nothing, Int}}}}
    lookups::Union{Nothing, Vector{SpectrumLookup}}
    ignore_low_overlap::Bool
    labeling_scheme::Symbol
    labeling_subsys_priority::Union{Nothing, Vector{Int}}
    labeling_BEs_count::Union{Nothing, Int}
    bare_only::Bool
    subsys_update_info::Union{Nothing, Dict{Symbol, Vector{Int}}}
    store_lookups::Bool
end

function _normalize_paramvals_by_name(paramvals_by_name::AbstractDict)
    param_order = Symbol[]
    param_vals = Dict{Symbol, Vector{Float64}}()
    for (name, vals) in pairs(paramvals_by_name)
        sym_name = Symbol(name)
        push!(param_order, sym_name)
        param_vals[sym_name] = collect(Float64, vals)
    end
    isempty(param_order) && throw(ArgumentError("paramvals_by_name must not be empty"))
    return param_order, param_vals
end

function _normalize_subsys_update_info(hs::HilbertSpace, subsys_update_info)
    subsys_update_info === nothing && return nothing

    normalized = Dict{Symbol, Vector{Int}}()
    for (param_name, subsys_list) in pairs(subsys_update_info)
        indices = Int[]
        for subsys in subsys_list
            idx = if subsys isa Integer
                Int(subsys)
            elseif subsys isa AbstractQuantumSystem
                findfirst(candidate -> candidate === subsys, hs.subsystems)
            else
                throw(ArgumentError(
                    "subsys_update_info values must contain subsystem indices or subsystem objects"))
            end
            idx === nothing && throw(ArgumentError(
                "subsystem in subsys_update_info not found in the provided HilbertSpace"))
            1 <= idx <= length(hs.subsystems) || throw(BoundsError(hs.subsystems, idx))
            push!(indices, idx)
        end
        normalized[Symbol(param_name)] = sort!(unique(indices))
    end
    return normalized
end

function _parameter_point_count(param_order::Vector{Symbol},
                                param_vals::Dict{Symbol, Vector{Float64}})
    return prod(length(param_vals[name]) for name in param_order)
end

function _parameter_product(param_order::Vector{Symbol},
                            param_vals::Dict{Symbol, Vector{Float64}})
    arrays = (param_vals[name] for name in param_order)
    return Iterators.product(arrays...)
end

function _call_update_hilbertspace!(sweep::ParameterSweep,
                                    current::Dict{Symbol, Float64})
    vals = [current[name] for name in sweep.param_order]
    update = sweep.update_hilbertspace
    hs = sweep.hilbertspace

    if applicable(update, hs, current)
        update(hs, current)
    elseif applicable(update, sweep, vals...)
        update(sweep, vals...)
    elseif applicable(update, vals...)
        update(vals...)
    elseif applicable(update, hs, vals...)
        update(hs, vals...)
    else
        throw(ArgumentError(
            "update_hilbertspace must accept either `(hs, param_dict)`, " *
            "`(sweep, paramval1, ...)`, `(paramval1, ...)`, or `(hs, paramval1, ...)`."))
    end
    return nothing
end

function _affected_subsystems(sweep::ParameterSweep,
                              previous::Union{Nothing, Dict{Symbol, Float64}},
                              current::Dict{Symbol, Float64})
    n_sub = length(sweep.hilbertspace.subsystems)
    if previous === nothing || sweep.subsys_update_info === nothing
        return collect(1:n_sub)
    end

    changed = Symbol[
        name for name in sweep.param_order if !isequal(previous[name], current[name])
    ]
    isempty(changed) && return Int[]

    any(!haskey(sweep.subsys_update_info, name) for name in changed) &&
        return collect(1:n_sub)

    affected = Int[]
    for name in changed
        append!(affected, sweep.subsys_update_info[name])
    end
    return sort!(unique(affected))
end

function _parameter_dims_tuple(sweep::ParameterSweep)
    return Tuple(length(sweep.param_vals[name]) for name in sweep.param_order)
end

function _parameter_point_index(sweep::ParameterSweep,
                                param_indices::Tuple{Vararg{Int}})
    dims = _parameter_dims_tuple(sweep)
    length(param_indices) == length(dims) || throw(ArgumentError(
        "param_indices must have length $(length(dims)) matching sweep.param_order=$(sweep.param_order); got $param_indices"))
    for (idx, dim) in zip(param_indices, dims)
        1 <= idx <= dim || throw(BoundsError(dims, param_indices))
    end
    return LinearIndices(dims)[param_indices...]
end

function _normalize_sweep_lookup_options(sweep::ParameterSweep;
                                         ordering::Union{Symbol, AbstractString}=sweep.labeling_scheme,
                                         subsys_priority::Union{Nothing, AbstractVector{<:Integer}}=sweep.labeling_subsys_priority,
                                         BEs_count::Union{Nothing, Int}=sweep.labeling_BEs_count,
                                         overlap_threshold::Union{Nothing, Float64}=nothing)
    normalized_ordering = _normalize_lookup_ordering(ordering)
    normalized_priority = subsys_priority === nothing ? nothing : Int.(collect(subsys_priority))
    normalized_BEs_count = BEs_count
    normalized_overlap_threshold = overlap_threshold

    if normalized_ordering == :DE
        subsys_priority !== nothing &&
            @warn "subsys_priority is ignored for DE ordering."
        BEs_count !== nothing &&
            @warn "BEs_count is ignored for DE ordering."
        normalized_priority = nothing
        normalized_BEs_count = nothing
    elseif normalized_ordering == :LX
        _validate_subsys_priority(sweep.hilbertspace, normalized_priority)
        BEs_count !== nothing &&
            @warn "BEs_count is ignored for LX ordering."
        overlap_threshold !== nothing &&
            @warn "overlap_threshold is ignored for LX ordering."
        normalized_BEs_count = nothing
        normalized_overlap_threshold = nothing
    else
        _validate_subsys_priority(sweep.hilbertspace, normalized_priority)
        overlap_threshold !== nothing &&
            @warn "overlap_threshold is ignored for BE ordering."
        normalized_overlap_threshold = nothing
        normalized_BEs_count !== nothing && normalized_BEs_count < 1 && throw(ArgumentError(
            "BEs_count must be positive; got $normalized_BEs_count"))
    end

    return (; ordering=normalized_ordering,
            subsys_priority=normalized_priority,
            BEs_count=normalized_BEs_count,
            overlap_threshold=normalized_overlap_threshold)
end

function _ensure_sweep_lookup_buildable(sweep::ParameterSweep)
    sweep.bare_only && throw(ArgumentError(
        "Lookup generation requires dressed spectral data; this sweep was created with bare_only=true."))
    sweep.dressed_evals === nothing && throw(ArgumentError(
        "Call run!(sweep) before generating lookup data."))
    sweep.dressed_evecs === nothing && throw(ArgumentError(
        "Dressed eigenvectors are unavailable on this sweep; rerun with bare_only=false."))
    sweep.bare_evals === nothing && throw(ArgumentError(
        "Bare eigenvalue data are unavailable on this sweep; call run!(sweep) first."))
    sweep.bare_evecs === nothing && throw(ArgumentError(
        "Bare eigenvector data are unavailable on this sweep; call run!(sweep) first."))
    return nothing
end

function _build_sweep_lookup_at_point(sweep::ParameterSweep,
                                      point_index::Int;
                                      ordering::Union{Symbol, AbstractString}=sweep.labeling_scheme,
                                      subsys_priority::Union{Nothing, AbstractVector{<:Integer}}=sweep.labeling_subsys_priority,
                                      BEs_count::Union{Nothing, Int}=sweep.labeling_BEs_count,
                                      overlap_threshold::Union{Nothing, Float64}=nothing)
    opts = _normalize_sweep_lookup_options(sweep;
        ordering=ordering,
        subsys_priority=subsys_priority,
        BEs_count=BEs_count,
        overlap_threshold=overlap_threshold)

    data = _lookup_build_data_from_spectral_data(
        sweep.hilbertspace,
        @view(sweep.dressed_evals[point_index, :]),
        sweep.dressed_evecs[point_index],
        sweep.bare_evals[point_index],
        sweep.bare_evecs[point_index],
    )

    return _build_lookup_from_data(sweep.hilbertspace, data;
        ordering=opts.ordering,
        subsys_priority=opts.subsys_priority,
        BEs_count=opts.BEs_count,
        overlap_threshold=opts.overlap_threshold,
        ignore_low_overlap=sweep.ignore_low_overlap)
end

function _generate_sweep_lookups(sweep::ParameterSweep;
                                 ordering::Union{Symbol, AbstractString}=sweep.labeling_scheme,
                                 subsys_priority::Union{Nothing, AbstractVector{<:Integer}}=sweep.labeling_subsys_priority,
                                 BEs_count::Union{Nothing, Int}=sweep.labeling_BEs_count,
                                 overlap_threshold::Union{Nothing, Float64}=nothing)
    _ensure_sweep_lookup_buildable(sweep)

    n_points = _parameter_point_count(sweep.param_order, sweep.param_vals)
    lookups_vec = Vector{SpectrumLookup}(undef, n_points)
    dressed_indices_all = Vector{Vector{Union{Nothing, Int}}}(undef, n_points)

    for point_index in 1:n_points
        lookup = _build_sweep_lookup_at_point(sweep, point_index;
            ordering=ordering,
            subsys_priority=subsys_priority,
            BEs_count=BEs_count,
            overlap_threshold=overlap_threshold)
        lookups_vec[point_index] = lookup
        dressed_indices_all[point_index] = _canonical_dressed_indices(lookup)
    end

    return lookups_vec, dressed_indices_all
end

"""
    ParameterSweep(hilbertspace, paramvals_by_name, update_hilbertspace;
                   evals_count=20, subsys_update_info=nothing,
                   bare_only=false, labeling_scheme=:DE,
                   labeling_subsys_priority=nothing,
                   labeling_BEs_count=nothing,
                   ignore_low_overlap=false, autorun=true,
                   deepcopy=false, store_lookups=nothing)

Composite-system parameter sweep using scqubits-style naming.

The callback `update_hilbertspace` may use either the existing Julia form
`(hs, param_dict)` or a scqubits-style positional form
`(sweep, paramval1, ...)` / `(paramval1, ...)`.

This matches the scqubits public name for coupled-system sweeps. Julia keeps
single-system spectra under [`SingleSystemSweep`](@ref). The current type
provides naming and core analysis-flow parity, but not the full scqubits
preslicing or dict-like data-container interface.
"""
function ParameterSweep(hilbertspace::HilbertSpace,
                        paramvals_by_name::AbstractDict,
                        update_hilbertspace::Function;
                        evals_count::Int=20,
                        subsys_update_info=nothing,
                        bare_only::Bool=false,
                        labeling_scheme::Union{Symbol, AbstractString}=:DE,
                        labeling_subsys_priority::Union{Nothing, AbstractVector{<:Integer}}=nothing,
                        labeling_BEs_count::Union{Nothing, Int}=nothing,
                        ignore_low_overlap::Bool=false,
                        autorun::Bool=true,
                        deepcopy::Bool=false,
                        store_lookups::Union{Nothing, Bool}=nothing)
    param_order, param_vals = _normalize_paramvals_by_name(paramvals_by_name)
    normalized_update_info = _normalize_subsys_update_info(hilbertspace, subsys_update_info)
    normalized_scheme = _normalize_lookup_ordering(labeling_scheme)
    normalized_priority = labeling_subsys_priority === nothing ? nothing :
        _validate_subsys_priority(hilbertspace, labeling_subsys_priority)
    resolved_store_lookups = isnothing(store_lookups) ? !bare_only : store_lookups
    bare_only && resolved_store_lookups && throw(ArgumentError(
        "bare_only=true cannot be combined with store_lookups=true"))

    if normalized_update_info !== nothing
        extra_keys = setdiff(collect(keys(normalized_update_info)), param_order)
        isempty(extra_keys) || throw(ArgumentError(
            "subsys_update_info contains parameters not present in paramvals_by_name: $extra_keys"))
    end

    work_hs = deepcopy ? Base.deepcopy(hilbertspace) : hilbertspace
    sweep = ParameterSweep(
        work_hs,
        param_order,
        param_vals,
        update_hilbertspace,
        evals_count,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        ignore_low_overlap,
        normalized_scheme,
        normalized_priority,
        labeling_BEs_count,
        bare_only,
        normalized_update_info,
        resolved_store_lookups,
    )

    autorun && run!(sweep)
    return sweep
end

"""
    run!(sweep::ParameterSweep)

Execute a [`ParameterSweep`](@ref) in place and populate its stored spectral
data.

The sweep callback is evaluated at every point in the Cartesian product of
`param_vals`. Bare subsystem eigendata are always cached; dressed eigendata and
lookup tables are populated according to `bare_only` and `store_lookups`.
Returns the mutated `sweep`.
"""
function run!(sweep::ParameterSweep)
    n_points = _parameter_point_count(sweep.param_order, sweep.param_vals)
    n_sub = length(sweep.hilbertspace.subsystems)

    dressed_evals = sweep.bare_only ? nothing : Matrix{Float64}(undef, n_points, sweep.evals_count)
    bare_evals_all = Vector{Vector{Vector{Float64}}}(undef, n_points)
    bare_evecs_all = Vector{Vector{Matrix{ComplexF64}}}(undef, n_points)
    dressed_evecs_all = sweep.bare_only ? nothing : Vector{Matrix{ComplexF64}}(undef, n_points)

    cached_bare_evals = Vector{Vector{Float64}}(undef, n_sub)
    cached_bare_evecs = Vector{Matrix{ComplexF64}}(undef, n_sub)
    previous = nothing
    point_index = 1

    for combo in _parameter_product(sweep.param_order, sweep.param_vals)
        current = Dict{Symbol, Float64}(sweep.param_order[i] => combo[i] for i in eachindex(sweep.param_order))
        _call_update_hilbertspace!(sweep, current)

        affected = _affected_subsystems(sweep, previous, current)
        for subsys_idx in affected
            subsystem = sweep.hilbertspace.subsystems[subsys_idx]
            vals, vecs = eigensys(subsystem; evals_count=hilbertdim(subsystem))
            cached_bare_evals[subsys_idx] = vals
            cached_bare_evecs[subsys_idx] = vecs
        end
        previous === nothing && any(!isassigned(cached_bare_evals, i) for i in eachindex(cached_bare_evals)) &&
            throw(ArgumentError("Initial bare eigensystem cache was not fully populated"))
        previous === nothing && any(!isassigned(cached_bare_evecs, i) for i in eachindex(cached_bare_evecs)) &&
            throw(ArgumentError("Initial bare eigenvector cache was not fully populated"))

        bare_evals_all[point_index] = [cached_bare_evals[i] for i in 1:n_sub]
        bare_evecs_all[point_index] = [cached_bare_evecs[i] for i in 1:n_sub]

        if !sweep.bare_only
            vals, vecs = eigensys(sweep.hilbertspace; evals_count=sweep.evals_count)
            dressed_evals[point_index, :] .= vals
            dressed_evecs_all[point_index] = vecs
        end

        previous = current
        point_index += 1
    end

    sweep.dressed_evals = dressed_evals
    sweep.bare_evals = bare_evals_all
    sweep.bare_evecs = bare_evecs_all
    sweep.dressed_evecs = dressed_evecs_all

    if sweep.store_lookups && !sweep.bare_only
        sweep.lookups, sweep.dressed_indices = _generate_sweep_lookups(sweep;
            ordering=sweep.labeling_scheme,
            subsys_priority=sweep.labeling_subsys_priority,
            BEs_count=sweep.labeling_BEs_count)
    else
        sweep.dressed_indices = nothing
        sweep.lookups = nothing
    end
    return sweep
end

"""
    generate_lookup!(sweep::ParameterSweep; ordering=:DE,
                     subsys_priority=nothing, BEs_count=nothing,
                     overlap_threshold=nothing)

Build or rebuild sweep lookup data from the spectral data already stored on
`sweep`. `run!(sweep)` must have completed with dressed and bare eigendata
available.
"""
function generate_lookup!(sweep::ParameterSweep;
                          ordering::Union{Symbol, AbstractString}=sweep.labeling_scheme,
                          subsys_priority::Union{Nothing, AbstractVector{<:Integer}}=sweep.labeling_subsys_priority,
                          BEs_count::Union{Nothing, Int}=sweep.labeling_BEs_count,
                          overlap_threshold::Union{Nothing, Float64}=nothing)
    lookups_vec, dressed_indices_all = _generate_sweep_lookups(sweep;
        ordering=ordering,
        subsys_priority=subsys_priority,
        BEs_count=BEs_count,
        overlap_threshold=overlap_threshold)
    sweep.lookups = lookups_vec
    sweep.dressed_indices = dressed_indices_all
    return lookups_vec
end

"""
    lookup_exists(sweep::ParameterSweep)

Return `true` when lookup data have been built for every stored parameter
point of `sweep`.
"""
function lookup_exists(sweep::ParameterSweep)
    n_points = _parameter_point_count(sweep.param_order, sweep.param_vals)
    return sweep.lookups !== nothing &&
           sweep.dressed_indices !== nothing &&
           length(sweep.lookups) == n_points &&
           length(sweep.dressed_indices) == n_points
end

function _require_sweep_lookup(sweep::ParameterSweep)
    lookup_exists(sweep) && return nothing
    error("Sweep lookup data are unavailable. Create the sweep with store_lookups=true or call generate_lookup!(sweep) after run!(sweep).")
end

"""
    dressed_index(sweep::ParameterSweep, bare_labels...; param_indices)

Return the dressed-state index associated with `bare_labels` at a specific
parameter point of `sweep`.

`param_indices` selects the point in the Cartesian product ordering of the
sweep parameters and follows Julia's 1-based indexing conventions.
"""
function dressed_index(sweep::ParameterSweep, bare_labels::Tuple{Vararg{Int}};
                       param_indices::Tuple{Vararg{Int}})
    _require_sweep_lookup(sweep)
    point_index = _parameter_point_index(sweep, param_indices)
    return _lookup_dressed_index(sweep.lookups[point_index], bare_labels)
end

function dressed_index(sweep::ParameterSweep, bare_labels::Int...;
                       param_indices::Tuple{Vararg{Int}})
    return dressed_index(sweep, Tuple(bare_labels); param_indices=param_indices)
end

"""
    bare_index(sweep::ParameterSweep, dressed_idx::Int; param_indices)

Return the bare-state label tuple assigned to `dressed_idx` at a specific
parameter point of `sweep`.
"""
function bare_index(sweep::ParameterSweep, dressed_idx::Int;
                    param_indices::Tuple{Vararg{Int}})
    _require_sweep_lookup(sweep)
    point_index = _parameter_point_index(sweep, param_indices)
    return _lookup_bare_index(sweep.lookups[point_index], dressed_idx)
end

"""
    energy_by_dressed_index(sweep::ParameterSweep, dressed_idx::Int;
                            param_indices, subtract_ground=false)

Return the dressed energy at `dressed_idx` for the parameter point selected by
`param_indices`.
"""
function energy_by_dressed_index(sweep::ParameterSweep, dressed_idx::Int;
                                 param_indices::Tuple{Vararg{Int}},
                                 subtract_ground::Bool=false)
    _require_sweep_lookup(sweep)
    point_index = _parameter_point_index(sweep, param_indices)
    n_evals = size(sweep.dressed_evals, 2)
    1 <= dressed_idx <= n_evals || throw(BoundsError(axes(sweep.dressed_evals, 2), dressed_idx))
    energy = sweep.dressed_evals[point_index, dressed_idx]
    return subtract_ground ? energy - sweep.dressed_evals[point_index, 1] : energy
end

"""
    energy_by_bare_index(sweep::ParameterSweep, bare_labels...;
                         param_indices, subtract_ground=false)

Return the dressed energy associated with `bare_labels` at the parameter point
selected by `param_indices`.

Returns `NaN` when the requested bare label is not present in the lookup at
that parameter point.
"""
function energy_by_bare_index(sweep::ParameterSweep, bare_labels::Tuple{Vararg{Int}};
                              param_indices::Tuple{Vararg{Int}},
                              subtract_ground::Bool=false)
    _require_sweep_lookup(sweep)
    point_index = _parameter_point_index(sweep, param_indices)
    return _lookup_energy_by_bare_index(sweep.lookups[point_index], bare_labels;
                                        subtract_ground=subtract_ground)
end

function energy_by_bare_index(sweep::ParameterSweep, bare_labels::Int...;
                              param_indices::Tuple{Vararg{Int}},
                              subtract_ground::Bool=false)
    return energy_by_bare_index(sweep, Tuple(bare_labels);
                                param_indices=param_indices,
                                subtract_ground=subtract_ground)
end
