# ──────────────────────────────────────────────────────────────────────────────
# ParameterSweep: efficient multi-parameter eigenvalue computation
# ──────────────────────────────────────────────────────────────────────────────

"""
    ParameterSweep

Manages parameter sweeps over a single quantum system,
computing spectra at each parameter value.

# Example
```julia
sweep = ParameterSweep(transmon, :EJ, range(20, 40; length=50); evals_count=4)
```
"""
struct ParameterSweep
    system::AbstractQuantumSystem
    param_name::Symbol
    param_vals::Vector{Float64}
    spectrum::SpectrumData
end

function ParameterSweep(sys::AbstractQuantumSystem, param_name::Symbol,
                        param_vals::AbstractVector;
                        evals_count::Int=6, store_eigvecs::Bool=false)
    spectrum = get_spectrum_vs_paramvals(sys, param_name, param_vals;
                                         evals_count=evals_count,
                                         store_eigvecs=store_eigvecs)
    return ParameterSweep(sys, param_name, collect(Float64, param_vals), spectrum)
end

function ParameterSweep(sys::AbstractQuantumSystem, param_name::AbstractString,
                        param_vals::AbstractVector;
                        evals_count::Int=6, store_eigvecs::Bool=false)
    ParameterSweep(sys, Symbol(param_name), param_vals;
                   evals_count=evals_count,
                   store_eigvecs=store_eigvecs)
end

# ──────────────────────────────────────────────────────────────────────────────
# HilbertSpaceSweep: parameter sweep over a composite HilbertSpace
# ──────────────────────────────────────────────────────────────────────────────

"""
    HilbertSpaceSweep

Parameter sweep over a HilbertSpace with a user-provided update callback.
At each parameter point, the callback updates subsystem parameters,
the full Hamiltonian is rebuilt and diagonalized, and bare/dressed
state mappings are computed.

Stored sweep lookups use a sweep-local labeling policy controlled by
`ignore_low_overlap`; this setting is independent from
`hs.ignore_low_overlap`.

# Example
```julia
tmon = TunableTransmon(EJmax=20.0, EC=0.3, d=0.1)
osc = Oscillator(E_osc=6.0, truncated_dim=10)
hs = HilbertSpace([tmon, osc])
add_interaction!(hs, 0.1, [tmon, osc],
                 [s -> hamiltonian(s), s -> destroy(hilbertdim(s))])

sweep = HilbertSpaceSweep(hs,
    Dict(:flux => range(0, 0.5, length=101)),
    (hs, vals) -> begin
        tmon.flux = vals[:flux]
    end;
    evals_count=10
)
```

# Fields
- `hilbertspace`   — the HilbertSpace being swept
- `param_vals`     — Dict of param_name => values array
- `dressed_evals`  — dressed eigenvalues at each sweep point
- `bare_evals`     — bare eigenvalues per subsystem at each sweep point
- `lookups`        — SpectrumLookup at each sweep point (optional)
- `ignore_low_overlap` — sweep-local lookup policy used when `store_lookups=true`
"""
struct HilbertSpaceSweep
    hilbertspace::HilbertSpace
    param_vals::Dict{Symbol, Vector{Float64}}
    dressed_evals::Array{Float64}     # n_points × evals_count
    bare_evals::Vector{Vector{Vector{Float64}}}  # [point][subsys][evals]
    lookups::Union{Nothing, Vector{SpectrumLookup}}
    ignore_low_overlap::Bool
end

"""
    HilbertSpaceSweep(hs, paramvals_by_name, update_hilbertspace!;
                      evals_count=10, store_lookups=false,
                      ignore_low_overlap=false)

Perform a parameter sweep over a HilbertSpace.

# Arguments
- `hs::HilbertSpace` — the composite system
- `paramvals_by_name::Dict{Symbol, <:AbstractVector}` — parameter name → values
- `update_hilbertspace!` — callback `(hs, param_dict) -> nothing` that updates
  subsystem parameters at each sweep point. `param_dict` is
  `Dict{Symbol, Float64}` with current parameter values.
- `evals_count` — number of dressed eigenvalues to compute
- `store_lookups` — whether to store full SpectrumLookup at each point
- `ignore_low_overlap` — if `true`, force low-overlap bare↔dressed assignments
  for the stored sweep lookups only. This flag is **not** inherited from
  `hs.ignore_low_overlap`; pass `ignore_low_overlap=true` explicitly when a
  strong-hybridization sweep needs relaxed bare-label tracking.
"""
function HilbertSpaceSweep(hs::HilbertSpace,
                            paramvals_by_name::Dict{Symbol, <:AbstractVector},
                            update_hilbertspace!::Function;
                            evals_count::Int=10,
                            store_lookups::Bool=false,
                            ignore_low_overlap::Bool=false)
    # Convert to concrete types
    param_dict = Dict{Symbol, Vector{Float64}}(
        k => collect(Float64, v) for (k, v) in paramvals_by_name)

    # For now: single-parameter sweep (most common case)
    # Multi-parameter would need Cartesian product iteration
    param_names = collect(keys(param_dict))
    param_arrays = [param_dict[k] for k in param_names]

    if length(param_names) == 1
        pname = param_names[1]
        pvals = param_arrays[1]
        n_points = length(pvals)
    else
        # Multi-parameter: Cartesian product
        n_points = prod(length.(param_arrays))
    end

    dressed_evals = Matrix{Float64}(undef, n_points, evals_count)
    n_sub = length(hs.subsystems)
    bare_evals_all = Vector{Vector{Vector{Float64}}}(undef, n_points)
    lookups_vec = store_lookups ? Vector{SpectrumLookup}(undef, n_points) : nothing

    if length(param_names) == 1
        pname = param_names[1]
        pvals = param_arrays[1]

        for (idx, val) in enumerate(pvals)
            current = Dict{Symbol, Float64}(pname => val)
            update_hilbertspace!(hs, current)

            # Compute bare eigenvalues
            bare_evals_all[idx] = [eigenvals(s; evals_count=hilbertdim(s))
                                   for s in hs.subsystems]

            # Compute dressed eigenvalues
            vals = eigenvals(hs; evals_count=evals_count)
            dressed_evals[idx, :] .= vals

            # Optionally store full lookup
            if store_lookups
                lookup = _build_lookup(hs; evals_count=evals_count,
                                       ignore_low_overlap=ignore_low_overlap)
                lookups_vec[idx] = lookup
            end
        end
    else
        # Multi-parameter Cartesian product
        idx = 1
        for combo in Iterators.product(param_arrays...)
            current = Dict{Symbol, Float64}(
                param_names[i] => combo[i] for i in eachindex(param_names))
            update_hilbertspace!(hs, current)

            bare_evals_all[idx] = [eigenvals(s; evals_count=hilbertdim(s))
                                   for s in hs.subsystems]
            vals = eigenvals(hs; evals_count=evals_count)
            dressed_evals[idx, :] .= vals

            if store_lookups
                lookup = _build_lookup(hs; evals_count=evals_count,
                                       ignore_low_overlap=ignore_low_overlap)
                lookups_vec[idx] = lookup
            end
            idx += 1
        end
    end

    return HilbertSpaceSweep(hs, param_dict, dressed_evals,
                              bare_evals_all, lookups_vec, ignore_low_overlap)
end
