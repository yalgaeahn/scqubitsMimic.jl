module ScQubitsMimicMakieExt

using ScQubitsMimic
using CairoMakie

import ScQubitsMimic: plot_evals_vs_paramvals, plot_matrixelements,
                      plot_wavefunction, plot_chi_vs_paramvals, plot_transitions

# ── Eigenvalue spectrum vs parameter ────────────────────────────────────────

"""
    plot_evals_vs_paramvals(sweep::SingleSystemSweep; kwargs...) -> Figure

Plot eigenvalues vs parameter values from a single-system sweep.

# Keyword arguments
- `subtract_ground::Bool=true` — subtract ground state energy
- `evals_count::Int=nothing` — number of levels to plot (default: all)
- `fig_ax=nothing` — optional `(Figure, Axis)` to plot into
"""
function plot_evals_vs_paramvals(sweep::SingleSystemSweep;
                                  subtract_ground::Bool=true,
                                  evals_count::Union{Nothing,Int}=nothing,
                                  fig_ax=nothing)
    pvals = sweep.param_vals
    evals = copy(sweep.spectrum.eigenvalues)
    n_evals = evals_count === nothing ? size(evals, 2) : min(evals_count, size(evals, 2))

    if subtract_ground
        evals = evals .- evals[:, 1]
    end

    if fig_ax === nothing
        fig = Figure(size=(600, 400))
        ax = Axis(fig[1, 1],
                  xlabel=string(sweep.param_name),
                  ylabel=subtract_ground ? "Energy - E₀ (GHz)" : "Energy (GHz)")
    else
        fig, ax = fig_ax
    end

    for k in 1:n_evals
        lines!(ax, pvals, evals[:, k])
    end

    return fig
end

"""
    plot_evals_vs_paramvals(sweep::ParameterSweep; kwargs...) -> Figure

Plot dressed eigenvalues vs parameter from a composite ParameterSweep.
"""
function plot_evals_vs_paramvals(sweep::ParameterSweep;
                                  param_name::Union{Nothing,Symbol}=nothing,
                                  subtract_ground::Bool=true,
                                  evals_count::Union{Nothing,Int}=nothing,
                                  fig_ax=nothing)
    pname = param_name === nothing ? first(sweep.param_order) : param_name
    pvals = sweep.param_vals[pname]
    evals = copy(sweep.dressed_evals)
    n_evals = evals_count === nothing ? size(evals, 2) : min(evals_count, size(evals, 2))

    if subtract_ground
        evals = evals .- evals[:, 1]
    end

    if fig_ax === nothing
        fig = Figure(size=(600, 400))
        ax = Axis(fig[1, 1],
                  xlabel=string(pname),
                  ylabel=subtract_ground ? "Energy - E₀ (GHz)" : "Energy (GHz)")
    else
        fig, ax = fig_ax
    end

    for k in 1:n_evals
        lines!(ax, pvals, evals[:, k])
    end

    return fig
end

# ── Matrix elements heatmap ─────────────────────────────────────────────────

"""
    plot_matrixelements(sys, op; evals_count=6, mode=:abs) -> Figure

Plot matrix element heatmap |⟨i|op|j⟩| for an operator `op`.

# Arguments
- `sys` — quantum system (Transmon, Oscillator, etc.)
- `op` — operator (QuantumObject or function `sys -> QuantumObject`)
- `mode` — `:abs`, `:real`, `:imag`, or `:abs_sqr`
"""
function plot_matrixelements(sys::ScQubitsMimic.AbstractQuantumSystem, op;
                              evals_count::Int=6,
                              mode::Symbol=:abs)
    op_qobj = op isa Function ? op(sys) : op
    table = ScQubitsMimic.matrixelement_table(sys, op_qobj; evals_count=evals_count)

    data = if mode == :real
        real.(table)
    elseif mode == :imag
        imag.(table)
    elseif mode == :abs_sqr
        abs2.(table)
    else
        abs.(table)
    end

    fig = Figure(size=(500, 450))
    ax = Axis(fig[1, 1], xlabel="j", ylabel="i",
              title="Matrix elements ($(mode))",
              aspect=DataAspect())
    hm = heatmap!(ax, 0:evals_count-1, 0:evals_count-1, data)
    Colorbar(fig[1, 2], hm)

    return fig
end

# ── Wavefunction plot ───────────────────────────────────────────────────────

"""
    plot_wavefunction(sys::ScQubitsMimic.AbstractQubit1d, which=1;
                       mode=:abs_sqr) -> Figure

Plot wavefunction in charge basis for a 1D qubit.

`which` can be a single Int or a vector of indices (1-based eigenstate).
"""
function plot_wavefunction(sys::ScQubitsMimic.AbstractQubit1d,
                            which::Union{Int, Vector{Int}}=1;
                            mode::Symbol=:abs_sqr)
    indices = which isa Int ? [which] : which
    _, vecs = ScQubitsMimic.eigensys(sys; evals_count=maximum(indices))

    # For charge-basis qubits, the "position" axis is the charge quantum number
    ncut = sys.ncut
    charges = collect(-ncut:ncut)

    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1], xlabel="Charge n", ylabel="ψ(n)")

    for idx in indices
        psi = vecs[:, idx]
        ydata = if mode == :real
            real.(psi)
        elseif mode == :imag
            imag.(psi)
        elseif mode == :abs
            abs.(psi)
        else  # :abs_sqr
            abs2.(psi)
        end
        lines!(ax, charges, ydata, label="ψ_$(idx-1)")
    end

    if length(indices) > 1
        axislegend(ax)
    end

    return fig
end

# ── Chi vs parameter ────────────────────────────────────────────────────────

"""
    plot_chi_vs_paramvals(sweep::ParameterSweep;
                           subsys_pair=(1,2), param_name=nothing) -> Figure

Plot dispersive shift χ vs parameter from a sweep with stored lookups.
"""
function plot_chi_vs_paramvals(sweep::ParameterSweep;
                                subsys_pair::Tuple{Int,Int}=(1, 2),
                                param_name::Union{Nothing,Symbol}=nothing)
    sweep.lookups === nothing &&
        error("Sweep must be created with store_lookups=true for chi plot")

    chi_arr = ScQubitsMimic.chi_matrix(sweep)
    pname = param_name === nothing ? first(sweep.param_order) : param_name
    pvals = sweep.param_vals[pname]

    i, j = subsys_pair
    chi_vals = chi_arr[:, i, j]

    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1],
              xlabel=string(pname),
              ylabel="χ (GHz)",
              title="Dispersive shift χ_{$i,$j}")
    lines!(ax, pvals, chi_vals .* 1000)  # convert to MHz
    ax.ylabel = "χ (MHz)"

    return fig
end

function plot_transitions(slice::ScQubitsMimic.SweepSlice;
                          subsystems=nothing,
                          initial=nothing,
                          final=nothing,
                          sidebands::Bool=false,
                          photon_number::Int=1,
                          make_positive::Bool=true,
                          coloring::AbstractString="transition",
                          fig_ax=nothing,
                          linewidth::Float64=2.0)
    param_name, param_vals = ScQubitsMimic.slice_param_axis(slice)
    owns_axis = fig_ax === nothing
    if fig_ax === nothing
        fig = Figure(size=(760, 420))
        ax = Axis(fig[1, 1],
                  xlabel=string(param_name),
                  ylabel="Transition energy (GHz)")
    else
        fig, ax = fig_ax
    end

    if lowercase(coloring) == "plain"
        all_diffs = ScQubitsMimic.transition_background(slice;
            initial=initial,
            subsystems=subsystems,
            photon_number=photon_number,
            make_positive=make_positive)
        for k in axes(all_diffs, 2)
            lines!(ax, param_vals, all_diffs[:, k]; linewidth=1.2)
        end
        return fig
    elseif lowercase(coloring) != "transition"
        throw(ArgumentError("coloring must be either \"plain\" or \"transition\""))
    end

    all_diffs = ScQubitsMimic.transition_background(slice;
        initial=initial,
        subsystems=subsystems,
        photon_number=photon_number,
        make_positive=make_positive)
    for k in axes(all_diffs, 2)
        lines!(ax, param_vals, all_diffs[:, k];
               color=:gainsboro, linewidth=0.9)
    end

    spec = ScQubitsMimic.transitions(slice;
        as_specdata=true,
        subsystems=subsystems,
        initial=initial,
        final=final,
        sidebands=sidebands,
        photon_number=photon_number,
        make_positive=make_positive)
    for (idx, label) in enumerate(spec.labels)
        lines!(ax, spec.param_vals, spec.energy_table[:, idx];
               linewidth=linewidth, label=label)
    end
    owns_axis && !isempty(spec.labels) && axislegend(ax; position=:rb)
    return fig
end

function plot_transitions(sweep::ParameterSweep; kwargs...)
    length(sweep.param_order) == 1 || throw(ArgumentError(
        "plot_transitions(::ParameterSweep) only supports one-dimensional sweeps; use `sweep[...]` first"))
    return plot_transitions(ScQubitsMimic.full_slice(sweep); kwargs...)
end

end # module
