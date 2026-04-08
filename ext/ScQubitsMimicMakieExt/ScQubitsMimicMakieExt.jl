module ScQubitsMimicMakieExt

using ScQubitsMimic
using CairoMakie

import ScQubitsMimic: plot_evals_vs_paramvals, plot_matrixelements,
                      plot_wavefunction, plot_chi_vs_paramvals

# ── Eigenvalue spectrum vs parameter ────────────────────────────────────────

"""
    plot_evals_vs_paramvals(sweep::ParameterSweep; kwargs...) -> Figure

Plot eigenvalues vs parameter values from a single-system sweep.

# Keyword arguments
- `subtract_ground::Bool=true` — subtract ground state energy
- `evals_count::Int=nothing` — number of levels to plot (default: all)
- `fig_ax=nothing` — optional `(Figure, Axis)` to plot into
"""
function plot_evals_vs_paramvals(sweep::ParameterSweep;
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
    plot_evals_vs_paramvals(sweep::HilbertSpaceSweep; kwargs...) -> Figure

Plot dressed eigenvalues vs parameter from a HilbertSpace sweep.
"""
function plot_evals_vs_paramvals(sweep::HilbertSpaceSweep;
                                  param_name::Union{Nothing,Symbol}=nothing,
                                  subtract_ground::Bool=true,
                                  evals_count::Union{Nothing,Int}=nothing,
                                  fig_ax=nothing)
    pname = param_name === nothing ? first(keys(sweep.param_vals)) : param_name
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
    plot_chi_vs_paramvals(sweep::HilbertSpaceSweep;
                           subsys_pair=(1,2), param_name=nothing) -> Figure

Plot dispersive shift χ vs parameter from a sweep with stored lookups.
"""
function plot_chi_vs_paramvals(sweep::HilbertSpaceSweep;
                                subsys_pair::Tuple{Int,Int}=(1, 2),
                                param_name::Union{Nothing,Symbol}=nothing)
    sweep.lookups === nothing &&
        error("Sweep must be created with store_lookups=true for chi plot")

    chi_arr = ScQubitsMimic.chi_matrix(sweep)
    pname = param_name === nothing ? first(keys(sweep.param_vals)) : param_name
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

end # module
