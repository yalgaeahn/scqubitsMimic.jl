# ──────────────────────────────────────────────────────────────────────────────
# Grid types for discretized phase-basis representations
# ──────────────────────────────────────────────────────────────────────────────

"""
    Grid1d(min_val, max_val, npoints)

One-dimensional uniform grid for discretizing phase-space variables.
Used by extended-variable operators in the discretized (non-harmonic) basis.
"""
struct Grid1d
    min_val::Float64
    max_val::Float64
    npoints::Int

    function Grid1d(min_val::Real, max_val::Real, npoints::Int)
        min_val < max_val || throw(ArgumentError("min_val must be < max_val"))
        npoints >= 2 || throw(ArgumentError("npoints must be >= 2"))
        new(Float64(min_val), Float64(max_val), npoints)
    end
end

"""Return the grid spacing."""
grid_spacing(g::Grid1d) = (g.max_val - g.min_val) / (g.npoints - 1)

"""Return the grid points as a vector."""
grid_points(g::Grid1d) = range(g.min_val, g.max_val; length=g.npoints)
