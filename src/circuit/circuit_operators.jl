# ──────────────────────────────────────────────────────────────────────────────
# Numerical operator construction for circuit quantization
#
# Three basis types are supported:
# 1. Charge basis (periodic variables): |n⟩, n ∈ {-ncut, ..., ncut}
# 2. Harmonic oscillator basis (extended variables): Fock states |k⟩
# 3. Discretized grid basis (extended variables): φ grid points
# ──────────────────────────────────────────────────────────────────────────────

# ── Periodic variable operators (charge basis) ───────────────────────────────

"""
    n_operator_periodic(ncut::Int) -> QuantumObject

Number (charge) operator in the charge basis: n̂|n⟩ = n|n⟩.
Basis: |n⟩ for n ∈ {-ncut, -ncut+1, ..., ncut}. Dimension = 2*ncut + 1.
"""
function n_operator_periodic(ncut::Int)
    dim = 2 * ncut + 1
    diag_vals = collect(-ncut:ncut)
    return QuantumObject(spdiagm(0 => ComplexF64.(diag_vals)))
end

"""
    exp_i_theta_operator(ncut::Int) -> QuantumObject

Exponential operator e^{iθ} in the charge basis.
Acts as a shift: e^{iθ}|n⟩ = |n+1⟩.
"""
function exp_i_theta_operator(ncut::Int)
    dim = 2 * ncut + 1
    return QuantumObject(spdiagm(1 => ones(ComplexF64, dim - 1)))
end

"""
    cos_theta_operator(ncut::Int) -> QuantumObject

cos(θ) operator: (e^{iθ} + e^{-iθ}) / 2.
"""
function cos_theta_operator(ncut::Int)
    e_pos = exp_i_theta_operator(ncut)
    return (e_pos + e_pos') / 2
end

"""
    sin_theta_operator(ncut::Int) -> QuantumObject

sin(θ) operator: (e^{iθ} - e^{-iθ}) / (2i).
"""
function sin_theta_operator(ncut::Int)
    e_pos = exp_i_theta_operator(ncut)
    return (e_pos - e_pos') / (2im)
end

# ── Extended variable operators (harmonic oscillator / Fock basis) ────────────

"""
    phi_operator_ho(cutoff::Int, osc_length::Float64) -> QuantumObject

Phase (position) operator in harmonic oscillator basis:
φ̂ = osc_length * (â + â†) / √2
"""
function phi_operator_ho(cutoff::Int, osc_length::Float64)
    a = destroy(cutoff)
    return osc_length * (a + a') / sqrt(2)
end

"""
    n_operator_ho(cutoff::Int, osc_length::Float64) -> QuantumObject

Charge (momentum) operator in harmonic oscillator basis:
n̂ = i(â† - â) / (√2 * osc_length)
"""
function n_operator_ho(cutoff::Int, osc_length::Float64)
    a = destroy(cutoff)
    return 1im * (a' - a) / (sqrt(2) * osc_length)
end

"""
    cos_phi_operator_ho(cutoff::Int, osc_length::Float64) -> QuantumObject

cos(φ̂) in harmonic oscillator basis, computed via matrix exponentiation.
"""
function cos_phi_operator_ho(cutoff::Int, osc_length::Float64)
    phi = phi_operator_ho(cutoff, osc_length)
    exp_pos = QuantumObject(exp(Matrix(1im * phi.data)))
    return (exp_pos + exp_pos') / 2
end

"""
    sin_phi_operator_ho(cutoff::Int, osc_length::Float64) -> QuantumObject

sin(φ̂) in harmonic oscillator basis, computed via matrix exponentiation.
"""
function sin_phi_operator_ho(cutoff::Int, osc_length::Float64)
    phi = phi_operator_ho(cutoff, osc_length)
    exp_pos = QuantumObject(exp(Matrix(1im * phi.data)))
    return (exp_pos - exp_pos') / (2im)
end

# ── Extended variable operators (discretized grid basis) ─────────────────────

"""
    phi_operator_grid(grid::Grid1d) -> QuantumObject

Phase operator on a discretized grid: diagonal matrix of grid point values.
"""
function phi_operator_grid(grid::Grid1d)
    pts = collect(grid_points(grid))
    return QuantumObject(spdiagm(0 => ComplexF64.(pts)))
end

"""
    d_dphi_operator_grid(grid::Grid1d) -> QuantumObject

First derivative operator on a discretized grid using central finite differences.
"""
function d_dphi_operator_grid(grid::Grid1d)
    n = grid.npoints
    dx = grid_spacing(grid)
    # Central difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    upper = fill(ComplexF64(1.0 / (2dx)), n - 1)
    lower = fill(ComplexF64(-1.0 / (2dx)), n - 1)
    return QuantumObject(spdiagm(-1 => lower, 1 => upper))
end

"""
    d2_dphi2_operator_grid(grid::Grid1d) -> QuantumObject

Second derivative operator on a discretized grid.
"""
function d2_dphi2_operator_grid(grid::Grid1d)
    n = grid.npoints
    dx = grid_spacing(grid)
    # f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
    main = fill(ComplexF64(-2.0 / dx^2), n)
    off = fill(ComplexF64(1.0 / dx^2), n - 1)
    return QuantumObject(spdiagm(-1 => off, 0 => main, 1 => off))
end

"""
    cos_phi_operator_grid(grid::Grid1d) -> QuantumObject

cos(φ) operator on a discretized grid (diagonal).
"""
function cos_phi_operator_grid(grid::Grid1d)
    pts = collect(grid_points(grid))
    return QuantumObject(spdiagm(0 => ComplexF64.(cos.(pts))))
end

"""
    sin_phi_operator_grid(grid::Grid1d) -> QuantumObject

sin(φ) operator on a discretized grid (diagonal).
"""
function sin_phi_operator_grid(grid::Grid1d)
    pts = collect(grid_points(grid))
    return QuantumObject(spdiagm(0 => ComplexF64.(sin.(pts))))
end
