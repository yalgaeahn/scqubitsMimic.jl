# ──────────────────────────────────────────────────────────────────────────────
# TunableTransmon: flux-tunable transmon via SQUID loop
#
# H = 4 EC (n̂ - ng)² - EJ(Φext) cos(θ̂)
# EJ(Φext) = EJmax |cos(π Φext / Φ0)| √(1 + d² tan²(π Φext / Φ0))
#
# where d = (EJ2 - EJ1) / (EJ1 + EJ2) is the junction asymmetry,
# EJmax = EJ1 + EJ2, and Φext is in units of Φ0.
# ──────────────────────────────────────────────────────────────────────────────

"""
    TunableTransmon <: AbstractQubit1d

Flux-tunable transmon qubit with SQUID loop.

# Parameters
- `EJmax` — maximum Josephson energy (EJ1 + EJ2), in GHz
- `EC`    — charging energy, in GHz
- `d`     — junction asymmetry ratio (EJ2 - EJ1)/(EJ1 + EJ2); 0 = symmetric
- `flux`  — external flux in units of Φ0 (0 to 1)
- `ng`    — offset charge (in units of 2e)
- `ncut`  — charge basis truncation
- `truncated_dim` — number of eigenvalues to return by default
"""
@kwdef mutable struct TunableTransmon <: AbstractQubit1d
    EJmax::Float64
    EC::Float64
    d::Float64 = 0.0
    flux::Float64 = 0.0
    ng::Float64 = 0.0
    ncut::Int = 30
    truncated_dim::Int = 6
end

"""
    ej_effective(t::TunableTransmon)

Compute effective Josephson energy: EJmax * cos(π flux) * √(1 + d² tan²(π flux))
"""
function ej_effective(t::TunableTransmon)
    arg = π * t.flux
    return t.EJmax * abs(cos(arg)) * sqrt(1.0 + t.d^2 * tan(arg)^2)
end

function hilbertdim(t::TunableTransmon)
    return 2 * t.ncut + 1
end

function hamiltonian(t::TunableTransmon)
    dim = hilbertdim(t)
    EJ_eff = ej_effective(t)

    # Charge basis: n = -ncut, ..., ncut
    n_diag = collect(-t.ncut:t.ncut)

    # Kinetic: 4 EC (n - ng)²
    diag_vals = ComplexF64.(4.0 * t.EC * (n_diag .- t.ng) .^ 2)
    H = spdiagm(0 => diag_vals)

    # Potential: -EJ/2 (|n><n+1| + |n+1><n|) = -EJ * cos(θ)
    off_diag = fill(ComplexF64(-EJ_eff / 2.0), dim - 1)
    H += spdiagm(1 => off_diag, -1 => off_diag)

    return QuantumObject(H)
end

function eigenvals(t::TunableTransmon; evals_count::Int=t.truncated_dim)
    H = hamiltonian(t)
    vals = eigenenergies(H)
    n = min(evals_count, length(vals))
    return real.(vals[1:n])
end

function eigensys(t::TunableTransmon; evals_count::Int=t.truncated_dim)
    H = hamiltonian(t)
    result = eigenstates(H)
    n = min(evals_count, length(result.values))
    vals = real.(result.values[1:n])
    vecs = result.vectors[:, 1:n]
    return vals, vecs
end

# ── Operators ───────────────────────────────────────────────────────────────

n_operator(t::TunableTransmon) = n_operator_periodic(t.ncut)
exp_i_phi_operator(t::TunableTransmon) = exp_i_theta_operator(t.ncut)
cos_phi_operator(t::TunableTransmon) = cos_theta_operator(t.ncut)
sin_phi_operator(t::TunableTransmon) = sin_theta_operator(t.ncut)
