# ──────────────────────────────────────────────────────────────────────────────
# Transmon qubit (hardcoded, for validation against Circuit-derived version)
#
# Hamiltonian in charge basis:
#   H = 4 EC (n̂ - ng)² - EJ cos(φ̂)
# where n̂ is the Cooper pair number operator, φ̂ is the phase operator.
# ──────────────────────────────────────────────────────────────────────────────

"""
    Transmon(; EJ, EC, ng=0.0, ncut=30, truncated_dim=6)

Transmon qubit defined by Josephson energy `EJ` and charging energy `EC` (in GHz).

# Parameters
- `EJ::Float64` — Josephson energy
- `EC::Float64` — charging energy
- `ng::Float64` — offset charge (default 0)
- `ncut::Int` — charge basis cutoff (basis: |n⟩, n ∈ {-ncut,...,ncut})
- `truncated_dim::Int` — number of eigenvalues to compute by default
"""
@kwdef mutable struct Transmon <: AbstractQubit1d
    EJ::Float64
    EC::Float64
    ng::Float64 = 0.0
    ncut::Int = 30
    truncated_dim::Int = 6
end

hilbertdim(t::Transmon) = 2 * t.ncut + 1

function hamiltonian(t::Transmon)
    dim = hilbertdim(t)
    ncut = t.ncut

    # Charge operator: n̂ - ng
    n_diag = ComplexF64.(collect(-ncut:ncut) .- t.ng)
    H_charge = spdiagm(0 => 4.0 * t.EC * n_diag .^ 2)

    # Josephson operator: -EJ/2 * (e^{iφ} + e^{-iφ})
    ones_off = fill(ComplexF64(-t.EJ / 2.0), dim - 1)
    H_josephson = spdiagm(1 => ones_off, -1 => ones_off)

    return QuantumObject(H_charge + H_josephson)
end

function eigenvals(t::Transmon; evals_count::Int=t.truncated_dim)
    invoke(eigenvals, Tuple{AbstractQuantumSystem}, t; evals_count=evals_count)
end

function eigensys(t::Transmon; evals_count::Int=t.truncated_dim)
    invoke(eigensys, Tuple{AbstractQuantumSystem}, t; evals_count=evals_count)
end

# ── Transmon-specific operators ──────────────────────────────────────────────

"""Number operator n̂ for the Transmon."""
n_operator(t::Transmon) = n_operator_periodic(t.ncut)

"""Phase operator e^{iφ} for the Transmon."""
exp_i_phi_operator(t::Transmon) = exp_i_theta_operator(t.ncut)

"""cos(φ) operator for the Transmon."""
cos_phi_operator(t::Transmon) = cos_theta_operator(t.ncut)

"""sin(φ) operator for the Transmon."""
sin_phi_operator(t::Transmon) = sin_theta_operator(t.ncut)

"""
    potential(t::Transmon, phi)

Transmon potential energy: -EJ * cos(phi).
"""
potential(t::Transmon, phi::Real) = -t.EJ * cos(phi)
