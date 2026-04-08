# ──────────────────────────────────────────────────────────────────────────────
# Harmonic oscillator
# ──────────────────────────────────────────────────────────────────────────────

"""
    Oscillator(; E_osc, truncated_dim=10)

Quantum harmonic oscillator with frequency `E_osc` (in GHz).
Hamiltonian: H = E_osc * (â†â + 1/2)
"""
@kwdef mutable struct Oscillator <: AbstractOscillator
    E_osc::Float64
    truncated_dim::Int = 10
end

hilbertdim(o::Oscillator) = o.truncated_dim

function hamiltonian(o::Oscillator)
    return o.E_osc * (num(o.truncated_dim) + 0.5 * qeye(o.truncated_dim))
end

annihilation_operator(o::Oscillator) = destroy(o.truncated_dim)
creation_operator(o::Oscillator) = create(o.truncated_dim)
number_operator(o::Oscillator) = num(o.truncated_dim)
