# ──────────────────────────────────────────────────────────────────────────────
# Harmonic oscillator
# ──────────────────────────────────────────────────────────────────────────────

"""
    Oscillator(; E_osc, truncated_dim=10)

Quantum harmonic oscillator with frequency `E_osc` (in GHz).
Hamiltonian: H = E_osc * â†â
"""
@kwdef mutable struct Oscillator <: AbstractOscillator
    E_osc::Float64
    truncated_dim::Int = 10
end

hilbertdim(o::Oscillator) = o.truncated_dim

function hamiltonian(o::Oscillator)
    return o.E_osc * num(o.truncated_dim)
end

function eigenvals(o::Oscillator; evals_count::Int=o.truncated_dim)
    n = min(evals_count, o.truncated_dim)
    return Float64[o.E_osc * k for k in 0:(n - 1)]
end

function eigensys(o::Oscillator; evals_count::Int=o.truncated_dim)
    vals = eigenvals(o; evals_count=evals_count)
    n = length(vals)
    vecs = Matrix{ComplexF64}(I, o.truncated_dim, n)
    return vals, vecs
end

"""Return the annihilation operator `â` for an [`Oscillator`](@ref)."""
annihilation_operator(o::Oscillator) = destroy(o.truncated_dim)

"""Return the creation operator `â†` for an [`Oscillator`](@ref)."""
creation_operator(o::Oscillator) = create(o.truncated_dim)

"""Return the number operator `â†â` for an [`Oscillator`](@ref)."""
number_operator(o::Oscillator) = num(o.truncated_dim)
