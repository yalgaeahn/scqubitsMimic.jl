# ──────────────────────────────────────────────────────────────────────────────
# Kerr oscillator (anharmonic oscillator)
# ──────────────────────────────────────────────────────────────────────────────

"""
    KerrOscillator(; E_osc, K, truncated_dim=10)

Kerr (anharmonic) oscillator with frequency `E_osc` and Kerr nonlinearity `K` (GHz).
Hamiltonian: H = E_osc * â†â - K * â†â(â†â - 1)
"""
@kwdef mutable struct KerrOscillator <: AbstractOscillator
    E_osc::Float64
    K::Float64
    truncated_dim::Int = 10
end

hilbertdim(o::KerrOscillator) = o.truncated_dim

function hamiltonian(o::KerrOscillator)
    n_op = num(o.truncated_dim)
    I_op = qeye(o.truncated_dim)
    return o.E_osc * n_op - o.K * n_op * (n_op - I_op)
end

function _kerr_diagonal_energies(o::KerrOscillator)
    return Float64[(o.E_osc + o.K) * k - o.K * k^2
                   for k in 0:(o.truncated_dim - 1)]
end

function _kerr_lowest_indices(o::KerrOscillator, evals_count::Int)
    n = clamp(evals_count, 0, o.truncated_dim)
    n == 0 && return Int[]
    return sortperm(_kerr_diagonal_energies(o); alg=MergeSort)[1:n]
end

function eigenvals(o::KerrOscillator; evals_count::Int=o.truncated_dim)
    energies = _kerr_diagonal_energies(o)
    return energies[_kerr_lowest_indices(o, evals_count)]
end

function eigensys(o::KerrOscillator; evals_count::Int=o.truncated_dim)
    energies = _kerr_diagonal_energies(o)
    indices = _kerr_lowest_indices(o, evals_count)
    vecs = Matrix{ComplexF64}(I, o.truncated_dim, o.truncated_dim)[:, indices]
    return energies[indices], vecs
end

annihilation_operator(o::KerrOscillator) = destroy(o.truncated_dim)
creation_operator(o::KerrOscillator) = create(o.truncated_dim)
number_operator(o::KerrOscillator) = num(o.truncated_dim)
