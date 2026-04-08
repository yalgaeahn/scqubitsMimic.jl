# ──────────────────────────────────────────────────────────────────────────────
# GenericQubit: simple two-level system
# ──────────────────────────────────────────────────────────────────────────────

"""
    GenericQubit(E::Float64)

A generic two-level system with energy splitting `E` (in GHz).
Hamiltonian: H = E/2 * σ_z
"""
@kwdef mutable struct GenericQubit <: AbstractQubit
    E::Float64
end

hilbertdim(q::GenericQubit) = 2

function hamiltonian(q::GenericQubit)
    return q.E / 2 * sigmaz()
end
