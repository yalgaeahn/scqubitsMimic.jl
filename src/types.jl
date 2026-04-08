# ──────────────────────────────────────────────────────────────────────────────
# Abstract type hierarchy for quantum systems
#
# Maps Python scqubits' class hierarchy to Julia's abstract type system:
#   QuantumSystem → QubitBaseClass → QubitBaseClass1d
# becomes:
#   AbstractQuantumSystem → AbstractQubit → AbstractQubit1d / AbstractQubitNd
#   AbstractQuantumSystem → AbstractOscillator
# ──────────────────────────────────────────────────────────────────────────────

"""
    AbstractQuantumSystem

Top-level abstract type for all quantum systems (qubits, oscillators, circuits).
Every concrete quantum system must implement `hamiltonian(sys)` and `hilbertdim(sys)`.
"""
abstract type AbstractQuantumSystem end

"""
    AbstractQubit <: AbstractQuantumSystem

Abstract type for qubit-like systems requiring numerical diagonalization.
"""
abstract type AbstractQubit <: AbstractQuantumSystem end

"""
    AbstractQubit1d <: AbstractQubit

Abstract type for qubits with a single degree of freedom and a 1D potential
(e.g., Transmon, Fluxonium). Supports `potential(sys, phi)` and `wavefunction(sys)`.
"""
abstract type AbstractQubit1d <: AbstractQubit end

"""
    AbstractQubitNd <: AbstractQubit

Abstract type for qubits with multiple degrees of freedom
(e.g., ZeroPi, FluxQubit). Requires multi-dimensional basis handling.
"""
abstract type AbstractQubitNd <: AbstractQubit end

"""
    AbstractOscillator <: AbstractQuantumSystem

Abstract type for oscillator-like systems with analytical eigenvalues
(e.g., harmonic oscillator, Kerr oscillator).
"""
abstract type AbstractOscillator <: AbstractQuantumSystem end
