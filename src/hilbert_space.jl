# ──────────────────────────────────────────────────────────────────────────────
# HilbertSpace: composite quantum system
# ──────────────────────────────────────────────────────────────────────────────

"""
    HilbertSpace(subsystems)

Composite quantum system consisting of multiple coupled subsystems.

# Example
```julia
transmon = Transmon(EJ=30.0, EC=1.2)
osc = Oscillator(E_osc=6.0)
hs = HilbertSpace([transmon, osc])
add_interaction!(hs, 0.1, [transmon, osc], [cos_phi_operator, annihilation_operator])
H = hamiltonian(hs)
```
"""
mutable struct HilbertSpace <: AbstractQuantumSystem
    subsystems::Vector{AbstractQuantumSystem}
    interactions::Vector{InteractionTerm}
    extra_H_terms::Vector{QuantumObject}   # pre-built operators added directly to H
    lookup::Union{Nothing, SpectrumLookup}
    ignore_low_overlap::Bool
end

HilbertSpace(subsystems::Vector{<:AbstractQuantumSystem}; ignore_low_overlap::Bool=false) =
    HilbertSpace(collect(AbstractQuantumSystem, subsystems), InteractionTerm[], QuantumObject[],
                 nothing, ignore_low_overlap)

function hilbertdim(hs::HilbertSpace)
    return prod(hilbertdim(s) for s in hs.subsystems)
end

"""Add an interaction term to the Hilbert space."""
function add_interaction!(hs::HilbertSpace, g::Float64,
                          subsys_list::Vector{<:AbstractQuantumSystem},
                          op_list::Vector{Function})
    push!(hs.interactions, InteractionTerm(g, collect(AbstractQuantumSystem, subsys_list), op_list))
end

"""Add a pre-built operator directly to the Hamiltonian."""
function add_operator!(hs::HilbertSpace, op::QuantumObject)
    push!(hs.extra_H_terms, op)
end

function hamiltonian(hs::HilbertSpace)
    dims = [hilbertdim(s) for s in hs.subsystems]
    n = length(hs.subsystems)

    # Bare Hamiltonians
    H = sum(identity_wrap(hamiltonian(hs.subsystems[i]), i, dims) for i in 1:n)

    # Interaction terms
    for term in hs.interactions
        ops = QuantumObject[]
        for (s, op_func) in zip(term.subsys_list, term.op_list)
            idx = findfirst(==(s), hs.subsystems)
            idx === nothing && error("Subsystem not found in HilbertSpace")
            push!(ops, op_func(s))
        end

        # Build tensor product of interaction operators
        interaction_op = _build_interaction_operator(hs, term)
        H = H + term.g_strength * interaction_op
    end

    # Pre-built operator terms (e.g., cross-group Josephson)
    for op in hs.extra_H_terms
        H = H + op
    end

    return H
end

function _build_interaction_operator(hs::HilbertSpace, term::InteractionTerm)
    dims = [hilbertdim(s) for s in hs.subsystems]
    n = length(hs.subsystems)

    # Start with identity operators for all subsystems
    # Use Any[] to avoid QuantumObject storage type mismatch (Diagonal vs Sparse)
    op_per_subsys = Vector{Any}([qeye(dims[i]) for i in 1:n])

    for (s, op_func) in zip(term.subsys_list, term.op_list)
        idx = findfirst(==(s), hs.subsystems)
        op_per_subsys[idx] = op_func(s)
    end

    return reduce(kron, op_per_subsys)
end

"""
    diag!(hs::HilbertSpace; evals_count=10)

Diagonalize the full Hamiltonian. Returns `(eigenvalues, eigenvectors)`.
"""
function diag!(hs::HilbertSpace; evals_count::Int=10)
    return eigensys(hs; evals_count=evals_count)
end
