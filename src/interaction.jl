# ──────────────────────────────────────────────────────────────────────────────
# InteractionTerm: coupling between subsystems in a HilbertSpace
# ──────────────────────────────────────────────────────────────────────────────

"""
    InteractionTerm

Describes an interaction between subsystems in a composite Hilbert space.

The interaction Hamiltonian is:
  H_int = g_strength * op_list[1](subsys_list[1]) ⊗ op_list[2](subsys_list[2]) ⊗ ...

# Fields
- `g_strength::Float64` — coupling strength (GHz)
- `subsys_list::Vector{AbstractQuantumSystem}` — participating subsystems
- `op_list::Vector{Function}` — operator-generating functions for each subsystem
"""
struct InteractionTerm
    g_strength::Float64
    subsys_list::Vector{AbstractQuantumSystem}
    op_list::Vector{Function}
end
