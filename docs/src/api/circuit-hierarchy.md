# Circuit Hierarchy

These APIs provide the Julia-side equivalent of scqubits hierarchical circuit
workflows. Subsystem indices are always 1-based.

```@docs
configure!
sym_interaction
SubCircuit
HierarchyLeaf
HierarchyGroup
HierarchyNode
hierarchical_diag
truncation_template
```

## Notes

- `configure!` stores hierarchy state on an existing `Circuit`, while
  `hierarchical_diag` performs the workflow as a one-shot helper.
- `sym_hamiltonian(circ; subsystem_index=i)` is documented on the main
  [`Circuit`](./circuit.md) page because the same entry point is shared between
  whole-circuit and subsystem access.
- Hierarchy shapes and truncation templates must match exactly.
