# Symbolic Circuit

`SymbolicCircuit` is the symbolic-analysis layer between circuit topology and
the numerical `Circuit` object. It stores loop structure, node variables,
symbolic Hamiltonian pieces, and the metadata needed for mode decomposition.

```@docs
SymbolicCircuit
build_symbolic_circuit
compute_variable_transformation
VarCategories
```

## Notes

- `SymbolicCircuit` stores the node-basis symbolic Hamiltonian, not the final
  transformed-mode Hamiltonian.
- Symbolic flux labels such as `Φ1` are placeholders. Numerical values are
  supplied later when constructing or mutating a [`Circuit`](./circuit.md).
- The direct builder is useful for symbolic inspection before creating a full
  numerical circuit model.
