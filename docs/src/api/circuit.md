# Circuit

This page covers the public circuit entry workflow:

`description string -> CircuitGraph -> SymbolicCircuit -> Circuit`.

Julia keeps the scqubits-style concepts, but the main constructor is
`Circuit(description::String; ...)`, external flux values are stored in radians,
and subsystem indexing is 1-based.

## Graph And Topology

```@docs
BranchType
C_branch
L_branch
JJ_branch
CJ_branch
Branch
CircuitGraph
parse_circuit
find_spanning_tree
find_closure_branches
find_fundamental_loops
find_superconducting_loops
```

## Main Circuit Type

```@docs
Circuit
```

## Symbolic Accessors

```@docs
sym_hamiltonian
sym_hamiltonian_node
sym_lagrangian
variable_transformation
external_fluxes
sym_external_fluxes
offset_charges
offset_charge_transformation
```

## Circuit Mutation And Cache Control

`Circuit` also implements the generic [`set_param!`](./core.md) and
[`get_param`](./core.md) interface documented on the core page.

```@docs
set_external_flux!
set_offset_charge!
list_branch_params
invalidate_cache!
```

## See Also

- [`./symbolic-circuit.md`](./symbolic-circuit.md)
- [`./circuit-hierarchy.md`](./circuit-hierarchy.md)
- [`../guides/circuit-quantization.md`](../guides/circuit-quantization.md)
