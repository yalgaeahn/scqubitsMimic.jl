# API Reference

This section is the structured public API reference for `ScQubitsMimic.jl`.
Pages are organized by subsystem rather than by source file so the exported
surface can be browsed the same way users approach the package in practice.

| Page | Focus |
| --- | --- |
| [`core.md`](./core.md) | shared abstract types, units, grids, spectra, and common numerical interface |
| [`qubits.md`](./qubits.md) | built-in qubit models and their public operators |
| [`oscillators.md`](./oscillators.md) | oscillator models and ladder/number operators |
| [`circuit.md`](./circuit.md) | graph parsing, topology helpers, `Circuit`, symbolic accessors, and circuit mutation |
| [`symbolic-circuit.md`](./symbolic-circuit.md) | `SymbolicCircuit`, mode decomposition, and symbolic metadata |
| [`circuit-operators.md`](./circuit-operators.md) | low-level operator constructors used by circuit quantization |
| [`circuit-hierarchy.md`](./circuit-hierarchy.md) | hierarchical diagonalization, subsystem decomposition, and interaction accessors |
| [`hilbert-space.md`](./hilbert-space.md) | composite-system construction and the dressed-space container types |
| [`sweeps-and-lookup.md`](./sweeps-and-lookup.md) | parameter sweeps, lookup generation, and lookup-driven state accessors |
| [`analysis-and-transitions.md`](./analysis-and-transitions.md) | transitions, `χ`, self-Kerr, and Lamb-shift analysis |
| [`plotting.md`](./plotting.md) | Makie-backed plotting entry points provided by the package extension |

The guide notebooks remain available from [`../guides/index.md`](../guides/index.md).
