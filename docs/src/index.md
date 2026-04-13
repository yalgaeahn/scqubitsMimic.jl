# ScQubitsMimic.jl

`ScQubitsMimic.jl` mirrors the implemented portion of the Python
[`scqubits`](https://scqubits.readthedocs.io/en/latest/) workflow in Julia.
This documentation site separates runnable guides from structured API
reference pages so the package can be explored in the same way that scqubits
users browse tutorials plus class/function reference material.

## What This Site Contains

- `Guides`: short landing pages for the executed notebooks in
  `docs/src/notebooks`
- `API Reference`: subsystem-oriented reference pages built with `Documenter.jl`
- `API Audit`: a source-of-truth checklist mapping every exported symbol to its
  primary reference page

## Implemented Areas

- core spectral utilities and parameter sweeps for single systems
- `GenericQubit`, `Transmon`, and `TunableTransmon`
- oscillator models and shared operator helpers
- circuit parsing, topology analysis, symbolic quantization, and hierarchy APIs
- composite `HilbertSpace` workflows, bare/dressed lookup tables, and dispersive summaries
- Makie-backed plotting entry points exposed through the package extension

## Not Yet Ported From Python scqubits

- broader built-in qubit families such as `Fluxonium`, `FluxQubit`, and `ZeroPi`
- the Python GUI / explorer tooling
- the full plotting, coherence, and noise-analysis surface
- the full container-style scqubits interface around parameter sweeps

## Start Here

- If you want runnable workflows, start with [`guides/index.md`](./guides/index.md).
- If you want type/function reference pages, start with [`api/index.md`](./api/index.md).
- If you want to verify documentation coverage against the public API surface, use
  [`api-coverage.md`](./api-coverage.md).
