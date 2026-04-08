# ScQubitsMimic.jl — Implementation Plan

## Context

Port Python's `scqubits` to pure Julia, preserving its design philosophy (modular, physicist-friendly API) while leveraging Julia's strengths (multiple dispatch, JIT, type system). Key dependency mapping: QuTiP → QuantumToolbox.jl, SymPy → Symbolics.jl.

**Priority shift**: Circuit quantization pipeline (`graph → symbolic circuit → mode decomposition → quantization`) is the primary goal. Individual hardcoded qubit classes (Fluxonium, FluxQubit, ZeroPi, etc.) are deferred. The TunableTransmon serves as the first end-to-end validation target.

---

## Phase 1: Package Scaffold & Core Infrastructure

### 1.1 Package Setup
- `Project.toml` — name, uuid, deps (QuantumToolbox, Symbolics, LinearAlgebra, SparseArrays, Graphs)
- `src/ScQubitsMimic.jl` — main module with exports
- `test/runtests.jl` — test entry point

### 1.2 Abstract Type Hierarchy (`src/types.jl`)
```julia
abstract type AbstractQuantumSystem end
abstract type AbstractQubit <: AbstractQuantumSystem end
abstract type AbstractQubit1d <: AbstractQubit end
abstract type AbstractQubitNd <: AbstractQubit end
abstract type AbstractOscillator <: AbstractQuantumSystem end
```

### 1.3 Physical Constants & Units
- `src/constants.jl` — e, h, hbar, Phi_0, k_B, etc.
- `src/units.jl` — GHz ↔ eV ↔ K conversions

### 1.4 Grid & Spectrum Data
- `src/grid.jl` — `Grid1d` struct (min_val, max_val, npoints) for phase-basis discretization
- `src/spectrum_data.jl` — `SpectrumData` result container (eigenvalues, eigenvectors, parameter values)

### 1.5 Base Method Interface (`src/qubit_base.jl`)
Default implementations dispatching on `hamiltonian(sys)`:
```julia
hamiltonian(sys::AbstractQuantumSystem)   # each concrete type implements this
eigenvals(sys; evals_count=6)
eigensys(sys; evals_count=6)
matrixelement(sys, op, i, j)
matrixelement_table(sys, op; evals_count=6)
get_spectrum_vs_paramvals(sys, param_name, param_vals; evals_count)
```

### 1.6 Operator Helpers (`src/operators.jl`)
- `identity_wrap(op, subsys_index, dims)` — embed subsystem operator into full tensor-product space
- Bridge functions to QuantumToolbox.jl's `QuantumObject`

---

## Phase 2: Circuit Graph Representation

### 2.1 Graph Data Structures (`src/circuit/circuit_graph.jl`)

```julia
@enum BranchType C_branch L_branch JJ_branch CJ_branch

struct CircuitNode
    index::Int
    is_ground::Bool
end

struct Branch
    branch_type::BranchType
    node_i::Int        # from node
    node_j::Int        # to node
    parameters::Dict{Symbol, Any}  # :EJ, :EC, :EL, etc. (can be symbolic or numeric)
end

struct Coupler
    coupler_type::Symbol   # :ML (mutual inductance)
    branch1::Int
    branch2::Int
    parameters::Dict{Symbol, Float64}
end

struct CircuitGraph
    nodes::Vector{CircuitNode}
    branches::Vector{Branch}
    couplers::Vector{Coupler}
end
```

### 2.2 Circuit Input Parsing (`src/circuit/circuit_input.jl`)
- Parse YAML-style string description into `CircuitGraph`
- Support format compatible with scqubits:
  ```
  nodes: 3
  branches:
    - [JJ, 1, 2, EJ=10.0, EC=0.3]
    - [JJ, 1, 2, EJ=10.0, EC=0.3]
    - [C, 1, 2, EC=0.5]
    - [L, 2, 3, EL=0.8]
  ```

### 2.3 Spanning Tree & Loop Analysis (`src/circuit/circuit_topology.jl`)
- Use `Graphs.jl` for graph operations
- `find_spanning_tree(graph)` — automatic spanning tree selection (prefer capacitive branches)
- `find_closure_branches(graph, tree)` — identify cotree branches (independent loops)
- `find_loops(graph)` — enumerate fundamental loops for external flux allocation
- `assign_external_fluxes(graph, loops)` — allocate Phi_ext variables to loops

---

## Phase 3: Symbolic Circuit Analysis

### 3.1 Capacitance & Inductance Matrices (`src/circuit/symbolic_circuit.jl`)

Using Symbolics.jl for symbolic parameters:

```julia
struct SymbolicCircuit
    graph::CircuitGraph
    spanning_tree::Vector{Int}         # branch indices in tree
    closure_branches::Vector{Int}      # cotree branch indices

    # Symbolic matrices (Symbolics.jl Num types)
    capacitance_matrix::Matrix{Num}    # C matrix (node basis)
    inv_inductance_matrix::Matrix{Num} # L^{-1} matrix (node basis)

    # Variable transformation
    transformation_matrix::Matrix      # T: node vars → mode vars
    var_categories::Dict{Symbol, Vector{Int}}  # :periodic, :extended, :free, :frozen

    # Symbolic expressions
    lagrangian_symbolic::Num
    hamiltonian_symbolic::Num

    # Parameters
    symbolic_params::Dict{Symbol, Any}     # EJ, EC, EL values (symbolic or numeric)
    external_fluxes::Vector{Num}           # Phi_ext symbolic variables
    offset_charges::Vector{Num}            # n_g symbolic variables
end
```

**Key functions:**
```julia
build_capacitance_matrix(graph::CircuitGraph)::Matrix{Num}
build_inv_inductance_matrix(graph::CircuitGraph)::Matrix{Num}
generate_symbolic_lagrangian(sc::SymbolicCircuit)::Num
generate_symbolic_hamiltonian(sc::SymbolicCircuit)::Num  # Legendre transform
```

### 3.2 Lagrangian Construction
- **Kinetic energy**: T = (1/2) dot_phi^T * C * dot_phi
- **Potential energy**: V = (1/2) phi^T * L^{-1} * phi + Sum_j(-EJ_j * cos(phi_j - phi_ext_j))
- **Lagrangian**: L = T - V
- Ground node (index 0) eliminated, reducing N nodes → N-1 variables

### 3.3 Hamiltonian via Legendre Transform
- Canonical momenta: Q_i = dL/d(dot_phi_i) = C * dot_phi
- H = Q^T * dot_phi - L = (1/2) Q^T * C^{-1} * Q + V(phi)
- EC matrix: EC[i,j] = (1/2) * (C^{-1})[i,j] (charging energies)

---

## Phase 4: Mode Decomposition

### 4.1 Variable Transformation (`src/circuit/mode_decomposition.jl`)

Construct transformation matrix T that maps node variables to normal mode variables:

```julia
function compute_transformation_matrix(sc::SymbolicCircuit)
    # 1. Identify periodic modes (from L-branch loops)
    periodic = find_independent_modes(sc, :L)

    # 2. Identify frozen modes (coupled only to non-L branches)
    frozen = find_frozen_modes(sc)

    # 3. Identify free modes (no capacitive coupling)
    free = find_free_modes(sc)

    # 4. Extended modes = remaining after periodic/frozen/free
    extended = find_extended_modes(sc, periodic, frozen, free)

    # 5. Sigma mode [1,1,...,1] if ungrounded
    sigma = is_grounded(sc) ? Int[] : [length(sc.graph.nodes)]

    # 6. Build & complete basis to full rank
    T = build_basis_matrix(periodic, extended, free, frozen, sigma)

    return T, Dict(:periodic => ..., :extended => ..., :free => ..., :frozen => ..., :sigma => ...)
end
```

### 4.2 Mode Classification Rules
| Mode Type | Identification | Basis | Dimension |
|-----------|---------------|-------|-----------|
| **Periodic** | Independent L-branch loop modes | Charge basis \|n>, n in {-ncut,...,ncut} | 2*ncut+1 |
| **Extended** | Remaining non-periodic dynamic modes | HO basis \|k> (Fock) or discretized phi grid | cutoff_ext or grid points |
| **Free** | No inductive coupling; conserved charge | Eliminated or frozen | - |
| **Frozen** | Solved from equilibrium conditions | Substituted out | - |

### 4.3 Frozen Variable Elimination
- Solve dV/d(theta_frozen) = 0 for frozen variables
- Substitute solutions back into potential → reduced Hamiltonian
- Uses Symbolics.jl `solve_for` or symbolic substitution

### 4.4 Purely Harmonic Detection
- If Hamiltonian has NO cos/sin terms → purely harmonic
- Diagonalize via generalized eigenvalue problem: L^{-1} v = lambda * C * v
- Normal mode frequencies: omega_i = sqrt(8 * EL_i * EC_i)
- Directly construct in Fock basis with ladder operators

---

## Phase 5: Quantization & Numerical Hamiltonian

### 5.1 Circuit Quantization (`src/circuit/quantization.jl`)

The `Circuit` struct — the main user-facing type:

```julia
mutable struct Circuit <: AbstractQuantumSystem
    symbolic_circuit::SymbolicCircuit

    # Numerical parameters (substituted values)
    param_values::Dict{Symbol, Float64}

    # Hilbert space configuration per variable
    cutoff_names::Vector{Symbol}       # :cutoff_n_1, :cutoff_ext_2, etc.
    cutoffs::Dict{Symbol, Int}         # truncation dimensions

    # Discretization for extended variables (if grid-based)
    phi_ranges::Dict{Int, Tuple{Float64, Float64}}

    # Cached numerical Hamiltonian
    _hamiltonian_cache::Union{Nothing, QuantumObject}

    # Hierarchical diagonalization (optional)
    system_hierarchy::Union{Nothing, Vector{Vector{Int}}}
    subsystem_trunc_dims::Union{Nothing, Vector{Int}}
end
```

### 5.2 Operator Construction (`src/circuit/circuit_operators.jl`)

**For periodic variables** (charge basis):
```julia
function n_operator_periodic(ncut::Int)        # diag(-ncut:ncut)
function exp_i_theta_operator(ncut::Int)       # shift operator |n> -> |n+1>
function cos_theta_operator(ncut::Int)         # (e^{i*theta} + e^{-i*theta})/2
function sin_theta_operator(ncut::Int)         # (e^{i*theta} - e^{-i*theta})/(2i)
```

**For extended variables (harmonic/Fock basis):**
```julia
function phi_operator_ho(cutoff::Int, osc_length::Float64)  # x0*(a + a†)/sqrt(2)
function n_operator_ho(cutoff::Int, osc_length::Float64)    # (a† - a)/(i*sqrt(2)*x0)
function cos_phi_operator_ho(cutoff::Int, osc_length::Float64)
function sin_phi_operator_ho(cutoff::Int, osc_length::Float64)
```
Uses QuantumToolbox.jl's `destroy()`, `create()` for ladder operators.

**For extended variables (discretized grid basis):**
```julia
function phi_operator_grid(grid::Grid1d)             # diagonal
function d_dphi_operator_grid(grid::Grid1d)          # finite-difference matrix
function d2_dphi2_operator_grid(grid::Grid1d)        # second derivative
function cos_phi_operator_grid(grid::Grid1d)         # diagonal cos values
```

### 5.3 Full Hamiltonian Assembly

```julia
function hamiltonian(circ::Circuit)::QuantumObject
    # 1. For each term in symbolic H, construct numerical operator
    # 2. Identity-wrap each operator to full tensor-product space
    # 3. Sum all terms: H_kinetic + H_potential + H_josephson
    # 4. Return as QuantumToolbox QuantumObject (sparse)
end
```

Key steps:
1. Map symbolic variables → operator matrices based on var_categories
2. For multi-DOF systems: `kron(I, op, I)` via identity_wrap
3. Evaluate symbolic expression by substituting operators for variables
4. Cache result; invalidate on parameter change

### 5.4 Eigenvalue Interface
Inherits from `qubit_base.jl` defaults:
- `eigenvals(circ::Circuit; evals_count=6)` — uses QuantumToolbox `eigenstates()`
- `eigensys(circ::Circuit; evals_count=6)` — eigenvalues + eigenvectors
- `get_spectrum_vs_paramvals(circ, :EJ, EJ_vals; evals_count=6)` — parameter sweep

---

## Phase 6: Oscillator & Transmon (Simple Qubit Types)

### 6.1 Oscillator (`src/oscillators/oscillator.jl`)
```julia
@kwdef mutable struct Oscillator <: AbstractOscillator
    E_osc::Float64
    truncated_dim::Int = 10
end
```
- Direct QuantumToolbox operators: `destroy`, `create`, `num`

### 6.2 KerrOscillator (`src/oscillators/kerr_oscillator.jl`)
```julia
@kwdef mutable struct KerrOscillator <: AbstractOscillator
    E_osc::Float64
    K::Float64           # Kerr nonlinearity
    truncated_dim::Int = 10
end
```

### 6.3 GenericQubit (`src/qubits/generic_qubit.jl`)
- Simple two-level system for testing the interface

### 6.4 Transmon (`src/qubits/transmon.jl`)
```julia
@kwdef mutable struct Transmon <: AbstractQubit1d
    EJ::Float64
    EC::Float64
    ng::Float64 = 0.0
    ncut::Int = 30
    truncated_dim::Int = 6
end
```
- H = 4EC(n - ng)^2 - EJ cos(phi) in charge basis
- Serves as validation: compare hardcoded Transmon vs Circuit-derived Transmon

### 6.5 TunableTransmon Validation
- Define TunableTransmon as a `Circuit` (two JJs in SQUID loop + shunt capacitor)
- Verify eigenvalues match expected analytical/numerical results
- This is the **primary end-to-end test** for the circuit quantization pipeline

---

## Phase 7: Composite Systems

### 7.1 HilbertSpace (`src/hilbert_space.jl`)
```julia
mutable struct HilbertSpace
    subsystems::Vector{AbstractQuantumSystem}
    interactions::Vector{InteractionTerm}
    # cached bare/dressed spectra
end
```
- `hamiltonian(hs)` → full tensor-product Hamiltonian
- `identity_wrap(op, subsys, hs)` → subsystem operator in full space
- `diag!(hs; evals_count)` → bare + dressed eigenvalues

### 7.2 InteractionTerm (`src/interaction.jl`)
```julia
struct InteractionTerm
    g_strength::Float64
    subsys_list::Vector{AbstractQuantumSystem}
    op_list::Vector{Function}
end
```

### 7.3 ParameterSweep (`src/param_sweep.jl`)
- Multi-dimensional parameter sweeps with `Threads.@threads` parallelization
- Bare/dressed spectra caching

---

## Phase 8: Hierarchical Diagonalization

### 8.1 Subsystem Decomposition (`src/circuit/hierarchical_diag.jl`)
```julia
function configure_hierarchical!(circ::Circuit;
    system_hierarchy::Vector{Vector{Int}},
    subsystem_trunc_dims::Vector{Int})
```
- Partition circuit variables into subsystems
- Diagonalize each subsystem independently
- Use low-energy eigenstates as basis for full system
- Build interaction terms between subsystems in eigenbasis

---

## Phase 9: Noise & Decoherence (Deferred)

### 9.1 Framework (`src/noise/`)
```julia
supported_noise_channels(sys)::Vector{Symbol}
t1_effective(sys; noise_channels=...)
t2_effective(sys; ...)
eval_noise_channel(sys, ::Val{:channel_name}, i, j; kwargs...)
```
- Val-dispatch pattern for extensibility
- Per-qubit and per-circuit channel implementations

---

## Phase 10: Plotting & Serialization (Deferred)

### 10.1 Makie Extension (`ext/ScQubitsMimicMakieExt/`)
- `plot_evals_vs_paramvals`, `plot_wavefunction`, `plot_matrixelements`, `plot_potential`
- Weak dependency via package extensions

### 10.2 Serialization
- JLD2.jl for save/load

---

## Directory Structure

```
scqubitsMimic.jl/
├── Project.toml
├── src/
│   ├── ScQubitsMimic.jl              # Main module, exports
│   ├── types.jl                      # Abstract type hierarchy
│   ├── constants.jl                  # Physical constants
│   ├── units.jl                      # Unit conversions
│   ├── grid.jl                       # Grid1d for discretization
│   ├── spectrum_data.jl              # SpectrumData types
│   ├── operators.jl                  # Operator helpers, identity_wrap
│   ├── qubit_base.jl                 # Default methods for AbstractQubit
│   ├── circuit/
│   │   ├── circuit_graph.jl          # Node, Branch, CircuitGraph
│   │   ├── circuit_input.jl          # YAML/string parser
│   │   ├── circuit_topology.jl       # Spanning tree, loop analysis
│   │   ├── symbolic_circuit.jl       # Symbolic Lagrangian/Hamiltonian
│   │   ├── mode_decomposition.jl     # Variable transformation & classification
│   │   ├── circuit_operators.jl      # Operator construction (charge/HO/grid basis)
│   │   ├── quantization.jl           # Circuit struct, numerical Hamiltonian
│   │   └── hierarchical_diag.jl      # Hierarchical diagonalization
│   ├── qubits/
│   │   ├── generic_qubit.jl
│   │   └── transmon.jl               # Hardcoded Transmon for validation
│   ├── oscillators/
│   │   ├── oscillator.jl
│   │   └── kerr_oscillator.jl
│   ├── noise/                        # Deferred
│   │   ├── noise_constants.jl
│   │   ├── noise_channels.jl
│   │   └── noise_methods.jl
│   ├── hilbert_space.jl
│   ├── interaction.jl
│   └── param_sweep.jl
├── ext/
│   └── ScQubitsMimicMakieExt/        # Deferred
├── test/
│   ├── runtests.jl
│   ├── data/                         # Reference data from Python scqubits
│   ├── test_transmon.jl
│   ├── test_circuit_graph.jl
│   ├── test_symbolic_circuit.jl
│   ├── test_mode_decomposition.jl
│   ├── test_quantization.jl
│   └── test_tunable_transmon_circuit.jl
└── docs/
    ├── make.jl
    └── src/
```

## Key Dependencies (Project.toml)

```toml
[deps]
QuantumToolbox = "..."
Symbolics = "..."
LinearAlgebra = "..."
SparseArrays = "..."
Graphs = "..."        # for circuit graph operations

[weakdeps]
CairoMakie = "..."
JLD2 = "..."

[extensions]
ScQubitsMimicMakieExt = "CairoMakie"
ScQubitsMimicJLD2Ext = "JLD2"
```

## Verification Plan

1. **Circuit graph**: Parse TunableTransmon YAML → correct nodes, branches, topology
2. **Symbolic analysis**: Capacitance/inductance matrices match hand calculation
3. **Mode decomposition**: Correct variable classification (periodic vs extended)
4. **Quantization**: Circuit-derived Transmon eigenvalues match hardcoded Transmon (< 1e-10)
5. **TunableTransmon**: Eigenvalues vs external flux sweep match Python scqubits results
6. **Composite system**: HilbertSpace with Circuit-based Transmon + Oscillator
7. **Benchmark**: Diagonalization speed vs Python scqubits
