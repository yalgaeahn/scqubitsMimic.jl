module ScQubitsMimic

using LinearAlgebra
using SparseArrays
using QuantumToolbox
using Symbolics
using Latexify: latexify
using Graphs

# Core infrastructure
include("types.jl")
include("constants.jl")
include("units.jl")
include("grid.jl")
include("spectrum_data.jl")
include("operators.jl")
include("diag.jl")
include("qubit_base.jl")

# Circuit quantization pipeline
include("circuit/circuit_graph.jl")
include("circuit/circuit_input.jl")
include("circuit/circuit_topology.jl")
include("circuit/symbolic_circuit.jl")
include("circuit/mode_decomposition.jl")
include("circuit/circuit_operators.jl")
include("circuit/quantization.jl")
include("circuit/hierarchical_diag.jl")

# Simple qubit types
include("qubits/generic_qubit.jl")
include("qubits/transmon.jl")
include("qubits/tunable_transmon.jl")

# Oscillators
include("oscillators/oscillator.jl")
include("oscillators/kerr_oscillator.jl")

# Composite systems
include("interaction.jl")
include("hilbert_space.jl")
include("spectrum_lookup.jl")
include("param_sweep.jl")
include("transitions.jl")
include("dispersive.jl")

# ── Exports ──────────────────────────────────────────────────────────────────

# Types
export AbstractQuantumSystem, AbstractQubit, AbstractQubit1d, AbstractQubitNd,
       AbstractOscillator

# Constants & Units
export PhysicalConstants, convert_units

# Grid & Spectrum
export Grid1d, SpectrumData, grid_points, grid_spacing

# Base interface
export hamiltonian, eigenvals, eigensys, hilbertdim,
       matrixelement, matrixelement_table,
       get_spectrum_vs_paramvals,
       set_param!, get_param

# Operators
export identity_wrap

# Circuit pipeline
export BranchType, C_branch, L_branch, JJ_branch, CJ_branch
export Branch, CircuitGraph, parse_circuit
export find_spanning_tree, find_closure_branches, find_fundamental_loops,
       find_superconducting_loops
export SymbolicCircuit, build_symbolic_circuit
export compute_variable_transformation, VarCategories
export Circuit, set_external_flux!, set_offset_charge!, invalidate_cache!,
       list_branch_params, configure!
export sym_hamiltonian, sym_hamiltonian_node, sym_lagrangian,
       sym_interaction,
       variable_transformation, offset_charge_transformation,
       external_fluxes, sym_external_fluxes, offset_charges
export SubCircuit, HierarchyLeaf, HierarchyGroup, HierarchyNode, hierarchical_diag
export truncation_template

# Circuit operators
export n_operator_periodic, exp_i_theta_operator,
       cos_theta_operator, sin_theta_operator
export phi_operator_ho, n_operator_ho, cos_phi_operator_ho, sin_phi_operator_ho
export phi_operator_grid, d_dphi_operator_grid, d2_dphi2_operator_grid,
       cos_phi_operator_grid, sin_phi_operator_grid

# Qubit types
export GenericQubit, Transmon, TunableTransmon, ej_effective

# Oscillators
export Oscillator, KerrOscillator
export annihilation_operator, creation_operator, number_operator

# Qubit-specific operators
export n_operator, exp_i_phi_operator, cos_phi_operator, sin_phi_operator, potential

# Composite systems
export HilbertSpace, InteractionTerm, SingleSystemSweep, ParameterSweep, SweepSlice, SpectrumLookup
export add_interaction!, add_operator!, diag!
export run!
export generate_lookup!, lookup_exists, dressed_index, bare_index,
       energy_by_dressed_index, energy_by_bare_index, dressed_state_components,
       transitions,
       op_in_dressed_eigenbasis, OVERLAP_THRESHOLD
export chi_matrix, self_kerr, lamb_shift

# Plotting API stubs (implemented in ext/ScQubitsMimicMakieExt)
"""
    plot_evals_vs_paramvals(sweep; kwargs...) -> Figure

Plot eigenvalue traces from a [`SingleSystemSweep`](@ref) or
[`ParameterSweep`](@ref).

The plotting methods are provided by the `ScQubitsMimicMakieExt` extension and
require `CairoMakie` to be loaded.
"""
function plot_evals_vs_paramvals end

"""
    plot_matrixelements(sys, op; kwargs...) -> Figure

Plot a matrix-element heatmap for an operator acting on a supported quantum
system.

This plotting entry point is implemented by the `ScQubitsMimicMakieExt`
extension and requires `CairoMakie`.
"""
function plot_matrixelements end

"""
    plot_wavefunction(sys, which=1; kwargs...) -> Figure

Plot one or more 1D qubit wavefunctions using the Makie extension backend.
Requires `CairoMakie` and the `ScQubitsMimicMakieExt` extension.
"""
function plot_wavefunction end

"""
    plot_chi_vs_paramvals(sweep::ParameterSweep; kwargs...) -> Figure

Plot dispersive `χ` values extracted from a sweep with stored lookup data.
Implemented by the `ScQubitsMimicMakieExt` extension and requires `CairoMakie`.
"""
function plot_chi_vs_paramvals end

"""
    plot_transitions(sweep_or_slice; kwargs...) -> Figure

Plot transition energies from a one-dimensional [`ParameterSweep`](@ref) or a
[`SweepSlice`](@ref).

This plotting entry point is implemented by the `ScQubitsMimicMakieExt`
extension and requires `CairoMakie`.
"""
function plot_transitions end

export plot_evals_vs_paramvals, plot_matrixelements,
       plot_wavefunction, plot_chi_vs_paramvals, plot_transitions

end # module
