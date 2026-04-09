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
include("dispersive.jl")
include("effective_hamiltonian.jl")

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
export HilbertSpace, InteractionTerm, ParameterSweep, HilbertSpaceSweep, SpectrumLookup
export add_interaction!, add_operator!, diag!
export generate_lookup!, dressed_index, bare_index,
       energy_by_dressed_index, energy_by_bare_index,
       op_in_dressed_eigenbasis, OVERLAP_THRESHOLD
export chi_matrix, self_kerr, lamb_shift
export effective_hamiltonian, exchange_coupling, avoided_crossing_coupling

# Plotting API stubs (implemented in ext/ScQubitsMimicMakieExt)
function plot_evals_vs_paramvals end
function plot_matrixelements end
function plot_wavefunction end
function plot_chi_vs_paramvals end

export plot_evals_vs_paramvals, plot_matrixelements,
       plot_wavefunction, plot_chi_vs_paramvals

end # module
