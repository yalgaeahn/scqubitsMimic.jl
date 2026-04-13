# API Audit

This page is the source-of-truth checklist for the public API exported by
`ScQubitsMimic`. Every export appears exactly once below and points to its
primary reference page.

## [`api/core.md`](./api/core.md)

`AbstractQuantumSystem`, `AbstractQubit`, `AbstractQubit1d`, `AbstractQubitNd`,
`AbstractOscillator`, `PhysicalConstants`, `convert_units`, `Grid1d`,
`SpectrumData`, `grid_points`, `grid_spacing`, `hamiltonian`, `eigenvals`,
`eigensys`, `hilbertdim`, `matrixelement`, `matrixelement_table`,
`get_spectrum_vs_paramvals`, `set_param!`, `get_param`

## [`api/qubits.md`](./api/qubits.md)

`GenericQubit`, `Transmon`, `TunableTransmon`, `n_operator`,
`exp_i_phi_operator`, `cos_phi_operator`, `sin_phi_operator`, `potential`,
`ej_effective`

## [`api/oscillators.md`](./api/oscillators.md)

`Oscillator`, `KerrOscillator`, `annihilation_operator`,
`creation_operator`, `number_operator`

## [`api/circuit.md`](./api/circuit.md)

`BranchType`, `C_branch`, `L_branch`, `JJ_branch`, `CJ_branch`, `Branch`,
`CircuitGraph`, `parse_circuit`, `find_spanning_tree`,
`find_closure_branches`, `find_fundamental_loops`,
`find_superconducting_loops`, `Circuit`, `set_external_flux!`,
`set_offset_charge!`, `invalidate_cache!`, `list_branch_params`,
`sym_hamiltonian`, `sym_hamiltonian_node`, `sym_lagrangian`,
`variable_transformation`, `offset_charge_transformation`,
`external_fluxes`, `sym_external_fluxes`, `offset_charges`

## [`api/symbolic-circuit.md`](./api/symbolic-circuit.md)

`SymbolicCircuit`, `build_symbolic_circuit`,
`compute_variable_transformation`, `VarCategories`

## [`api/circuit-operators.md`](./api/circuit-operators.md)

`n_operator_periodic`, `exp_i_theta_operator`, `cos_theta_operator`,
`sin_theta_operator`, `phi_operator_ho`, `n_operator_ho`,
`cos_phi_operator_ho`, `sin_phi_operator_ho`, `phi_operator_grid`,
`d_dphi_operator_grid`, `d2_dphi2_operator_grid`,
`cos_phi_operator_grid`, `sin_phi_operator_grid`

## [`api/circuit-hierarchy.md`](./api/circuit-hierarchy.md)

`configure!`, `sym_interaction`, `SubCircuit`, `HierarchyLeaf`,
`HierarchyGroup`, `HierarchyNode`, `hierarchical_diag`,
`truncation_template`

## [`api/hilbert-space.md`](./api/hilbert-space.md)

`identity_wrap`, `HilbertSpace`, `InteractionTerm`, `SpectrumLookup`,
`add_interaction!`, `add_operator!`, `diag!`

## [`api/sweeps-and-lookup.md`](./api/sweeps-and-lookup.md)

`SingleSystemSweep`, `ParameterSweep`, `SweepSlice`, `run!`,
`generate_lookup!`, `lookup_exists`, `dressed_index`, `bare_index`,
`energy_by_dressed_index`, `energy_by_bare_index`,
`dressed_state_components`, `op_in_dressed_eigenbasis`,
`OVERLAP_THRESHOLD`

## [`api/analysis-and-transitions.md`](./api/analysis-and-transitions.md)

`transitions`, `chi_matrix`, `self_kerr`, `lamb_shift`

## [`api/plotting.md`](./api/plotting.md)

`plot_evals_vs_paramvals`, `plot_matrixelements`, `plot_wavefunction`,
`plot_chi_vs_paramvals`, `plot_transitions`

Total audited exports: `113`
