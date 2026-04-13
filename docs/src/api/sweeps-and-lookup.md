# Sweeps And Lookup

This page documents the public sweep types plus the lookup-driven accessors
used for bare/dressed labeling and state inspection.

```@docs
SingleSystemSweep
ParameterSweep
SweepSlice
run!
generate_lookup!
lookup_exists
dressed_index
bare_index
energy_by_dressed_index
energy_by_bare_index
dressed_state_components
op_in_dressed_eigenbasis
OVERLAP_THRESHOLD
```

## Notes

- `SingleSystemSweep` is the lightweight single-system helper. `ParameterSweep`
  is the public coupled-system sweep API.
- `SweepSlice` is the bridge from an `N`-dimensional sweep to one-dimensional
  transition plots and fixed-point state analysis.
- Sweep lookup accessors use explicit `param_indices` to select a point in the
  Cartesian product of sweep parameters.
