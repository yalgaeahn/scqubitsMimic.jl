# Circuit Operators

These helpers expose the low-level operators used by circuit quantization.
They are useful when building custom diagnostics or comparing the Julia
implementation against scqubits expressions.

```@docs
n_operator_periodic
exp_i_theta_operator
cos_theta_operator
sin_theta_operator
phi_operator_ho
n_operator_ho
cos_phi_operator_ho
sin_phi_operator_ho
phi_operator_grid
d_dphi_operator_grid
d2_dphi2_operator_grid
cos_phi_operator_grid
sin_phi_operator_grid
```
