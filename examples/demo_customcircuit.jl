# =============================================================================
# demo_customcircuit.jl — Circuit quantization from YAML description
#
# Julia equivalent of scqubits-examples/demo_customcircuit.ipynb
# Shows the graph → symbolic → mode decomposition → quantization pipeline
# =============================================================================

using ScQubitsMimic

# =============================================================================
# Example 1: Single Transmon (simplest circuit)
# =============================================================================
println("="^60)
println("Example 1: Single Transmon")
println("="^60)

desc1 = """
branches:
  - [JJ, 0, 1, EJ=30.0, EC=1.2]
"""

circ1 = Circuit(desc1; ncut=30)

println("\nCircuit description:")
println("  1 node, 1 JJ branch (ground ↔ node 1)")

# Symbolic analysis
println("\nSymbolic Hamiltonian (node basis):")
println("  H = ", sym_hamiltonian_node(circ1))

println("\nSymbolic Hamiltonian (mode basis):")
println("  H = ", sym_hamiltonian(circ1))

# Variable transformation
T, vc = variable_transformation(circ1)
println("\nVariable transformation matrix T:")
println("  T = ", T)
println("  Periodic modes: ", vc.periodic)
println("  Extended modes: ", vc.extended)

# External fluxes
println("  External fluxes: ", external_fluxes(circ1))
println("  Offset charges: ", offset_charges(circ1))

# Eigenvalues
evals1 = eigenvals(circ1; evals_count=4)
println("\nFirst 4 eigenvalues:")
for (i, e) in enumerate(evals1)
    println("  E_$(i-1) = $(round(e, digits=6)) GHz")
end
println("  ω₀₁ = $(round(evals1[2] - evals1[1], digits=6)) GHz")

# =============================================================================
# Example 2: Transmon with shunt capacitor
# =============================================================================
println("\n\n", "="^60)
println("Example 2: Transmon + Shunt Capacitor")
println("="^60)

desc2 = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.5]
  - [C, 0, 1, EC=1.0]
"""

circ2 = Circuit(desc2; ncut=30)

println("\nCircuit description:")
println("  JJ (EJ=10, EC=0.5) + C (EC=1.0) between ground and node 1")
println("  Total capacitance: C_JJ + C_shunt")

# Symbolic
println("\nSymbolic Hamiltonian:")
println("  ", sym_hamiltonian(circ2))

evals2 = eigenvals(circ2; evals_count=4)
w01 = evals2[2] - evals2[1]
println("\n  ω₀₁ = $(round(w01, digits=6)) GHz")

# =============================================================================
# Example 3: Tunable Transmon (SQUID loop)
# =============================================================================
println("\n\n", "="^60)
println("Example 3: SQUID (Tunable Transmon)")
println("="^60)

desc3 = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [C, 0, 1, EC=0.5]
"""

circ3 = Circuit(desc3; ncut=30)

println("\nCircuit: two JJs + shunt cap between ground and node 1")
println("  This forms a SQUID loop with tunable EJ")

# Symbolic
println("\nSymbolic Hamiltonian:")
println("  ", sym_hamiltonian(circ3))
println("External flux variables: ", external_fluxes(circ3))
println("Flux loop map: ", sym_external_fluxes(circ3))

# Flux sweep
println("\nSpectrum vs external flux:")
println("  Φ_ext     ω₀₁ (GHz)    ω₁₂ (GHz)    α (GHz)")
println("  " * "-"^50)
for phi in range(0.0, π, length=11)
    set_external_flux!(circ3, 1, phi)
    e = eigenvals(circ3; evals_count=4)
    w01 = e[2] - e[1]
    w12 = e[3] - e[2]
    alpha = w12 - w01
    println("  $(round(phi, digits=3))     $(round(w01, digits=4))       $(round(w12, digits=4))       $(round(alpha, digits=4))")
end

# Use get_spectrum_vs_paramvals for automated sweep
sd3 = get_spectrum_vs_paramvals(circ3, Symbol("Φ1"), range(0.0, π, length=21); evals_count=4)
println("\nSweep completed: $(length(sd3.param_vals)) points, $(size(sd3.eigenvalues, 2)) eigenvalues each")

# =============================================================================
# Example 4: Asymmetric SQUID
# =============================================================================
println("\n\n", "="^60)
println("Example 4: Asymmetric SQUID (EJ1 ≠ EJ2)")
println("="^60)

desc4 = """
branches:
  - [JJ, 0, 1, EJ=12.0, EC=0.3]
  - [JJ, 0, 1, EJ=8.0, EC=0.3]
  - [C, 0, 1, EC=0.5]
"""

circ4 = Circuit(desc4; ncut=30)

d_asym = (12.0 - 8.0) / (12.0 + 8.0)
println("\nAsymmetry parameter d = (EJ1-EJ2)/(EJ1+EJ2) = $d_asym")
println("At frustration (Φ=π), residual EJ = EJmax * d = $(20.0 * d_asym)")

println("\nω₀₁ comparison: symmetric vs asymmetric SQUID")
println("  Φ_ext     symmetric    asymmetric")
println("  " * "-"^40)
for phi in range(0.0, π, length=6)
    set_external_flux!(circ3, 1, phi)
    set_external_flux!(circ4, 1, phi)
    e_sym = eigenvals(circ3; evals_count=2)
    e_asym = eigenvals(circ4; evals_count=2)
    w_sym = e_sym[2] - e_sym[1]
    w_asym = e_asym[2] - e_asym[1]
    println("  $(round(phi, digits=3))     $(round(w_sym, digits=4))      $(round(w_asym, digits=4))")
end

# =============================================================================
# Example 5: Circuit-derived vs hardcoded validation
# =============================================================================
println("\n\n", "="^60)
println("Example 5: Validation against hardcoded Transmon")
println("="^60)

# EC_branch = 0.3 gives EC_total = 4 * 0.3 = 1.2 GHz (single JJ to ground)
desc_val = """
branches:
  - [JJ, 0, 1, EJ=30.0, EC=0.3]
"""
circ_val = Circuit(desc_val; ncut=30)
tmon_val = Transmon(EJ=30.0, EC=1.2, ng=0.0, ncut=30, truncated_dim=6)

e_circ = eigenvals(circ_val; evals_count=6)
e_tmon = eigenvals(tmon_val; evals_count=6)

println("\n  Level    Circuit         Hardcoded       Difference")
println("  " * "-"^55)
for i in 1:6
    diff = abs(e_circ[i] - e_tmon[i])
    println("  E_$(i-1)     $(round(e_circ[i], digits=8))    $(round(e_tmon[i], digits=8))    $(diff)")
end
println("\n  Maximum difference: $(maximum(abs.(e_circ .- e_tmon)))")
println("  Validation: ", maximum(abs.(e_circ .- e_tmon)) < 1e-10 ? "PASSED" : "FAILED")
