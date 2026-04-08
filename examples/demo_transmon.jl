# =============================================================================
# demo_transmon.jl — Transmon qubit basics
#
# Julia equivalent of scqubits-examples/demo_transmon.ipynb
# =============================================================================

using ScQubitsMimic

# --- 1. Create a Transmon qubit ---
tmon = Transmon(EJ=30.0, EC=1.2, ng=0.0, ncut=30, truncated_dim=6)

println("Transmon parameters: EJ=$(tmon.EJ) GHz, EC=$(tmon.EC) GHz")
println("Hilbert space dimension: $(hilbertdim(tmon))")

# --- 2. Energy eigenvalues ---
evals = eigenvals(tmon; evals_count=6)
println("\nFirst 6 eigenvalues (GHz):")
for (i, e) in enumerate(evals)
    println("  E_$(i-1) = $(round(e, digits=6))")
end

# Transition frequencies (relative to ground state)
println("\nTransition frequencies from ground state:")
for i in 2:6
    println("  ω_{0→$(i-1)} = $(round(evals[i] - evals[1], digits=6)) GHz")
end

# --- 3. Anharmonicity ---
omega_01 = evals[2] - evals[1]
omega_12 = evals[3] - evals[2]
anharmonicity = omega_12 - omega_01
println("\nω₀₁ = $(round(omega_01, digits=6)) GHz")
println("ω₁₂ = $(round(omega_12, digits=6)) GHz")
println("Anharmonicity α = ω₁₂ - ω₀₁ = $(round(anharmonicity, digits=6)) GHz")
println("Expected (transmon limit): α ≈ -EC = $(round(-tmon.EC, digits=6)) GHz")

# --- 4. Transmon approximation check ---
# In the transmon regime (EJ >> EC): ω₀₁ ≈ √(8 EJ EC) - EC
omega_approx = sqrt(8 * tmon.EJ * tmon.EC) - tmon.EC
println("\nTransmon approximation: ω₀₁ ≈ √(8 EJ EC) - EC = $(round(omega_approx, digits=6)) GHz")
println("Exact: ω₀₁ = $(round(omega_01, digits=6)) GHz")
println("Error: $(round(abs(omega_01 - omega_approx) / omega_01 * 100, digits=3))%")

# --- 5. Charge dispersion (varying ng) ---
println("\nCharge dispersion (ω₀₁ vs offset charge ng):")
for ng in [0.0, 0.1, 0.25, 0.5]
    tmon.ng = ng
    e = eigenvals(tmon; evals_count=2)
    w01 = e[2] - e[1]
    println("  ng = $ng → ω₀₁ = $(round(w01, digits=8)) GHz")
end
tmon.ng = 0.0  # reset

# --- 6. Parameter sweep: EJ dependence ---
println("\nEigenvalue sweep over EJ:")
sd = get_spectrum_vs_paramvals(tmon, :EJ, range(10.0, 50.0, length=5); evals_count=4)
for (i, ej) in enumerate(sd.param_vals)
    e = sd.eigenvalues[i, :]
    w01 = e[2] - e[1]
    println("  EJ = $(round(ej, digits=1)) → ω₀₁ = $(round(w01, digits=4)) GHz")
end

# --- 7. Matrix elements ---
println("\nCharge operator matrix elements ⟨i|n̂|j⟩:")
n_op = n_operator_periodic(tmon.ncut)
mel_table = matrixelement_table(tmon, n_op; evals_count=4)
for i in 1:4, j in 1:4
    v = mel_table[i, j]
    if abs(v) > 1e-6
        println("  ⟨$(i-1)|n̂|$(j-1)⟩ = $(round(v, digits=6))")
    end
end

# --- 8. Circuit-derived Transmon (validation) ---
# In Circuit, a JJ branch EC defines the junction capacitance: C_JJ = 1/(8*EC).
# For a single JJ to ground: EC_total = C^{-1}/2 = 4*EC_branch.
# So to get EC_total = 1.2 GHz, we set EC_branch = 0.3 GHz.
println("\n--- Validation: Circuit-derived Transmon ---")
desc = """
branches:
  - [JJ, 0, 1, EJ=30.0, EC=0.3]
"""
circ = Circuit(desc; ncut=30)
evals_circ = eigenvals(circ; evals_count=4)
evals_hard = eigenvals(Transmon(EJ=30.0, EC=1.2, ncut=30, truncated_dim=4); evals_count=4)

println("Hardcoded (EC=1.2):       ", round.(evals_hard, digits=8))
println("Circuit (EC_branch=0.3):  ", round.(evals_circ, digits=8))
println("Max difference:           ", maximum(abs.(evals_circ .- evals_hard)))
