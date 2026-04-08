# =============================================================================
# demo_tunable_transmon.jl — Flux-tunable transmon (SQUID)
#
# Julia equivalent of scqubits-examples/scqubits_paper_source_code.ipynb
# and the TunableTransmon portions of demo_parametersweep.ipynb
# =============================================================================

using ScQubitsMimic

# --- 1. Create a TunableTransmon ---
# EJmax = EJ1 + EJ2, d = (EJ2 - EJ1)/(EJ1 + EJ2) is junction asymmetry
tmon = TunableTransmon(EJmax=15.0, EC=0.6, d=0.1, flux=0.0, ncut=30, truncated_dim=6)

println("TunableTransmon: EJmax=$(tmon.EJmax) GHz, EC=$(tmon.EC) GHz, d=$(tmon.d)")

# --- 2. Effective Josephson energy vs flux ---
println("\nEffective EJ vs external flux (units of Φ₀):")
for flux in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    tmon.flux = flux
    ej_eff = ej_effective(tmon)
    println("  Φ_ext/Φ₀ = $flux → EJ_eff = $(round(ej_eff, digits=4)) GHz")
end

# --- 3. Eigenvalue spectrum vs flux ---
println("\nTransition frequency ω₀₁ vs flux:")
flux_vals = range(0.0, 0.5, length=11)
for flux in flux_vals
    tmon.flux = flux
    e = eigenvals(tmon; evals_count=3)
    w01 = e[2] - e[1]
    w12 = e[3] - e[2]
    alpha = w12 - w01
    println("  Φ/Φ₀ = $(round(flux, digits=2)) → ω₀₁ = $(round(w01, digits=4)), α = $(round(alpha, digits=4)) GHz")
end

# --- 4. Comparison: symmetric vs asymmetric SQUID ---
println("\n--- Symmetric (d=0) vs Asymmetric (d=0.1) ---")
tmon_sym = TunableTransmon(EJmax=15.0, EC=0.6, d=0.0, flux=0.0, ncut=30, truncated_dim=4)
tmon_asym = TunableTransmon(EJmax=15.0, EC=0.6, d=0.1, flux=0.0, ncut=30, truncated_dim=4)

println("  At Φ/Φ₀ = 0.5 (frustration point):")
tmon_sym.flux = 0.5
tmon_asym.flux = 0.5
println("  Symmetric:   EJ_eff = $(round(ej_effective(tmon_sym), digits=6)) GHz")
println("  Asymmetric:  EJ_eff = $(round(ej_effective(tmon_asym), digits=6)) GHz")

e_sym = eigenvals(tmon_sym; evals_count=2)
e_asym = eigenvals(tmon_asym; evals_count=2)
println("  Symmetric:   ω₀₁ = $(round(e_sym[2] - e_sym[1], digits=4)) GHz")
println("  Asymmetric:  ω₀₁ = $(round(e_asym[2] - e_asym[1], digits=4)) GHz")

# --- 5. Full parameter sweep ---
println("\n--- Full flux sweep via get_spectrum_vs_paramvals ---")
tmon.flux = 0.0
sd = get_spectrum_vs_paramvals(tmon, :flux, range(0.0, 0.5, length=6); evals_count=4)
println("Eigenvalues at each flux point:")
for (i, flux) in enumerate(sd.param_vals)
    e = sd.eigenvalues[i, :]
    println("  Φ/Φ₀ = $(round(flux, digits=2)) → E = $(round.(e, digits=3))")
end

# --- 6. Validate against Circuit-derived SQUID ---
println("\n--- Validation: Circuit-derived SQUID ---")
# Symmetric SQUID (2 JJs, no shunt cap): EJ1 = EJ2 = EJmax/2 = 7.5
# For 2 JJs between ground and node 1:
#   C_total = C1 + C2 = 2/(8*EC_JJ) → EC_total = 1/(2*C_total) = 4*EC_JJ/2 = 2*EC_JJ
#   With EC_JJ = 0.3: EC_total = 0.6 GHz ✓
desc = """
branches:
  - [JJ, 0, 1, EJ=7.5, EC=0.3]
  - [JJ, 0, 1, EJ=7.5, EC=0.3]
"""
circ = Circuit(desc; ncut=30)
println("Number of external fluxes: $(length(circ.external_flux_values))")

println("\nComparing at different flux values:")
for flux in [0.0, 0.25]
    # TunableTransmon (hardcoded)
    tt = TunableTransmon(EJmax=15.0, EC=0.6, d=0.0, flux=flux, ncut=30, truncated_dim=3)
    e_tt = eigenvals(tt; evals_count=3)

    # Circuit SQUID: Φext = 2π * flux
    set_external_flux!(circ, 1, 2π * flux)
    e_circ = eigenvals(circ; evals_count=3)

    w01_tt = e_tt[2] - e_tt[1]
    w01_circ = e_circ[2] - e_circ[1]
    println("  Φ/Φ₀ = $flux: TunableTransmon ω₀₁=$(round(w01_tt, digits=4)), " *
            "Circuit ω₀₁=$(round(w01_circ, digits=4))")
end
