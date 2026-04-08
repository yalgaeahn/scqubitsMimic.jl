# =============================================================================
# demo_parametersweep.jl — Parameter sweeps over composite systems
#
# Julia equivalent of scqubits-examples/demo_parametersweep.ipynb
# =============================================================================

using ScQubitsMimic

# --- 1. Define the system: TunableTransmon + Oscillator ---
tmon = TunableTransmon(EJmax=20.0, EC=0.3, d=0.1, flux=0.0, ng=0.0,
                       ncut=15, truncated_dim=4)
osc = Oscillator(E_osc=5.5, truncated_dim=6)

hs = HilbertSpace([tmon, osc])

# Coupling: g * n̂_tmon ⊗ (a + a†)
g = 0.15
add_interaction!(hs, g, [tmon, osc],
    [s -> n_operator_periodic(s.ncut),
     s -> annihilation_operator(s) + creation_operator(s)])

println("System: TunableTransmon + Oscillator")
println("  TunableTransmon: EJmax=$(tmon.EJmax), EC=$(tmon.EC), d=$(tmon.d)")
println("  Oscillator: ω_r=$(osc.E_osc) GHz")
println("  Coupling: g=$g GHz")
println("  Total Hilbert space: $(hilbertdim(hs))")

# --- 2. Single-subsystem sweep ---
println("\n--- Single-qubit flux sweep ---")
sd = get_spectrum_vs_paramvals(tmon, :flux, range(0.0, 0.5, length=11); evals_count=4)
println("  ω₀₁ vs flux:")
for (i, flux) in enumerate(sd.param_vals)
    w01 = sd.eigenvalues[i, 2] - sd.eigenvalues[i, 1]
    println("    Φ/Φ₀ = $(round(flux, digits=2)) → ω₀₁ = $(round(w01, digits=4)) GHz")
end

# --- 3. HilbertSpace sweep: flux dependence of dressed spectrum ---
println("\n--- HilbertSpace flux sweep (dressed spectrum) ---")
flux_vals = collect(range(0.0, 0.5, length=11))

sweep = HilbertSpaceSweep(hs,
    Dict(:flux => flux_vals),
    (hs, vals) -> begin
        tmon.flux = vals[:flux]
    end;
    evals_count=8
)

println("Dressed ω₀₁ vs flux:")
for (i, flux) in enumerate(flux_vals)
    w01 = sweep.dressed_evals[i, 2] - sweep.dressed_evals[i, 1]
    println("  Φ/Φ₀ = $(round(flux, digits=2)) → ω₀₁ = $(round(w01, digits=4)) GHz")
end

# --- 4. Bare vs dressed comparison ---
println("\n--- Bare vs Dressed transitions ---")
println("Bare qubit ω₀₁ | Resonator ω | Dressed ω₀₁")
for (i, flux) in enumerate(flux_vals)
    tmon.flux = flux
    bare_w01 = let e = eigenvals(tmon; evals_count=2); e[2] - e[1] end
    dressed_w01 = sweep.dressed_evals[i, 2] - sweep.dressed_evals[i, 1]
    println("  $(round(bare_w01, digits=4))       | $(osc.E_osc)   | $(round(dressed_w01, digits=4))")
end
tmon.flux = 0.0  # reset

# --- 5. Sweep with lookup (state tracking) ---
println("\n--- Sweep with bare/dressed state tracking ---")
flux_fine = collect(range(0.0, 0.5, length=6))

sweep2 = HilbertSpaceSweep(hs,
    Dict(:flux => flux_fine),
    (hs, vals) -> begin
        tmon.flux = vals[:flux]
    end;
    evals_count=10,
    store_lookups=true
)

println("State tracking across flux sweep:")
for (i, flux) in enumerate(flux_fine)
    println("\n  Φ/Φ₀ = $(round(flux, digits=3)):")
    lk = sweep2.lookups[i]
    for di in 1:min(6, length(lk.dressed_evals))
        bi = lk.dressed_to_bare[di]
        e = lk.dressed_evals[di]
        println("    dressed[$di] → |q=$(bi[1]-1), n=$(bi[2]-1)⟩  E=$(round(e, digits=4)) GHz")
    end
end

# --- 6. Dispersive shift vs flux ---
println("\n--- Dispersive shift χ vs flux ---")
for (i, flux) in enumerate(flux_fine)
    lk = sweep2.lookups[i]
    try
        E00 = lk.dressed_evals[lk.bare_to_dressed[(1,1)]]
        E10 = lk.dressed_evals[lk.bare_to_dressed[(2,1)]]
        E01 = lk.dressed_evals[lk.bare_to_dressed[(1,2)]]
        E11 = lk.dressed_evals[lk.bare_to_dressed[(2,2)]]
        chi = (E11 - E10) - (E01 - E00)
        println("  Φ/Φ₀ = $(round(flux, digits=3)) → χ/2π = $(round(chi * 1000, digits=2)) MHz")
    catch ex
        println("  Φ/Φ₀ = $(round(flux, digits=3)) → χ could not be computed")
    end
end

# --- 7. Multi-qubit system ---
println("\n\n========================================")
println("Two TunableTransmons + Resonator bus")
println("========================================\n")

tmon1 = TunableTransmon(EJmax=25.0, EC=0.4, d=0.05, flux=0.0, ncut=10, truncated_dim=3)
tmon2 = TunableTransmon(EJmax=22.0, EC=0.35, d=0.08, flux=0.0, ncut=10, truncated_dim=3)
bus = Oscillator(E_osc=6.0, truncated_dim=4)

hs3 = HilbertSpace([tmon1, tmon2, bus])

# Qubit-bus couplings
g1 = 0.12
g2 = 0.10
add_interaction!(hs3, g1, [tmon1, bus],
    [s -> n_operator_periodic(s.ncut),
     s -> annihilation_operator(s) + creation_operator(s)])
add_interaction!(hs3, g2, [tmon2, bus],
    [s -> n_operator_periodic(s.ncut),
     s -> annihilation_operator(s) + creation_operator(s)])

println("Qubit 1: EJmax=$(tmon1.EJmax), EC=$(tmon1.EC)")
println("Qubit 2: EJmax=$(tmon2.EJmax), EC=$(tmon2.EC)")
println("Bus resonator: ω=$(bus.E_osc) GHz")
println("Total Hilbert space: $(hilbertdim(hs3))")

# Sweep qubit 1 flux
sweep3 = HilbertSpaceSweep(hs3,
    Dict(:flux1 => collect(range(0.0, 0.4, length=5))),
    (hs, vals) -> begin
        tmon1.flux = vals[:flux1]
    end;
    evals_count=8,
    store_lookups=true
)

println("\nDressed spectrum vs qubit 1 flux:")
for (i, flux) in enumerate(sweep3.param_vals[:flux1])
    lk = sweep3.lookups[i]
    println("  Φ₁/Φ₀ = $(round(flux, digits=2)):")
    for di in 1:min(5, length(lk.dressed_evals))
        bi = lk.dressed_to_bare[di]
        e = lk.dressed_evals[di]
        println("    dressed[$di] → |$(bi[1]-1),$(bi[2]-1),$(bi[3]-1)⟩  E=$(round(e, digits=3)) GHz")
    end
end
