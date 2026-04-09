# =============================================================================
# demo_hilbertspace.jl — Composite quantum systems
#
# Julia equivalent of scqubits-examples/demo_hilbertspace.ipynb
# =============================================================================

using ScQubitsMimic
using LinearAlgebra

# --- 1. Define subsystems ---
# Two Kerr oscillators coupled via a shared inductance (rf SQUID coupling)
kosc1 = KerrOscillator(E_osc=4.5, K=0.05, truncated_dim=8)
kosc2 = KerrOscillator(E_osc=6.0, K=0.03, truncated_dim=8)

println("Subsystem 1: KerrOscillator, ω=$(kosc1.E_osc) GHz, K=$(kosc1.K) GHz")
println("Subsystem 2: KerrOscillator, ω=$(kosc2.E_osc) GHz, K=$(kosc2.K) GHz")

# --- 2. Create HilbertSpace ---
hs = HilbertSpace([kosc1, kosc2])
println("\nTotal Hilbert space dimension: $(hilbertdim(hs))")

# --- 3. Add interaction ---
# Coupling: g * (a₁† a₂ + a₁ a₂†) — beam-splitter interaction
g = 0.1  # GHz
add_interaction!(hs, g, [kosc1, kosc2],
    [s -> creation_operator(s), s -> annihilation_operator(s)])
add_interaction!(hs, g, [kosc1, kosc2],
    [s -> annihilation_operator(s), s -> creation_operator(s)])

println("Added beam-splitter interaction: g = $g GHz")

# --- 4. Bare Hamiltonian ---
println("\n--- Bare eigenvalues (no interaction) ---")
hs_bare = HilbertSpace([kosc1, kosc2])  # no interaction
bare_vals = eigenvals(hs_bare; evals_count=10)
for (i, e) in enumerate(bare_vals)
    println("  E_$(i-1) = $(round(e, digits=6)) GHz")
end

# --- 5. Dressed (full) Hamiltonian ---
println("\n--- Dressed eigenvalues (with interaction) ---")
dressed_vals = eigenvals(hs; evals_count=10)
for (i, e) in enumerate(dressed_vals)
    println("  E_$(i-1) = $(round(e, digits=6)) GHz")
end

# --- 6. Generate lookup: bare ↔ dressed mapping ---
println("\n--- Bare ↔ Dressed state mapping ---")
lookup = generate_lookup!(hs; evals_count=10)

println("State mapping (bare → dressed):")
for i in 1:10
    bi = bare_index(hs, i)
    e = energy_by_dressed_index(hs, i)
    println("  dressed[$i] → bare$(bi), E = $(round(e, digits=6)) GHz")
end

# --- 7. Specific state lookup ---
println("\n--- Looking up specific bare states ---")
states_to_find = [(1,1), (2,1), (1,2), (2,2), (3,1), (1,3)]
for (n1, n2) in states_to_find
    try
        di = dressed_index(hs, n1, n2)
        e = energy_by_bare_index(hs, n1, n2)
        println("  |$(n1-1),$(n2-1)⟩ → dressed[$di], E = $(round(e, digits=6)) GHz")
    catch ex
        println("  |$(n1-1),$(n2-1)⟩ → not in lookup")
    end
end

# --- 8. Operator in dressed eigenbasis ---
println("\n--- Number operator in dressed eigenbasis ---")
# n̂₁ in full space
dims = [hilbertdim(s) for s in hs.subsystems]
n1_full = identity_wrap(number_operator(kosc1), 1, dims)

n1_dressed = op_in_dressed_eigenbasis(hs, n1_full; truncated_dim=5)
println("⟨dressed_i|n̂₁|dressed_j⟩ (5×5):")
for i in 1:5
    row = ["$(round(real(n1_dressed[i,j]), digits=4))" for j in 1:5]
    println("  ", join(row, "  "))
end

# --- 9. Transmon + Oscillator system ---
println("\n\n========================================")
println("Transmon + Oscillator (dispersive readout)")
println("========================================\n")

tmon = Transmon(EJ=30.0, EC=1.2, ng=0.0, ncut=15, truncated_dim=6)
res = Oscillator(E_osc=7.0, truncated_dim=10)

hs2 = HilbertSpace([tmon, res])

# Jaynes-Cummings coupling: g * (n̂_tmon ⊗ (a + a†))
g_jc = 0.2
add_interaction!(hs2, g_jc, [tmon, res],
    [s -> n_operator_periodic(s.ncut), s -> annihilation_operator(s) + creation_operator(s)])

println("Transmon: EJ=$(tmon.EJ), EC=$(tmon.EC) GHz")
println("Resonator: ω_r=$(res.E_osc) GHz")
println("Coupling: g=$(g_jc) GHz")
println("Total dimension: $(hilbertdim(hs2))")

# Dressed spectrum
lookup2 = generate_lookup!(hs2; evals_count=15)

println("\nDressed spectrum with state assignments:")
for i in 1:15
    bi = bare_index(hs2, i)
    e = energy_by_dressed_index(hs2, i)
    # bi[1] = transmon level (1-based), bi[2] = photon number (1-based)
    println("  dressed[$i] → |q=$(bi[1]-1), n=$(bi[2]-1)⟩, E = $(round(e, digits=4)) GHz")
end

# Dispersive shift: χ = ω_{|1,1⟩→|1,0⟩} - ω_{|0,1⟩→|0,0⟩}
try
    E00 = energy_by_bare_index(hs2, 1, 1)  # |g,0⟩
    E10 = energy_by_bare_index(hs2, 2, 1)  # |e,0⟩
    E01 = energy_by_bare_index(hs2, 1, 2)  # |g,1⟩
    E11 = energy_by_bare_index(hs2, 2, 2)  # |e,1⟩

    chi = (E11 - E10) - (E01 - E00)
    println("\nDispersive shift χ/2π = $(round(chi * 1000, digits=2)) MHz")
catch ex
    println("\nCould not compute dispersive shift: $ex")
end
