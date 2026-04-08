# ──────────────────────────────────────────────────────────────────────────────
# Physical constants relevant for superconducting circuits
# All values in SI units unless otherwise noted.
# ──────────────────────────────────────────────────────────────────────────────

module PhysicalConstants

# Fundamental constants
const e     = 1.602176634e-19      # elementary charge [C]
const h     = 6.62607015e-34       # Planck constant [J·s]
const hbar  = h / (2π)             # reduced Planck constant [J·s]
const k_B   = 1.380649e-23         # Boltzmann constant [J/K]

# Superconducting circuit constants
const Phi_0 = h / (2e)             # magnetic flux quantum [Wb]
const phi_0 = Phi_0 / (2π)         # reduced flux quantum [Wb]

# Derived conversion factors
const GHz_to_eV  = h * 1e9 / e     # 1 GHz in eV
const eV_to_GHz  = 1.0 / GHz_to_eV
const GHz_to_K   = h * 1e9 / k_B   # 1 GHz in Kelvin
const K_to_GHz   = 1.0 / GHz_to_K

end # module PhysicalConstants
