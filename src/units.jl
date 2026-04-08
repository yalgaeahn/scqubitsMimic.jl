# ──────────────────────────────────────────────────────────────────────────────
# Unit conversion utilities for energy scales commonly used in circuit QED.
# Default energy unit throughout the package is GHz.
# ──────────────────────────────────────────────────────────────────────────────

"""
    convert_units(value, from::Symbol, to::Symbol)

Convert `value` between energy units.
Supported units: `:GHz`, `:MHz`, `:eV`, `:meV`, `:K`, `:J`.

# Example
```julia
convert_units(5.0, :GHz, :K)   # ≈ 0.24 K
```
"""
function convert_units(value::Real, from::Symbol, to::Symbol)
    from == to && return Float64(value)
    ghz = _to_ghz(value, from)
    return _from_ghz(ghz, to)
end

function _to_ghz(value::Real, unit::Symbol)
    unit == :GHz  && return Float64(value)
    unit == :MHz  && return value * 1e-3
    unit == :eV   && return value * PhysicalConstants.eV_to_GHz
    unit == :meV  && return value * 1e-3 * PhysicalConstants.eV_to_GHz
    unit == :K    && return value * PhysicalConstants.K_to_GHz
    unit == :J    && return value / (PhysicalConstants.h * 1e9)
    throw(ArgumentError("Unknown energy unit: $unit"))
end

function _from_ghz(ghz::Float64, unit::Symbol)
    unit == :GHz  && return ghz
    unit == :MHz  && return ghz * 1e3
    unit == :eV   && return ghz * PhysicalConstants.GHz_to_eV
    unit == :meV  && return ghz * PhysicalConstants.GHz_to_eV * 1e3
    unit == :K    && return ghz * PhysicalConstants.GHz_to_K
    unit == :J    && return ghz * PhysicalConstants.h * 1e9
    throw(ArgumentError("Unknown energy unit: $unit"))
end
