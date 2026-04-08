# ──────────────────────────────────────────────────────────────────────────────
# Data containers for eigenvalue/eigenvector results and parameter sweeps
# ──────────────────────────────────────────────────────────────────────────────

"""
    SpectrumData

Container for eigenvalues and eigenvectors computed across a parameter sweep.

# Fields
- `param_name::Symbol` — name of the swept parameter
- `param_vals::Vector{Float64}` — parameter values
- `eigenvalues::Matrix{Float64}` — eigenvalues[i, j] = j-th eigenvalue at param_vals[i]
- `eigenvectors` — optional eigenvectors; `nothing` if not stored
"""
struct SpectrumData
    param_name::Symbol
    param_vals::Vector{Float64}
    eigenvalues::Matrix{Float64}
    eigenvectors::Union{Nothing, Vector{Matrix{ComplexF64}}}
end

"""
    SpectrumLookup

Bare ↔ dressed state mapping for composite HilbertSpace systems.
Created by `generate_lookup!`.
"""
struct SpectrumLookup
    dressed_evals::Vector{Float64}
    dressed_evecs::Matrix{ComplexF64}
    bare_evals::Vector{Vector{Float64}}
    overlap_matrix::Matrix{Float64}
    bare_to_dressed::Dict{Tuple, Int}
    dressed_to_bare::Dict{Int, Tuple}
end
