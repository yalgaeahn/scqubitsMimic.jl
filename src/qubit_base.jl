# ──────────────────────────────────────────────────────────────────────────────
# Default method implementations for AbstractQuantumSystem
#
# Concrete types must implement:
#   hamiltonian(sys) → QuantumObject{Operator}
#   hilbertdim(sys)  → Int
# All other methods have default implementations dispatching on hamiltonian().
# ──────────────────────────────────────────────────────────────────────────────

"""
    hamiltonian(sys::AbstractQuantumSystem)

Return the Hamiltonian as a QuantumToolbox `QuantumObject`.
Must be implemented by each concrete type.
"""
function hamiltonian end

"""
    hilbertdim(sys::AbstractQuantumSystem)

Return the Hilbert space dimension for this system.
Must be implemented by each concrete type.
"""
function hilbertdim end

"""
    eigenvals(sys::AbstractQuantumSystem; evals_count=6)

Compute the lowest `evals_count` eigenvalues of the system Hamiltonian.
"""
function eigenvals(sys::AbstractQuantumSystem; evals_count::Int=6)
    H = hamiltonian(sys)
    vals = eigenenergies(H)
    n = min(evals_count, length(vals))
    return real.(vals[1:n])
end

"""
    eigensys(sys::AbstractQuantumSystem; evals_count=6)

Compute eigenvalues and eigenvectors. Returns `(eigenvalues, eigenvectors)`.
"""
function eigensys(sys::AbstractQuantumSystem; evals_count::Int=6)
    H = hamiltonian(sys)
    dim = hilbertdim(sys)

    result = eigenstates(H)
    n = min(evals_count, length(result.values))
    vals = real.(result.values[1:n])
    vecs = result.vectors[:, 1:n]
    return vals, vecs
end

"""
    matrixelement(sys::AbstractQuantumSystem, op, i::Int, j::Int)

Compute the matrix element ⟨ψ_i|op|ψ_j⟩ where ψ_i, ψ_j are eigenstates.
Indices `i`, `j` are 1-based.
"""
function matrixelement(sys::AbstractQuantumSystem, op, i::Int, j::Int;
                       evals_count::Int=max(i, j))
    _, vecs = eigensys(sys; evals_count=evals_count)
    op_mat = op isa QuantumObject ? op.data : op
    return dot(vecs[:, i], op_mat, vecs[:, j])
end

"""
    matrixelement_table(sys::AbstractQuantumSystem, op; evals_count=6)

Compute matrix elements ⟨ψ_i|op|ψ_j⟩ for all pairs (i, j) up to `evals_count`.
Returns a complex matrix.
"""
function matrixelement_table(sys::AbstractQuantumSystem, op; evals_count::Int=6)
    _, vecs = eigensys(sys; evals_count=evals_count)
    n = size(vecs, 2)
    op_mat = op isa QuantumObject ? op.data : op
    table = zeros(ComplexF64, n, n)
    for j in 1:n, i in 1:n
        table[i, j] = dot(vecs[:, i], op_mat, vecs[:, j])
    end
    return table
end

"""
    set_param!(sys::AbstractQuantumSystem, param_name::Symbol, val)

Set a parameter on a quantum system. Default uses `setfield!`.
Override for types where parameters are stored differently (e.g., Circuit).
"""
function set_param!(sys::AbstractQuantumSystem, param_name::Symbol, val)
    setfield!(sys, param_name, val)
end

"""
    get_param(sys::AbstractQuantumSystem, param_name::Symbol)

Get a parameter from a quantum system. Default uses `getfield`.
"""
function get_param(sys::AbstractQuantumSystem, param_name::Symbol)
    getfield(sys, param_name)
end

"""
    get_spectrum_vs_paramvals(sys, param_name, param_vals; evals_count=6, store_eigvecs=false)

Sweep a parameter and compute eigenvalues at each value. Returns `SpectrumData`.
Uses `set_param!`/`get_param` dispatch — override these for custom parameter handling.
"""
function get_spectrum_vs_paramvals(sys::AbstractQuantumSystem,
                                  param_name::Symbol,
                                  param_vals::AbstractVector;
                                  evals_count::Int=6,
                                  store_eigvecs::Bool=false)
    nvals = length(param_vals)
    eigenvalues = Matrix{Float64}(undef, nvals, evals_count)
    eigvec_store = store_eigvecs ? Vector{Matrix{ComplexF64}}(undef, nvals) : nothing

    original_val = get_param(sys, param_name)

    for (idx, val) in enumerate(param_vals)
        set_param!(sys, param_name, val)
        if store_eigvecs
            vals, vecs = eigensys(sys; evals_count=evals_count)
            eigenvalues[idx, :] .= vals
            eigvec_store[idx] = vecs
        else
            eigenvalues[idx, :] .= eigenvals(sys; evals_count=evals_count)
        end
    end

    set_param!(sys, param_name, original_val)

    return SpectrumData(param_name, collect(Float64, param_vals),
                        eigenvalues, eigvec_store)
end
