# ──────────────────────────────────────────────────────────────────────────────
# Internal diagonalization helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    _lowest_hermitian_eigenvalues(H, evals_count)

Return the lowest algebraic eigenvalues of a Hermitian operator in ascending order.
"""
function _lowest_hermitian_eigenvalues(H::QuantumObject, evals_count::Int)
    dense_H = Hermitian(Matrix(H.data))
    return _lowest_hermitian_eigenvalues(dense_H, evals_count)
end

function _lowest_hermitian_eigenvalues(H::Union{Hermitian, SymTridiagonal}, evals_count::Int)
    n = clamp(evals_count, 0, size(H, 1))
    n == 0 && return Float64[]
    return Float64.(eigvals(H, 1:n))
end

"""
    _lowest_hermitian_eigensystem(H, evals_count)

Return the lowest algebraic eigenpairs of a Hermitian operator in ascending order.
Eigenvectors are returned column-wise as a `Matrix{ComplexF64}` for compatibility
with the existing public APIs and spectrum containers.
"""
function _lowest_hermitian_eigensystem(H::QuantumObject, evals_count::Int)
    dense_H = Hermitian(Matrix(H.data))
    return _lowest_hermitian_eigensystem(dense_H, evals_count)
end

function _lowest_hermitian_eigensystem(H::Union{Hermitian, SymTridiagonal}, evals_count::Int)
    n = clamp(evals_count, 0, size(H, 1))
    if n == 0
        return Float64[], Matrix{ComplexF64}(undef, size(H, 1), 0)
    end

    result = eigen(H, 1:n)
    vals = Float64.(result.values)
    vecs = ComplexF64.(result.vectors)
    return vals, vecs
end

"""
    _transmon_charge_basis_tridiagonal(EJ, EC, ng, ncut)

Construct the transmon Hamiltonian in charge basis as a real symmetric tridiagonal
matrix for indexed low-energy diagonalization.
"""
function _transmon_charge_basis_tridiagonal(EJ::Real, EC::Real, ng::Real, ncut::Int)
    dim = 2 * ncut + 1
    diagonal = 4.0 .* Float64(EC) .* (collect(-ncut:ncut) .- Float64(ng)) .^ 2
    off_diagonal = fill(-Float64(EJ) / 2.0, dim - 1)
    return SymTridiagonal(diagonal, off_diagonal)
end

# TODO: add a sparse iterative backend with a fixed start vector, degeneracy
# re-orthogonalization, and shift-invert policy for closer scqubits parity.
