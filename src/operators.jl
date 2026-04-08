# ──────────────────────────────────────────────────────────────────────────────
# Operator helper functions bridging to QuantumToolbox.jl
# ──────────────────────────────────────────────────────────────────────────────

"""
    identity_wrap(op, subsys_index::Int, dims::Vector{Int})

Embed a subsystem operator `op` (acting on subsystem `subsys_index`) into the
full tensor-product Hilbert space defined by `dims`.

Returns `I ⊗ ... ⊗ op ⊗ ... ⊗ I` as a QuantumToolbox `QuantumObject`.

# Arguments
- `op` — operator matrix or `QuantumObject` for the target subsystem
- `subsys_index` — 1-based index of the subsystem in `dims`
- `dims` — vector of Hilbert space dimensions `[d1, d2, ..., dN]`
"""
function identity_wrap(op, subsys_index::Int, dims::Vector{Int})
    n_subsys = length(dims)
    1 <= subsys_index <= n_subsys || throw(BoundsError(dims, subsys_index))

    ops = Vector{QuantumObject}(undef, n_subsys)
    for i in 1:n_subsys
        if i == subsys_index
            ops[i] = _ensure_qobj(op, dims[i])
        else
            ops[i] = qeye(dims[i])
        end
    end

    return reduce(kron, ops)
end

"""Convert a raw matrix to QuantumObject if needed."""
function _ensure_qobj(op::QuantumObject, expected_dim::Int)
    return op
end

function _ensure_qobj(op::AbstractMatrix, expected_dim::Int)
    size(op, 1) == expected_dim || throw(DimensionMismatch(
        "operator size $(size(op, 1)) doesn't match expected dimension $expected_dim"))
    return QuantumObject(op)
end
