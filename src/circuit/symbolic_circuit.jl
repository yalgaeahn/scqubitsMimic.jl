# ──────────────────────────────────────────────────────────────────────────────
# Symbolic circuit analysis
#
# Constructs capacitance/inductance matrices and derives symbolic
# Lagrangian and Hamiltonian using Symbolics.jl.
# ──────────────────────────────────────────────────────────────────────────────

"""
    SymbolicCircuit

Holds the symbolic analysis results of a superconducting circuit:
capacitance/inductance matrices, variable transformation, and
symbolic Lagrangian/Hamiltonian.
"""
struct SymbolicCircuit
    graph::CircuitGraph
    spanning_tree::Vector{Int}
    closure_branches::Vector{Int}
    loops::Vector{Vector{Tuple{Int, Int}}}
    superconducting_closure_branches::Vector{Int}
    superconducting_loops::Vector{Vector{Tuple{Int, Int}}}

    # Symbolic node variables
    node_vars::Vector{Num}         # φ_1, φ_2, ... (node fluxes)
    node_dot_vars::Vector{Num}     # φ̇_1, φ̇_2, ...

    # Matrices in node basis (N_dynamic × N_dynamic)
    capacitance_matrix::Matrix{Num}
    inv_inductance_matrix::Matrix{Num}

    # Symbolic Hamiltonian pieces
    hamiltonian_symbolic::Num

    # Charging energy matrix: EC[i,j] = (1/2) (C⁻¹)[i,j]
    ec_matrix::Matrix{Num}

    # External fluxes and offset charges
    external_fluxes::Vector{Num}
    offset_charges::Vector{Num}

    # Per-branch external flux allocation (length = n_branches)
    # Each entry is the symbolic external flux for that branch (0 for non-closure branches)
    branch_flux_allocations::Vector{Num}

    # Josephson terms (symbolic): list of (EJ_value, phase_expression)
    josephson_terms::Vector{Tuple{Num, Num}}
end

"""
    build_symbolic_circuit(cg::CircuitGraph) -> SymbolicCircuit

Perform full symbolic analysis of a circuit graph:
1. Find spanning tree and loops
2. Build capacitance and inverse-inductance matrices
3. Derive symbolic Hamiltonian

All energies are in units of GHz.
"""
function build_symbolic_circuit(cg::CircuitGraph)
    n = cg.num_nodes
    n > 0 || throw(ArgumentError("Circuit must have at least one non-ground node"))

    # Topology on full graph (needed for capacitance matrix, mode decomposition, etc.)
    tree = find_spanning_tree(cg)
    closure = find_closure_branches(cg, tree)
    loops = find_fundamental_loops(cg, tree)

    # Superconducting loops: exclude capacitive branches to find physical flux loops
    sc_closure_indices, sc_loops = find_superconducting_loops(cg)
    n_ext = length(sc_closure_indices)

    # Symbolic node flux variables
    @variables t
    node_vars = Num[Symbolics.variable(:φ, i) for i in 1:n]
    node_dot_vars = Num[Symbolics.variable(:φ̇, i) for i in 1:n]

    # scqubits-style symbolic names: Φ1, Φ2, ... and ng1, ng2, ...
    ext_fluxes = Num[Num(Symbolics.variable(Symbol("Φ$(i)"))) for i in 1:n_ext]

    # Offset charges: one per node (for periodic variables)
    offset_charges = Num[Num(Symbolics.variable(Symbol("ng$(i)"))) for i in 1:n]

    # Branch flux allocations: per-branch symbolic flux
    branch_flux_alloc = _build_branch_flux_allocations(cg, sc_closure_indices, ext_fluxes)

    # Build matrices
    C_mat = _build_capacitance_matrix(cg, n)
    L_inv = _build_inv_inductance_matrix(cg, n)

    # Charging energy matrix: EC = (1/2) * C⁻¹
    C_inv = _symbolic_matrix_inverse(C_mat)
    EC_mat = C_inv ./ 2

    # Josephson terms: use branch flux allocations (no loop search)
    jj_terms = _build_josephson_terms(cg, node_vars, branch_flux_alloc)

    # Symbolic Hamiltonian: H = (1/2) Q^T C⁻¹ Q + Σ (EL/2)(branch_flux)² - Σ EJ cos(...)
    charge_vars = [Symbolics.variable(:n, i) for i in 1:n]

    H_charge = sum(4 * EC_mat[i, j] * charge_vars[i] * charge_vars[j]
                   for i in 1:n, j in 1:n)

    # Inductive energy: branch-level with external flux
    H_inductive = _build_inductive_terms(cg, node_vars, branch_flux_alloc)

    H_josephson = sum(-ej * cos(phase) for (ej, phase) in jj_terms; init=Num(0))

    H_symbolic = Symbolics.simplify(H_charge + H_inductive + H_josephson)

    return SymbolicCircuit(
        cg, tree, closure, loops, sc_closure_indices, sc_loops,
        node_vars, node_dot_vars,
        C_mat, L_inv,
        H_symbolic, EC_mat,
        ext_fluxes, offset_charges,
        branch_flux_alloc,
        jj_terms
    )
end

# ── Matrix construction ──────────────────────────────────────────────────────

function _build_capacitance_matrix(cg::CircuitGraph, n::Int)
    C = zeros(Num, n, n)

    for b in cg.branches
        ec = get(b.parameters, :EC, nothing)
        ec === nothing && continue
        # EC = e²/(2C) => C = 1/(8*EC) in our units
        # But we work in the EC basis directly. The capacitance matrix
        # stores 1/(8*EC) for dimensional consistency.
        cap = 1.0 / (8.0 * ec)

        i, j = b.node_i, b.node_j
        if i != 0 && j != 0
            C[i, i] += cap
            C[j, j] += cap
            C[i, j] -= cap
            C[j, i] -= cap
        elseif i == 0 && j != 0
            C[j, j] += cap
        elseif j == 0 && i != 0
            C[i, i] += cap
        end
    end

    return C
end

function _build_inv_inductance_matrix(cg::CircuitGraph, n::Int)
    L_inv = zeros(Num, n, n)

    for b in cg.branches
        b.branch_type == L_branch || continue
        el = b.parameters[:EL]

        i, j = b.node_i, b.node_j
        if i != 0 && j != 0
            L_inv[i, i] += el
            L_inv[j, j] += el
            L_inv[i, j] -= el
            L_inv[j, i] -= el
        elseif i == 0 && j != 0
            L_inv[j, j] += el
        elseif j == 0 && i != 0
            L_inv[i, i] += el
        end
    end

    return L_inv
end

# ── Branch flux allocations ──────────────────────────────────────────────────

"""
    _build_branch_flux_allocations(cg, sc_closure_indices, ext_fluxes)

Compute per-branch external flux allocation. Each closure branch in the
superconducting subgraph receives its corresponding `Φ_k`; all other
branches receive 0.

Returns a `Vector{Num}` of length `length(cg.branches)`.
"""
function _build_branch_flux_allocations(cg::CircuitGraph,
                                         sc_closure_indices::Vector{Int},
                                         ext_fluxes::Vector{Num})
    n_branches = length(cg.branches)
    alloc = fill(Num(0), n_branches)
    for (k, ci) in enumerate(sc_closure_indices)
        alloc[ci] = ext_fluxes[k]
    end
    return alloc
end

# ── Inductive terms ──────────────────────────────────────────────────────────

"""
    _build_inductive_terms(cg, node_vars, branch_flux_alloc)

Build inductive energy as a branch-level sum: `Σ (EL/2) * (φ_j - φ_i + Φext)²`.
This correctly incorporates external flux into inductor branches, unlike the
node-basis quadratic form `(1/2) φ^T L_inv φ` which only captures the
flux-independent part.
"""
function _build_inductive_terms(cg::CircuitGraph, node_vars::Vector{Num},
                                 branch_flux_alloc::Vector{Num})
    H = Num(0)
    for (bi, b) in enumerate(cg.branches)
        b.branch_type == L_branch || continue
        el = b.parameters[:EL]
        branch_flux = _branch_phase(b, node_vars) + branch_flux_alloc[bi]
        H += el / 2 * branch_flux^2
    end
    return H
end

# ── Josephson terms ──────────────────────────────────────────────────────────

function _build_josephson_terms(cg::CircuitGraph, node_vars::Vector{Num},
                                branch_flux_alloc::Vector{Num})
    terms = Tuple{Num, Num}[]
    for (bi, b) in enumerate(cg.branches)
        b.branch_type == JJ_branch || continue
        ej = Num(b.parameters[:EJ])
        phase = _branch_phase(b, node_vars) + branch_flux_alloc[bi]
        push!(terms, (ej, phase))
    end
    return terms
end

function _branch_phase(b::Branch, node_vars::Vector{Num})
    phase = Num(0)
    if b.node_j != 0
        phase += node_vars[b.node_j]
    end
    if b.node_i != 0
        phase -= node_vars[b.node_i]
    end
    return phase
end


# ── Symbolic matrix inverse ──────────────────────────────────────────────────

function _symbolic_matrix_inverse(M::Matrix{Num})
    n = size(M, 1)
    if n == 1
        return reshape([1 / M[1, 1]], 1, 1)
    end

    # For small matrices, use explicit formulas; for larger, numerical at evaluation
    if n == 2
        det = M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]
        return [M[2, 2] -M[1, 2]; -M[2, 1] M[1, 1]] ./ det
    end

    # General case: convert to Float64 if all entries are numeric
    try
        M_float = Float64.(Symbolics.value.(M))
        M_inv = inv(M_float)
        return Num.(M_inv)
    catch
        # Truly symbolic — use cofactor expansion (expensive for large n)
        return _cofactor_inverse(M)
    end
end

function _cofactor_inverse(M::Matrix{Num})
    n = size(M, 1)
    cofactors = zeros(Num, n, n)
    det_M = _symbolic_det(M)

    for i in 1:n, j in 1:n
        minor = M[setdiff(1:n, i), setdiff(1:n, j)]
        cofactors[j, i] = (-1)^(i + j) * _symbolic_det(minor)
    end

    return cofactors ./ det_M
end

function _symbolic_det(M::Matrix{Num})
    n = size(M, 1)
    n == 1 && return M[1, 1]
    n == 2 && return M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]

    det = Num(0)
    for j in 1:n
        minor = M[2:end, setdiff(1:n, j)]
        det += (-1)^(j + 1) * M[1, j] * _symbolic_det(minor)
    end
    return det
end
