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

    # Topology
    tree = find_spanning_tree(cg)
    closure = find_closure_branches(cg, tree)
    loops = find_fundamental_loops(cg, tree)

    # Symbolic node flux variables
    @variables t
    node_vars = Num[Symbolics.variable(:φ, i) for i in 1:n]
    node_dot_vars = Num[Symbolics.variable(:φ̇, i) for i in 1:n]

    # External fluxes: one per loop with inductive/JJ elements
    n_ext = _count_flux_loops(cg, loops)
    ext_fluxes = Num[Symbolics.variable(:Φext, i) for i in 1:n_ext]

    # Offset charges: one per node (for periodic variables)
    offset_charges = Num[Symbolics.variable(:ng, i) for i in 1:n]

    # Build matrices
    C_mat = _build_capacitance_matrix(cg, n)
    L_inv = _build_inv_inductance_matrix(cg, n)

    # Charging energy matrix: EC = (1/2) * C⁻¹
    # Use symbolic inversion for small matrices, numerical for larger
    C_inv = _symbolic_matrix_inverse(C_mat)
    EC_mat = C_inv ./ 2

    # Josephson terms
    jj_terms = _build_josephson_terms(cg, node_vars, ext_fluxes, loops)

    # Symbolic Hamiltonian: H = (1/2) Q^T C⁻¹ Q + (1/2) φ^T L⁻¹ φ - Σ EJ cos(...)
    # In terms of charge operators: H_charge = Σ_{ij} 4*EC[i,j] * n_i * n_j
    # (factor of 4 from convention: EC = e²/(2C), and n is Cooper pair number)
    charge_vars = [Symbolics.variable(:n, i) for i in 1:n]

    H_charge = sum(4 * EC_mat[i, j] * charge_vars[i] * charge_vars[j]
                   for i in 1:n, j in 1:n)

    H_inductive = Num(0)
    if !iszero(L_inv)
        H_inductive = sum(L_inv[i, j] * node_vars[i] * node_vars[j] / 2
                          for i in 1:n, j in 1:n if !iszero(L_inv[i, j]))
    end

    H_josephson = sum(-ej * cos(phase) for (ej, phase) in jj_terms; init=Num(0))

    H_symbolic = Symbolics.simplify(H_charge + H_inductive + H_josephson)

    return SymbolicCircuit(
        cg, tree, closure, loops,
        node_vars, node_dot_vars,
        C_mat, L_inv,
        H_symbolic, EC_mat,
        ext_fluxes, offset_charges,
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

# ── Josephson terms ──────────────────────────────────────────────────────────

function _build_josephson_terms(cg::CircuitGraph, node_vars::Vector{Num},
                                ext_fluxes::Vector{Num},
                                loops::Vector{Vector{Tuple{Int, Int}}})
    terms = Tuple{Num, Num}[]
    flux_idx = 0

    # Identify which loops contain JJ or L branches (carry external flux)
    loop_has_flux = _loops_with_flux(cg, loops)

    for b in cg.branches
        b.branch_type == JJ_branch || continue
        ej = Num(b.parameters[:EJ])

        # Phase difference across junction: φ_j - φ_i
        phase = _branch_phase(b, node_vars)

        # Add external flux if this junction is in a loop
        for (li, loop) in enumerate(loops)
            if loop_has_flux[li] && any(bi == findfirst(==(b), cg.branches) for (bi, _) in loop)
                flux_idx += 1
                if flux_idx <= length(ext_fluxes)
                    phase = phase - ext_fluxes[flux_idx]
                end
                break
            end
        end

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

function _count_flux_loops(cg::CircuitGraph, loops)
    count = 0
    for loop in loops
        for (bi, _) in loop
            b = cg.branches[bi]
            if b.branch_type in (L_branch, JJ_branch)
                count += 1
                break
            end
        end
    end
    return count
end

function _loops_with_flux(cg::CircuitGraph, loops)
    return [any(cg.branches[bi].branch_type in (L_branch, JJ_branch)
                for (bi, _) in loop)
            for loop in loops]
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
