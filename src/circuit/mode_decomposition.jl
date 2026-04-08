# ──────────────────────────────────────────────────────────────────────────────
# Mode decomposition: variable transformation and classification
#
# Transforms node flux variables (φ_1, φ_2, ...) into normal mode variables
# (θ_1, θ_2, ...) and classifies each mode as periodic, extended, free, or frozen.
#
# Classification rules (per node in the identity-transformed basis):
#   periodic: connected to JJ but NOT to any inductor → compact phase, charge basis
#   extended: connected to inductor (L) → non-compact, HO or grid basis
#   frozen:   connected only to capacitors (C/CJ), no JJ or L → eliminated
#   free:     no capacitive coupling at all → conserved charge, eliminated
# ──────────────────────────────────────────────────────────────────────────────

"""
    VarCategories

Classification of transformed circuit variables into mode types.
Each field contains 1-based indices into the transformed variable vector.

- `periodic`: modes with compact (2π-periodic) phase variable — quantized in charge basis
- `extended`: non-compact modes — quantized in HO or discretized basis
- `free`: decoupled modes with conserved charge — eliminated
- `frozen`: modes determined by equilibrium conditions — substituted out
"""
struct VarCategories
    periodic::Vector{Int}
    extended::Vector{Int}
    free::Vector{Int}
    frozen::Vector{Int}
end

function Base.show(io::IO, vc::VarCategories)
    print(io, "VarCategories(periodic=$(vc.periodic), extended=$(vc.extended), ",
          "free=$(vc.free), frozen=$(vc.frozen))")
end

"""
    compute_variable_transformation(sc::SymbolicCircuit)

Compute the transformation matrix T and variable classification.

Returns `(T, var_categories)` where:
- `T` is an N×N matrix mapping node fluxes to mode variables: θ = T * φ
- `var_categories::VarCategories` classifies each θ_i
"""
function compute_variable_transformation(sc::SymbolicCircuit)
    n = sc.graph.num_nodes
    cg = sc.graph

    # Classify each node variable
    periodic_indices = Int[]
    extended_indices = Int[]
    free_indices = Int[]
    frozen_indices = Int[]

    for node in 1:n
        has_jj, has_inductor, has_capacitor = _node_connections(cg, node)

        if !has_capacitor && !has_jj
            push!(free_indices, node)
        elseif !has_jj && !has_inductor
            push!(frozen_indices, node)
        elseif has_inductor
            push!(extended_indices, node)
        else
            push!(periodic_indices, node)
        end
    end

    # Build transformation matrix
    if length(extended_indices) <= 1
        # Single or no extended mode: identity transform is correct
        T = Matrix{Float64}(I, n, n)
    else
        # Multi-extended: diagonalize LC subspace via generalized eigenvalue problem
        T = _build_normal_mode_transform(sc, periodic_indices, extended_indices, n)
    end

    categories = VarCategories(periodic_indices, extended_indices,
                               free_indices, frozen_indices)
    return T, categories
end

"""
Build the transformation matrix T that diagonalizes the LC (extended) subspace.

Solves the generalized eigenvalue problem `L_inv_ext * v = λ * C_ext * v`
for the extended-mode subblock. Periodic modes keep identity rows.
After transformation, `T' * L_inv * T` is diagonal for the extended block.
"""
function _build_normal_mode_transform(sc::SymbolicCircuit,
                                      periodic_indices::Vector{Int},
                                      extended_indices::Vector{Int},
                                      n::Int)
    C_float = Float64.(Symbolics.value.(sc.capacitance_matrix))
    L_inv_float = Float64.(Symbolics.value.(sc.inv_inductance_matrix))

    n_ext = length(extended_indices)
    C_ext = C_float[extended_indices, extended_indices]
    L_inv_ext = L_inv_float[extended_indices, extended_indices]

    # Generalized eigenvalue problem: L_inv * v = λ * C * v
    # λ_k = ω_k² (squared mode frequencies)
    F = eigen(Symmetric(L_inv_ext), Symmetric(C_ext))

    # Sort by eigenvalue (ascending frequency)
    perm = sortperm(F.values)
    T_ext = F.vectors[:, perm]

    # Mass-normalize: T_ext' * C_ext * T_ext = I
    for k in 1:n_ext
        norm_k = sqrt(abs(T_ext[:, k]' * C_ext * T_ext[:, k]))
        if norm_k > 1e-15
            T_ext[:, k] ./= norm_k
        end
    end

    # Build full T: identity for all, then fill extended block
    # Convention: L_inv_θ = T' * L_inv * T must be diagonal.
    # Since V' * L_inv * V = Λ (from generalized eigen), we need
    # the extended sub-block of T to be V (eigenvectors as columns).
    T = Matrix{Float64}(I, n, n)
    for (new_col, old_col) in enumerate(extended_indices)
        for (new_row, old_row) in enumerate(extended_indices)
            T[old_row, old_col] = T_ext[new_row, new_col]
        end
    end

    return T
end

"""Determine what types of branches connect to a given node."""
function _node_connections(cg::CircuitGraph, node::Int)
    has_jj = false
    has_inductor = false
    has_capacitor = false

    for b in cg.branches
        touches = (b.node_i == node || b.node_j == node)
        touches || continue
        if b.branch_type == JJ_branch
            has_jj = true
            has_capacitor = true  # JJ also has capacitance (EC parameter)
        elseif b.branch_type == L_branch
            has_inductor = true
        elseif b.branch_type in (C_branch, CJ_branch)
            has_capacitor = true
        end
    end

    return has_jj, has_inductor, has_capacitor
end

# ── Transform Hamiltonian ────────────────────────────────────────────────────

"""
    transform_hamiltonian(sc::SymbolicCircuit, T::Matrix{Float64}, vc::VarCategories)

Express the Hamiltonian in terms of transformed (mode) variables.
Returns the symbolic Hamiltonian in mode coordinates.
"""
function transform_hamiltonian(sc::SymbolicCircuit, T::Matrix{Float64},
                               vc::VarCategories)
    n = sc.graph.num_nodes
    T_inv = inv(T)

    # Transformed EC matrix: EC_θ = T^{-T} * EC * T^{-1}
    ec_transformed = T_inv' * Float64.(Symbolics.value.(sc.ec_matrix)) * T_inv

    # Mode charge variables
    mode_charges = [Symbolics.variable(:nθ, i) for i in 1:n]
    mode_phases = [Symbolics.variable(:θ, i) for i in 1:n]

    # Kinetic (charging) energy in mode basis
    H_charge = Num(0)
    for i in 1:n, j in 1:n
        abs(ec_transformed[i, j]) < 1e-15 && continue
        H_charge += 4 * ec_transformed[i, j] * mode_charges[i] * mode_charges[j]
    end

    # Inductive energy in mode basis
    L_inv_float = Float64.(Symbolics.value.(sc.inv_inductance_matrix))
    L_inv_transformed = T' * L_inv_float * T

    H_inductive = Num(0)
    for i in 1:n, j in 1:n
        abs(L_inv_transformed[i, j]) < 1e-15 && continue
        H_inductive += L_inv_transformed[i, j] * mode_phases[i] * mode_phases[j] / 2
    end

    # Josephson terms: substitute φ = T^{-1} θ
    H_josephson = Num(0)
    for (ej, phase_expr) in sc.josephson_terms
        subs = Dict(sc.node_vars[i] => sum(T_inv[i, j] * mode_phases[j] for j in 1:n)
                     for i in 1:n)
        new_phase = Symbolics.substitute(phase_expr, subs)
        H_josephson += -ej * cos(new_phase)
    end

    return Symbolics.simplify(H_charge + H_inductive + H_josephson)
end
