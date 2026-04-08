# ──────────────────────────────────────────────────────────────────────────────
# Circuit topology analysis: spanning trees, loops, external flux assignment
#
# Uses Graphs.jl for graph algorithms. The spanning tree determines which
# branches are "tree" (dependent) vs "closure/cotree" (independent loops).
# ──────────────────────────────────────────────────────────────────────────────

"""
    find_spanning_tree(cg::CircuitGraph; prefer_capacitive=true)

Find a spanning tree of the circuit graph. Returns indices of branches in the tree.

If `prefer_capacitive=true`, capacitive branches (C, CJ) are preferred for the tree,
leaving inductive and junction branches as closure branches. This choice simplifies
the separation of kinetic (capacitive) and potential (inductive/JJ) terms.
"""
function find_spanning_tree(cg::CircuitGraph; prefer_capacitive::Bool=true)
    g, node_list = to_simple_graph(cg)
    n_vertices = nv(g)
    n_vertices == 0 && return Int[]

    # Map branches to edges (may have multi-edges)
    # Sort branches: capacitive first if preferred
    branch_order = collect(1:length(cg.branches))
    if prefer_capacitive
        sort!(branch_order, by = i -> _tree_priority(cg.branches[i]))
    end

    # Kruskal-like spanning tree: add branches greedily
    parent = collect(1:n_vertices)
    rank = zeros(Int, n_vertices)
    tree_indices = Int[]

    node_to_idx = Dict(n => i for (i, n) in enumerate(node_list))

    for bi in branch_order
        b = cg.branches[bi]
        u = node_to_idx[b.node_i]
        v = node_to_idx[b.node_j]
        ru = _find_root(parent, u)
        rv = _find_root(parent, v)
        if ru != rv
            push!(tree_indices, bi)
            _union!(parent, rank, ru, rv)
            length(tree_indices) == n_vertices - 1 && break
        end
    end

    return tree_indices
end

# Priority: lower = more preferred for tree
_tree_priority(b::Branch) =
    b.branch_type == C_branch  ? 0 :
    b.branch_type == CJ_branch ? 1 :
    b.branch_type == L_branch  ? 2 :
    3  # JJ_branch last (prefer as closure)

function _find_root(parent::Vector{Int}, x::Int)
    while parent[x] != x
        parent[x] = parent[parent[x]]  # path compression
        x = parent[x]
    end
    return x
end

function _union!(parent::Vector{Int}, rank::Vector{Int}, x::Int, y::Int)
    if rank[x] < rank[y]
        parent[x] = y
    elseif rank[x] > rank[y]
        parent[y] = x
    else
        parent[y] = x
        rank[x] += 1
    end
end

"""
    find_closure_branches(cg::CircuitGraph, tree_indices::Vector{Int})

Return indices of branches NOT in the spanning tree (= cotree / closure branches).
Each closure branch defines one fundamental loop.
"""
function find_closure_branches(cg::CircuitGraph, tree_indices::Vector{Int})
    tree_set = Set(tree_indices)
    return [i for i in 1:length(cg.branches) if !(i in tree_set)]
end

"""
    find_fundamental_loops(cg::CircuitGraph, tree_indices::Vector{Int})

For each closure branch, find the fundamental loop it creates with the spanning tree.
Returns a vector of loops, where each loop is a vector of `(branch_index, sign)` pairs.
The sign indicates direction: +1 if traversed in branch direction, -1 if reversed.
"""
function find_fundamental_loops(cg::CircuitGraph, tree_indices::Vector{Int})
    closure_indices = find_closure_branches(cg, tree_indices)
    tree_set = Set(tree_indices)

    # Build adjacency from tree branches only
    node_list = node_indices(cg)
    if cg.has_ground
        pushfirst!(node_list, 0)
    end
    node_to_idx = Dict(n => i for (i, n) in enumerate(node_list))
    n = length(node_list)

    # For each closure branch, find path in tree between its endpoints
    # Build tree adjacency with branch index tracking
    tree_adj = [Tuple{Int, Int, Int}[] for _ in 1:n]  # (neighbor_idx, branch_idx, sign)
    for bi in tree_indices
        b = cg.branches[bi]
        u = node_to_idx[b.node_i]
        v = node_to_idx[b.node_j]
        push!(tree_adj[u], (v, bi, +1))
        push!(tree_adj[v], (u, bi, -1))
    end

    loops = Vector{Vector{Tuple{Int, Int}}}()

    for ci in closure_indices
        cb = cg.branches[ci]
        start = node_to_idx[cb.node_i]
        target = node_to_idx[cb.node_j]

        # BFS to find path from start to target in tree
        path = _find_tree_path(tree_adj, start, target, n)
        if path !== nothing
            # Loop = closure branch + tree path (reversed direction)
            loop = Tuple{Int, Int}[(ci, +1)]
            append!(loop, path)
            push!(loops, loop)
        end
    end

    return loops
end

function _find_tree_path(adj, start::Int, target::Int, n::Int)
    visited = falses(n)
    parent = Vector{Tuple{Int, Int, Int}}(undef, n)  # (prev_node, branch_idx, sign)
    visited[start] = true
    queue = Int[start]

    while !isempty(queue)
        u = popfirst!(queue)
        u == target && break
        for (v, bi, sign) in adj[u]
            if !visited[v]
                visited[v] = true
                parent[v] = (u, bi, sign)
                push!(queue, v)
            end
        end
    end

    !visited[target] && return nothing

    # Reconstruct path from target back to start
    path = Tuple{Int, Int}[]
    node = target
    while node != start
        prev, bi, sign = parent[node]
        # Reverse direction: we go target → start, but branch was stored start → target
        push!(path, (bi, -sign))
        node = prev
    end
    reverse!(path)
    return path
end

"""
    loop_to_node_vector(cg::CircuitGraph, loop::Vector{Tuple{Int,Int}})

Convert a loop (list of (branch_index, sign) pairs) to a vector in node-flux space.
The returned vector has length `num_nodes` (excluding ground).
Entry `i` is the coefficient of node flux φ_i in the loop variable.
"""
function loop_to_node_vector(cg::CircuitGraph, loop::Vector{Tuple{Int, Int}})
    n = cg.num_nodes
    vec = zeros(Int, n)
    for (bi, sign) in loop
        b = cg.branches[bi]
        # Branch contributes: sign * (φ_{node_j} - φ_{node_i})
        # In node vector: +sign at node_j, -sign at node_i
        if b.node_j != 0
            vec[b.node_j] += sign
        end
        if b.node_i != 0
            vec[b.node_i] -= sign
        end
    end
    return vec
end
