# ──────────────────────────────────────────────────────────────────────────────
# Circuit graph data structures
#
# A superconducting circuit is represented as a directed graph where:
#   - Nodes are superconducting islands (node 0 = ground)
#   - Branches are circuit elements (C, L, JJ, CJ) connecting two nodes
# ──────────────────────────────────────────────────────────────────────────────

"""
Circuit branch element types.

- `C_branch`: Capacitor (parameter: EC)
- `L_branch`: Inductor (parameter: EL)
- `JJ_branch`: Josephson junction (parameters: EJ, EC)
- `CJ_branch`: Capacitive junction — junction capacitance without Josephson energy
"""
@enum BranchType C_branch L_branch JJ_branch CJ_branch

"""
    Branch(branch_type, node_i, node_j, parameters)

A circuit element connecting node `node_i` to node `node_j`.

# Fields
- `branch_type::BranchType` — element type
- `node_i::Int` — start node (0 = ground)
- `node_j::Int` — end node (0 = ground)
- `parameters::Dict{Symbol, Float64}` — element parameters (:EJ, :EC, :EL)
"""
struct Branch
    branch_type::BranchType
    node_i::Int
    node_j::Int
    parameters::Dict{Symbol, Float64}
end

function Base.show(io::IO, b::Branch)
    type_str = b.branch_type == C_branch ? "C" :
               b.branch_type == L_branch ? "L" :
               b.branch_type == JJ_branch ? "JJ" : "CJ"
    params = join(["$k=$v" for (k, v) in b.parameters], ", ")
    print(io, "Branch($type_str, $(b.node_i)→$(b.node_j), $params)")
end

"""
    CircuitGraph(branches; has_ground=true)

Graph representation of a superconducting circuit.

# Fields
- `branches::Vector{Branch}` — all circuit elements
- `num_nodes::Int` — number of nodes (excluding ground if present)
- `has_ground::Bool` — whether node 0 (ground) is part of the circuit
"""
struct CircuitGraph
    branches::Vector{Branch}
    num_nodes::Int
    has_ground::Bool
end

function CircuitGraph(branches::Vector{Branch}; has_ground::Bool=true)
    all_nodes = Set{Int}()
    for b in branches
        push!(all_nodes, b.node_i, b.node_j)
    end
    if has_ground
        delete!(all_nodes, 0)
    end
    num_nodes = isempty(all_nodes) ? 0 : maximum(all_nodes)
    return CircuitGraph(branches, num_nodes, has_ground)
end

"""Return the number of dynamic (non-ground) nodes."""
num_dynamic_nodes(g::CircuitGraph) = g.num_nodes

"""Check if any branch connects to ground (node 0)."""
is_grounded(g::CircuitGraph) = g.has_ground && any(b -> b.node_i == 0 || b.node_j == 0, g.branches)

"""Filter branches by type."""
function branches_of_type(g::CircuitGraph, btype::BranchType)
    return filter(b -> b.branch_type == btype, g.branches)
end

"""Get all unique non-ground node indices, sorted."""
function node_indices(g::CircuitGraph)
    nodes = Set{Int}()
    for b in g.branches
        b.node_i != 0 && push!(nodes, b.node_i)
        b.node_j != 0 && push!(nodes, b.node_j)
    end
    return sort!(collect(nodes))
end

"""
Convert CircuitGraph to a Graphs.jl SimpleGraph for topology analysis.
Returns `(graph, node_map)` where `node_map[i]` gives the original node index.
Ground node (0) is included if present.
"""
function to_simple_graph(cg::CircuitGraph)
    all_nodes = Set{Int}()
    for b in cg.branches
        push!(all_nodes, b.node_i, b.node_j)
    end
    sorted_nodes = sort!(collect(all_nodes))
    node_to_idx = Dict(n => i for (i, n) in enumerate(sorted_nodes))

    g = SimpleGraph(length(sorted_nodes))
    for b in cg.branches
        add_edge!(g, node_to_idx[b.node_i], node_to_idx[b.node_j])
    end
    return g, sorted_nodes
end
