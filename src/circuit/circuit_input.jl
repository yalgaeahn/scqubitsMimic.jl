# ──────────────────────────────────────────────────────────────────────────────
# Circuit input parsing
#
# Parses a YAML-style string description into a CircuitGraph.
# Compatible with scqubits circuit specification format.
# ──────────────────────────────────────────────────────────────────────────────

const BRANCH_TYPE_MAP = Dict(
    "C"  => C_branch,
    "L"  => L_branch,
    "JJ" => JJ_branch,
    "CJ" => CJ_branch,
)

"""
    parse_circuit(description::String) -> CircuitGraph

Parse a YAML-style circuit description string into a `CircuitGraph`.

# Format
```
branches:
  - [TYPE, node_i, node_j, param1=val1, param2=val2]
  ...
```

Where TYPE is one of: C, L, JJ, CJ.

# Example
```julia
cg = parse_circuit(\"\"\"
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [C,  0, 1, EC=0.5]
\"\"\")
```
"""
function parse_circuit(description::String)
    branches = Branch[]
    in_branches = false

    for raw_line in eachline(IOBuffer(description))
        line = strip(raw_line)
        isempty(line) && continue
        startswith(line, "#") && continue

        if startswith(line, "branches:")
            in_branches = true
            continue
        end

        if in_branches && startswith(line, "- [")
            branch = _parse_branch_line(line)
            push!(branches, branch)
        end
    end

    isempty(branches) && throw(ArgumentError("No branches found in circuit description"))

    has_ground = any(b -> b.node_i == 0 || b.node_j == 0, branches)
    return CircuitGraph(branches; has_ground=has_ground)
end

function _parse_branch_line(line::AbstractString)
    # Extract content between [ and ]
    m = match(r"\[(.+)\]", line)
    m === nothing && throw(ArgumentError("Invalid branch line: $line"))
    content = strip(m.captures[1])

    parts = [strip(p) for p in split(content, ",")]
    length(parts) >= 3 || throw(ArgumentError("Branch needs at least type, node_i, node_j: $line"))

    type_str = uppercase(parts[1])
    haskey(BRANCH_TYPE_MAP, type_str) || throw(ArgumentError("Unknown branch type: $type_str"))
    btype = BRANCH_TYPE_MAP[type_str]

    node_i = parse(Int, parts[2])
    node_j = parse(Int, parts[3])

    params = Dict{Symbol, Float64}()
    for i in 4:length(parts)
        _parse_param!(params, parts[i])
    end

    _validate_branch_params(btype, params, line)

    return Branch(btype, node_i, node_j, params)
end

function _parse_param!(params::Dict{Symbol, Float64}, s::AbstractString)
    m = match(r"(\w+)\s*=\s*([\d.eE+\-]+)", s)
    m === nothing && throw(ArgumentError("Invalid parameter: $s"))
    params[Symbol(m.captures[1])] = parse(Float64, m.captures[2])
end

function _validate_branch_params(btype::BranchType, params::Dict{Symbol, Float64}, line::AbstractString)
    if btype == C_branch
        haskey(params, :EC) || throw(ArgumentError("C branch requires EC: $line"))
    elseif btype == L_branch
        haskey(params, :EL) || throw(ArgumentError("L branch requires EL: $line"))
    elseif btype == JJ_branch
        haskey(params, :EJ) || throw(ArgumentError("JJ branch requires EJ: $line"))
        haskey(params, :EC) || throw(ArgumentError("JJ branch requires EC: $line"))
    elseif btype == CJ_branch
        haskey(params, :EC) || throw(ArgumentError("CJ branch requires EC: $line"))
    end
end
