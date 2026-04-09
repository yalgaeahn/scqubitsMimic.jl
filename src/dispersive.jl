# ──────────────────────────────────────────────────────────────────────────────
# Dispersive shift / Kerr extraction from dressed spectrum
#
# Post-processing on SpectrumLookup to extract χ (cross-Kerr),
# self-Kerr (anharmonicity), and Lamb shifts.
# ──────────────────────────────────────────────────────────────────────────────

"""
    chi_matrix(hs::HilbertSpace) -> Matrix{Float64}

Cross-Kerr (dispersive shift) matrix between all pairs of subsystems.

    χ_ij = E(1_i,1_j) - E(1_i,0_j) - E(0_i,1_j) + E(0_i,0_j)

where E(n_i, n_j) denotes the dressed energy with n_i excitations in subsystem i
and n_j in subsystem j (all other subsystems in ground state).

Bare labels are 1-based: ground = 1, first excited = 2, etc.

Requires `generate_lookup!(hs)` to have been called first.
"""
function chi_matrix(hs::HilbertSpace)
    hs.lookup === nothing && error("Call generate_lookup!(hs) first")
    return _chi_from_lookup(hs.lookup, length(hs.subsystems))
end

"""
    chi_matrix(sweep::HilbertSpaceSweep) -> Array{Float64, 3}

Chi matrix at each sweep point. Returns array of size `(n_points, n_sub, n_sub)`.
Requires the sweep to have been created with `store_lookups=true`.
The result depends only on the lookups stored on the sweep itself; if strong
hybridization requires relaxed bare-label tracking, construct the sweep with
`ignore_low_overlap=true`.
"""
function chi_matrix(sweep::HilbertSpaceSweep)
    sweep.lookups === nothing &&
        error("Sweep must be created with store_lookups=true to compute chi_matrix")
    n_sub = length(sweep.hilbertspace.subsystems)
    n_points = size(sweep.dressed_evals, 1)
    result = Array{Float64, 3}(undef, n_points, n_sub, n_sub)
    for k in 1:n_points
        result[k, :, :] .= _chi_from_lookup(sweep.lookups[k], n_sub)
    end
    return result
end

function _chi_from_lookup(lookup::SpectrumLookup, n_sub::Int)
    chi = zeros(Float64, n_sub, n_sub)
    ground = ones(Int, n_sub)  # all subsystems in ground state (1-based)

    for i in 1:n_sub
        for j in i:n_sub
            if i == j
                # Diagonal: self-Kerr = E(3) - 2*E(2) + E(1)
                chi[i, i] = _self_kerr_from_lookup(lookup, i, n_sub)
            else
                # Off-diagonal: cross-Kerr
                label_00 = copy(ground)
                label_10 = copy(ground); label_10[i] = 2
                label_01 = copy(ground); label_01[j] = 2
                label_11 = copy(ground); label_11[i] = 2; label_11[j] = 2

                E00 = _energy_from_lookup(lookup, Tuple(label_00))
                E10 = _energy_from_lookup(lookup, Tuple(label_10))
                E01 = _energy_from_lookup(lookup, Tuple(label_01))
                E11 = _energy_from_lookup(lookup, Tuple(label_11))

                chi[i, j] = E11 - E10 - E01 + E00
                chi[j, i] = chi[i, j]
            end
        end
    end
    return chi
end

"""
    self_kerr(hs::HilbertSpace, subsys_idx::Int) -> Float64

Self-Kerr (anharmonicity) of subsystem `subsys_idx` in the dressed basis:

    K = E(2) - 2·E(1) + E(0)

where E(n) is the dressed energy with n excitations in `subsys_idx`,
all other subsystems in ground state. Bare labels are 1-based.

Requires `generate_lookup!(hs)` to have been called first.
"""
function self_kerr(hs::HilbertSpace, subsys_idx::Int)
    hs.lookup === nothing && error("Call generate_lookup!(hs) first")
    return _self_kerr_from_lookup(hs.lookup, subsys_idx, length(hs.subsystems))
end

"""
    self_kerr(sweep::HilbertSpaceSweep, subsys_idx::Int) -> Vector{Float64}

Self-Kerr at each sweep point. Requires `store_lookups=true`.
Uses the sweep-stored lookup policy, not `sweep.hilbertspace.ignore_low_overlap`.
"""
function self_kerr(sweep::HilbertSpaceSweep, subsys_idx::Int)
    sweep.lookups === nothing &&
        error("Sweep must be created with store_lookups=true to compute self_kerr")
    n_sub = length(sweep.hilbertspace.subsystems)
    n_points = size(sweep.dressed_evals, 1)
    return [_self_kerr_from_lookup(sweep.lookups[k], subsys_idx, n_sub)
            for k in 1:n_points]
end

function _self_kerr_from_lookup(lookup::SpectrumLookup, subsys_idx::Int, n_sub::Int)
    ground = ones(Int, n_sub)

    label_0 = copy(ground)                       # ground: all 1's
    label_1 = copy(ground); label_1[subsys_idx] = 2  # first excited
    label_2 = copy(ground); label_2[subsys_idx] = 3  # second excited

    E0 = _energy_from_lookup(lookup, Tuple(label_0))
    E1 = _energy_from_lookup(lookup, Tuple(label_1))
    E2 = _energy_from_lookup(lookup, Tuple(label_2))

    return E2 - 2 * E1 + E0
end

"""
    lamb_shift(hs::HilbertSpace, subsys_idx::Int) -> Float64

Lamb shift of subsystem `subsys_idx`: difference between dressed and bare
transition frequency (0→1).

    Δω = (E_dressed(1) - E_dressed(0)) - (E_bare(1) - E_bare(0))

Requires `generate_lookup!(hs)` to have been called first.
"""
function lamb_shift(hs::HilbertSpace, subsys_idx::Int)
    hs.lookup === nothing && error("Call generate_lookup!(hs) first")
    n_sub = length(hs.subsystems)
    lookup = hs.lookup

    ground = ones(Int, n_sub)
    label_0 = copy(ground)
    label_1 = copy(ground); label_1[subsys_idx] = 2

    dressed_01 = _energy_from_lookup(lookup, Tuple(label_1)) -
                 _energy_from_lookup(lookup, Tuple(label_0))

    bare_evals = lookup.bare_evals[subsys_idx]
    bare_01 = bare_evals[2] - bare_evals[1]

    return dressed_01 - bare_01
end

"""
    lamb_shift(sweep::HilbertSpaceSweep, subsys_idx::Int) -> Vector{Float64}

Lamb shift at each sweep point. Requires `store_lookups=true`.
Uses the sweep-stored lookup policy, not `sweep.hilbertspace.ignore_low_overlap`.
"""
function lamb_shift(sweep::HilbertSpaceSweep, subsys_idx::Int)
    sweep.lookups === nothing &&
        error("Sweep must be created with store_lookups=true to compute lamb_shift")
    n_sub = length(sweep.hilbertspace.subsystems)
    n_points = size(sweep.dressed_evals, 1)
    result = Vector{Float64}(undef, n_points)
    for k in 1:n_points
        lookup = sweep.lookups[k]
        ground = ones(Int, n_sub)
        label_0 = copy(ground)
        label_1 = copy(ground); label_1[subsys_idx] = 2

        dressed_01 = _energy_from_lookup(lookup, Tuple(label_1)) -
                     _energy_from_lookup(lookup, Tuple(label_0))
        bare_01 = lookup.bare_evals[subsys_idx][2] - lookup.bare_evals[subsys_idx][1]
        result[k] = dressed_01 - bare_01
    end
    return result
end

# ── Internal helper ─────────────────────────────────────────────────────────

function _energy_from_lookup(lookup::SpectrumLookup, bare_label::Tuple)
    idx = get(lookup.bare_to_dressed, bare_label, nothing)
    idx === nothing && error(
        "Bare state $bare_label not found in lookup. " *
        "Increase evals_count or regenerate the relevant lookup with " *
        "ignore_low_overlap=true.")
    return lookup.dressed_evals[idx]
end
