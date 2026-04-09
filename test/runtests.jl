using Test
using LinearAlgebra
using SparseArrays
using ScQubitsMimic

@testset "ScQubitsMimic.jl" begin

    @testset "Grid1d" begin
        g = Grid1d(-π, π, 101)
        @test g.npoints == 101
        pts = collect(grid_points(g))
        @test length(pts) == 101
        @test pts[1] ≈ -π
        @test pts[end] ≈ π
        @test grid_spacing(g) ≈ 2π / 100
    end

    @testset "Unit conversion" begin
        val = 5.0
        @test convert_units(convert_units(val, :GHz, :K), :K, :GHz) ≈ val rtol=1e-10
        @test convert_units(convert_units(val, :GHz, :eV), :eV, :GHz) ≈ val rtol=1e-10
        @test convert_units(val, :GHz, :GHz) == val
    end

    @testset "GenericQubit" begin
        q = GenericQubit(E=5.0)
        @test hilbertdim(q) == 2
        evals = eigenvals(q; evals_count=2)
        @test length(evals) == 2
        @test evals[1] ≈ -2.5
        @test evals[2] ≈ 2.5
    end

    @testset "Transmon" begin
        t = Transmon(EJ=30.0, EC=1.2, ng=0.0, ncut=30, truncated_dim=4)
        @test hilbertdim(t) == 61

        evals = eigenvals(t; evals_count=4)
        @test length(evals) == 4
        @test issorted(evals)

        # Transmon regime (EJ >> EC): ω01 ≈ sqrt(8*EJ*EC) - EC
        ω01 = evals[2] - evals[1]
        ω_approx = sqrt(8 * t.EJ * t.EC) - t.EC
        @test abs(ω01 - ω_approx) / ω_approx < 0.05
    end

    @testset "Oscillator" begin
        osc = Oscillator(E_osc=6.0, truncated_dim=10)
        @test hilbertdim(osc) == 10
        evals = eigenvals(osc; evals_count=5)
        # H = E_osc * n, so E_k = E_osc * k
        for k in 0:4
            @test evals[k + 1] ≈ 6.0 * k rtol=1e-10
        end
    end

    @testset "KerrOscillator" begin
        kosc = KerrOscillator(E_osc=6.0, K=0.2, truncated_dim=10)
        evals = eigenvals(kosc; evals_count=4)
        # H = E_osc * n - K * n*(n-1), so E_n = (E_osc + K) * n - K * n^2
        for n in 0:3
            expected = (6.0 + 0.2) * n - 0.2 * n^2
            @test evals[n + 1] ≈ expected rtol=1e-10
        end
        ω01 = evals[2] - evals[1]
        ω12 = evals[3] - evals[2]
        @test ω12 < ω01
    end

    @testset "Circuit graph parsing" begin
        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [C, 0, 1, EC=0.5]
"""
        cg = parse_circuit(desc)
        @test length(cg.branches) == 2
        @test cg.num_nodes == 1
        @test cg.has_ground == true
        @test cg.branches[1].branch_type == JJ_branch
        @test cg.branches[1].parameters[:EJ] == 10.0
        @test cg.branches[2].branch_type == C_branch
    end

    @testset "Circuit topology" begin
        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [C, 0, 1, EC=0.5]
"""
        cg = parse_circuit(desc)
        tree = find_spanning_tree(cg)
        @test length(tree) >= 1
        closure = find_closure_branches(cg, tree)
        @test length(tree) + length(closure) == length(cg.branches)
    end

    @testset "Periodic operators" begin
        ncut = 5
        dim = 2 * ncut + 1
        n_op = n_operator_periodic(ncut)
        @test size(n_op.data) == (dim, dim)

        # Check diagonal values
        for (i, n) in enumerate(-ncut:ncut)
            @test real(n_op.data[i, i]) ≈ n
        end

        # cos²θ + sin²θ ≈ I (interior elements; boundary truncation causes edge errors)
        cos_op = cos_theta_operator(ncut)
        sin_op = sin_theta_operator(ncut)
        identity_check = cos_op * cos_op + sin_op * sin_op
        id_mat = Matrix{ComplexF64}(I, dim, dim)
        diff = Matrix(identity_check.data) - id_mat
        # Interior elements (excluding first/last row+col) should be exact
        interior = diff[2:end-1, 2:end-1]
        @test all(abs.(interior) .< 1e-10)
    end

    @testset "HO operators" begin
        cutoff = 20
        osc_len = 1.0
        phi = phi_operator_ho(cutoff, osc_len)
        n_op = n_operator_ho(cutoff, osc_len)

        # [φ, n] ≈ i (within truncation errors)
        comm = phi * n_op - n_op * phi
        @test abs(comm.data[1, 1] - 1im) < 0.1
    end

    @testset "TunableTransmon" begin
        tt = TunableTransmon(EJmax=20.0, EC=0.5, d=0.0, flux=0.0, ncut=30, truncated_dim=4)

        # At flux=0, symmetric SQUID: EJ_eff = EJmax
        @test ej_effective(tt) ≈ 20.0

        # At flux=0.5, symmetric SQUID: EJ_eff = 0
        tt.flux = 0.5
        @test ej_effective(tt) ≈ 0.0 atol=1e-10

        # With asymmetry d=0.1: EJ_eff at flux=0 is still EJmax
        tt2 = TunableTransmon(EJmax=20.0, EC=0.5, d=0.1, flux=0.0, ncut=30, truncated_dim=4)
        @test ej_effective(tt2) ≈ 20.0

        # EJ_eff at flux=0.5 with asymmetry: EJ_eff = EJmax * d
        tt2.flux = 0.5
        @test ej_effective(tt2) ≈ 20.0 * 0.1 rtol=1e-10

        # ω01 should decrease with flux (transmon regime)
        tt3 = TunableTransmon(EJmax=20.0, EC=0.5, d=0.0, flux=0.0, ncut=30, truncated_dim=4)
        e0 = eigenvals(tt3; evals_count=2)
        tt3.flux = 0.25
        e1 = eigenvals(tt3; evals_count=2)
        @test (e0[2] - e0[1]) > (e1[2] - e1[1])

        # Flux=0 should match regular Transmon with same EJ, EC
        tt4 = TunableTransmon(EJmax=30.0, EC=1.2, d=0.0, flux=0.0, ncut=30, truncated_dim=4)
        t_ref = Transmon(EJ=30.0, EC=1.2, ng=0.0, ncut=30, truncated_dim=4)
        @test eigenvals(tt4; evals_count=4) ≈ eigenvals(t_ref; evals_count=4) atol=1e-10
    end

    @testset "Circuit flux sweep via set_param!" begin
        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [C, 0, 1, EC=0.5]
"""
        circ = Circuit(desc; ncut=30)
        sd = get_spectrum_vs_paramvals(circ, Symbol("Φ1"), [0.0, Float64(π)]; evals_count=3)
        # At Φ=0, full EJ; at Φ=π, EJ→0 so ω01 → small
        w01_0 = sd.eigenvalues[1, 2] - sd.eigenvalues[1, 1]
        w01_pi = sd.eigenvalues[2, 2] - sd.eigenvalues[2, 1]
        @test w01_0 > w01_pi
        @test w01_0 > 5.0   # should be well in transmon regime
        @test w01_pi < 3.0   # near charge regime at frustration
    end

    @testset "Circuit scqubits-style indexed parameters" begin
        desc = """
branches:
  - [JJ, 0, 1, EJ=4.5, EC=0.1]
  - [JJ, 1, 0, EJ=10.5, EC=0.1]
  - [C, 1, 0, EC=0.2]
  - [C, 1, 2, EC=5.0]
  - [JJ, 0, 2, EJ=30.0, EC=0.1]
  - [JJ, 2, 0, EJ=20.0, EC=0.1]
  - [C, 2, 0, EC=0.1]
  - [C, 2, 3, EC=5.0]
  - [JJ, 0, 3, EJ=4.6, EC=0.1]
  - [JJ, 3, 0, EJ=10.0, EC=0.1]
  - [C, 3, 0, EC=0.2]
  - [C, 1, 3, EC=500.0]
"""
        circ = Circuit(desc; ncut=6)

        @test string.(external_fluxes(circ)) == ["Φ1", "Φ2", "Φ3"]
        @test string.(offset_charges(circ)) == ["ng1", "ng2", "ng3"]
        flux_map = sym_external_fluxes(circ)
        @test length(flux_map) == 3
        for (k, flux) in enumerate(external_fluxes(circ))
            @test haskey(flux_map, flux)
            @test flux_map[flux].closure_branch == circ.symbolic_circuit.superconducting_closure_branches[k]
            @test flux_map[flux].loop == circ.symbolic_circuit.superconducting_loops[k]
        end

        cg = parse_circuit(desc)
        tree = find_spanning_tree(cg)
        closure = find_closure_branches(cg, tree)
        closure_sc, loops_sc = find_superconducting_loops(cg)
        floops = find_fundamental_loops(cg, tree)

        @test tree == [3, 4, 8]
        @test closure == [1, 2, 5, 6, 7, 9, 10, 11, 12]
        @test closure_sc == [2, 6, 10]
        @test loops_sc == [[(2, 1), (1, 1)], [(6, 1), (5, 1)], [(10, 1), (9, 1)]]
        @test floops[end] == [(12, 1), (4, -1), (8, -1)]

        @test occursin("Φ1", string(sym_hamiltonian(circ)))
        @test occursin("Φ1", string(sym_hamiltonian_node(circ)))
        @test !occursin("Φext", string(sym_hamiltonian(circ)))
        @test !occursin("Φext", string(sym_hamiltonian_node(circ)))

        set_param!(circ, Symbol("Φ2"), 0.3)
        @test get_param(circ, Symbol("Φ2")) == 0.3

        set_param!(circ, :ng2, 0.125)
        @test get_param(circ, :ng2) == 0.125

        phi_vals = [0.0, π / 4, π / 2]
        sd_sym = get_spectrum_vs_paramvals(circ, Symbol("Φ2"), phi_vals; evals_count=2)
        @test size(sd_sym.eigenvalues) == (3, 2)
        @test get_param(circ, Symbol("Φ2")) == 0.3

        sd_str = get_spectrum_vs_paramvals(circ, "Φ2", phi_vals; evals_count=2)
        @test sd_str.param_name == Symbol("Φ2")
        @test sd_str.eigenvalues ≈ sd_sym.eigenvalues

        ps = ParameterSweep(circ, "Φ2", phi_vals; evals_count=2)
        @test ps.param_name == Symbol("Φ2")

        @test_throws ErrorException set_param!(circ, Symbol("Φext_2"), 0.1)
        @test_throws ErrorException get_param(circ, Symbol("Φext_2"))
        @test_throws ErrorException set_param!(circ, :ng_2, 0.1)
        @test_throws ErrorException get_param(circ, :ng_2)
        @test_throws ErrorException set_param!(circ, Symbol("Φ99"), 0.1)
        @test_throws ErrorException get_param(circ, Symbol("Φ99"))
        @test_throws ErrorException set_param!(circ, :ng99, 0.1)
        @test_throws ErrorException get_param(circ, :ng99)
    end

    @testset "Circuit periodic-only offset charges" begin
        desc = """
branches:
  - [JJ, 0, 1, EJ=2.0, EC=1.0]
  - [C, 1, 2, EC=0.5]
  - [L, 0, 2, EL=0.6]
"""
        circ = Circuit(desc; ncut=6, cutoff_ext=8)

        @test circ.var_categories.periodic == [1]
        @test circ.var_categories.extended == [2]
        @test string.(offset_charges(circ)) == ["ng1"]
        @test get_param(circ, :ng1) == 0.0
        @test_throws ErrorException set_param!(circ, :ng2, 0.1)
        @test_throws ErrorException get_param(circ, :ng2)
        @test_throws ArgumentError set_offset_charge!(circ, 2, 0.1)

        evals0 = eigenvals(circ; evals_count=2)
        set_param!(circ, :ng1, 0.25)
        @test get_param(circ, :ng1) == 0.25
        evals1 = eigenvals(circ; evals_count=2)
        @test !isapprox(evals0[2] - evals0[1], evals1[2] - evals1[1]; atol=1e-8)
    end

    @testset "Circuit offset charge transformation" begin
        Sym = ScQubitsMimic.Symbolics
        desc = """
branches:
  - [L, 0, 1, EL=0.6]
  - [C, 1, 2, EC=0.5]
  - [JJ, 0, 2, EJ=8.0, EC=0.4]
"""
        circ = Circuit(desc; ncut=6)
        @test circ.var_categories.periodic == [2]
        @test circ.var_categories.extended == [1]

        eqs = offset_charge_transformation(circ)
        @test length(eqs) == 1
        eq = eqs[1]
        @test string(eq.lhs) == "ng2"
        @test occursin("q_n1", string(eq.rhs)) || occursin("q_n2", string(eq.rhs))

        Tt_inv = inv(circ.transformation_matrix')
        q_nodes = [Sym.Num(Sym.variable(Symbol("q_n$(j)"))) for j in 1:circ.symbolic_circuit.graph.num_nodes]
        expected_rhs = sum(Tt_inv[2, j] * q_nodes[j] for j in 1:length(q_nodes); init=Sym.Num(0))
        @test string(ScQubitsMimic.Symbolics.simplify(eq.rhs - expected_rhs)) == "0"
    end

    @testset "Circuit cutoff property API" begin
        desc = """
branches:
  - [JJ, 0, 1, EJ=2.0, EC=1.0]
  - [C, 1, 2, EC=0.5]
  - [L, 0, 2, EL=0.6]
"""
        circ = Circuit(desc; ncut=6, cutoff_ext=8)

        @test circ.cutoff_names == [:cutoff_n_1, :cutoff_ext_2]
        @test :cutoff_names in propertynames(circ)
        @test :cutoff_n_1 in propertynames(circ)
        @test :cutoff_ext_2 in propertynames(circ)
        @test circ.cutoff_n_1 == 6
        @test circ.cutoff_ext_2 == 8

        hamiltonian(circ)
        @test circ._hamiltonian_cache !== nothing

        circ.cutoff_n_1 = 4
        @test circ.cutoff_n_1 == 4
        @test circ.cutoffs[1] == 9
        @test circ._hamiltonian_cache === nothing

        hamiltonian(circ)
        circ.cutoff_ext_2 = 7
        @test circ.cutoff_ext_2 == 7
        @test circ.cutoffs[2] == 7
        @test hilbertdim(circ) == 9 * 7
        @test circ._hamiltonian_cache === nothing

        @test_throws ArgumentError setproperty!(circ, :cutoff_n_2, 3)
        @test_throws ArgumentError setproperty!(circ, :cutoff_ext_1, 3)
        @test_throws ArgumentError setproperty!(circ, :cutoff_n_1, -1)
        @test_throws ArgumentError setproperty!(circ, :cutoff_ext_2, 0)
        @test_throws ArgumentError setproperty!(circ, :cutoff_n_1, 1.5)
    end

    @testset "SpectrumLookup" begin
        t = Transmon(EJ=30.0, EC=1.2, ng=0.0, ncut=10, truncated_dim=4)
        osc = Oscillator(E_osc=6.0, truncated_dim=5)
        hs = HilbertSpace([t, osc])

        lookup = generate_lookup!(hs; evals_count=8)

        # Ground state should map to (1,1)
        @test bare_index(hs, 1) == (1, 1)

        # Round-trip: dressed_index ∘ bare_index = identity
        for i in 1:8
            bi = bare_index(hs, i)
            @test dressed_index(hs, bi...) == i
        end

        # energy_by_dressed_index should match dressed_evals
        for i in 1:8
            @test energy_by_dressed_index(hs, i) ≈ lookup.dressed_evals[i]
        end

        # energy_by_bare_index should give same result
        @test energy_by_bare_index(hs, 1, 1) ≈ lookup.dressed_evals[1]
    end

    @testset "op_in_dressed_eigenbasis" begin
        t = Transmon(EJ=30.0, EC=1.2, ng=0.0, ncut=10, truncated_dim=4)
        osc = Oscillator(E_osc=6.0, truncated_dim=5)
        hs = HilbertSpace([t, osc])
        generate_lookup!(hs; evals_count=8)

        # Hamiltonian in dressed basis should be diagonal
        H = hamiltonian(hs)
        H_dressed = op_in_dressed_eigenbasis(hs, H; truncated_dim=8)
        # Off-diagonal elements should be ~0
        for i in 1:8, j in 1:8
            if i != j
                @test abs(H_dressed[i, j]) < 1e-10
            end
        end
        # Diagonal should match dressed eigenvalues
        for i in 1:8
            @test real(H_dressed[i, i]) ≈ hs.lookup.dressed_evals[i] atol=1e-10
        end
    end

    @testset "HilbertSpaceSweep" begin
        tmon = TunableTransmon(EJmax=20.0, EC=0.3, d=0.0, flux=0.0, ncut=10, truncated_dim=3)
        osc = Oscillator(E_osc=6.0, truncated_dim=4)
        hs = HilbertSpace([tmon, osc])

        sweep = HilbertSpaceSweep(hs,
            Dict(:flux => [0.0, 0.25, 0.5]),
            (hs, vals) -> begin
                tmon.flux = vals[:flux]
            end;
            evals_count=4
        )

        @test size(sweep.dressed_evals) == (3, 4)
        # ω01 should decrease with flux
        w01_0 = sweep.dressed_evals[1, 2] - sweep.dressed_evals[1, 1]
        w01_half = sweep.dressed_evals[3, 2] - sweep.dressed_evals[3, 1]
        @test w01_0 > w01_half
    end

    @testset "Dispersive shifts (chi/Kerr)" begin
        t = Transmon(EJ=30.0, EC=1.2, ng=0.0, ncut=10, truncated_dim=6)
        osc = Oscillator(E_osc=7.0, truncated_dim=8)
        hs = HilbertSpace([t, osc])
        add_interaction!(hs, 0.15, [t, osc],
            [s -> cos_phi_operator(s), s -> annihilation_operator(s) + creation_operator(s)])
        generate_lookup!(hs; evals_count=15)

        # chi_matrix matches manual computation
        chi = chi_matrix(hs)
        @test size(chi) == (2, 2)
        manual_chi = energy_by_bare_index(hs, 2, 2) - energy_by_bare_index(hs, 2, 1) -
                     energy_by_bare_index(hs, 1, 2) + energy_by_bare_index(hs, 1, 1)
        @test chi[1, 2] ≈ manual_chi atol=1e-12
        @test chi[2, 1] ≈ chi[1, 2]  # symmetric

        # self_kerr: transmon anharmonicity should be negative, ~EC magnitude
        K_t = self_kerr(hs, 1)
        @test K_t < 0
        @test abs(K_t) < 3 * t.EC  # within reasonable range

        # self_kerr: harmonic oscillator should have ~0 anharmonicity
        K_osc = self_kerr(hs, 2)
        @test abs(K_osc) < 0.5

        # lamb_shift should be finite
        ls = lamb_shift(hs, 1)
        @test isfinite(ls)
        @test abs(ls) < 2.0  # should be a small correction

        # Error without lookup
        hs2 = HilbertSpace([GenericQubit(E=5.0), Oscillator(E_osc=6.0, truncated_dim=3)])
        @test_throws ErrorException chi_matrix(hs2)
        @test_throws ErrorException self_kerr(hs2, 1)
        @test_throws ErrorException lamb_shift(hs2, 1)
    end

    @testset "Dispersive sweep" begin
        tmon = TunableTransmon(EJmax=20.0, EC=0.3, d=0.0, flux=0.0, ncut=8, truncated_dim=4)
        osc = Oscillator(E_osc=6.0, truncated_dim=5)
        hs = HilbertSpace([tmon, osc])
        add_interaction!(hs, 0.1, [tmon, osc],
            [s -> cos_phi_operator(s), s -> annihilation_operator(s) + creation_operator(s)])

        sweep = HilbertSpaceSweep(hs,
            Dict(:flux => [0.0, 0.125, 0.25]),
            (hs, vals) -> begin tmon.flux = vals[:flux] end;
            evals_count=10, store_lookups=true)

        chi_arr = chi_matrix(sweep)
        @test size(chi_arr) == (3, 2, 2)

        kerr_vec = self_kerr(sweep, 1)
        @test length(kerr_vec) == 3
        @test all(k -> k < 0, kerr_vec)  # transmon anharmonicity is negative

        lamb_vec = lamb_shift(sweep, 1)
        @test length(lamb_vec) == 3
    end

    @testset "Normal-mode decomposition" begin
        Sym = ScQubitsMimic.Symbolics  # access Symbolics via parent module

        # Single-node Transmon: identity transform unchanged
        desc1 = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [C, 0, 1, EC=0.5]
"""
        circ1 = Circuit(desc1; ncut=10)
        T1, vc1 = variable_transformation(circ1)
        @test T1 ≈ Matrix{Float64}(I, 1, 1)
        @test 1 in vc1.periodic

        # Two-node LC circuit: should diagonalize extended subspace
        desc2 = """
branches:
  - [L, 0, 1, EL=1.0]
  - [C, 0, 1, EC=0.5]
  - [L, 1, 2, EL=0.5]
  - [C, 1, 2, EC=0.3]
  - [L, 0, 2, EL=0.8]
  - [C, 0, 2, EC=0.4]
"""
        cg2 = parse_circuit(desc2)
        sc2 = build_symbolic_circuit(cg2)
        T2, vc2 = compute_variable_transformation(sc2)
        @test length(vc2.extended) == 2
        @test size(T2) == (2, 2)

        # After transformation, L_inv should be diagonal for extended block
        L_inv = Float64.(Sym.value.(sc2.inv_inductance_matrix))
        L_inv_t = T2' * L_inv * T2
        @test abs(L_inv_t[1, 2]) < 1e-10
        @test abs(L_inv_t[2, 1]) < 1e-10
        @test L_inv_t[1, 1] > 0  # positive mode frequency²
        @test L_inv_t[2, 2] > 0

        # Mass normalization: T' * C * T = I for extended block
        C_mat = Float64.(Sym.value.(sc2.capacitance_matrix))
        C_t = T2' * C_mat * T2
        @test C_t ≈ Matrix{Float64}(I, 2, 2) atol=1e-10

        # Single extended mode (Fluxonium-like): identity still correct
        desc3 = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [L, 0, 1, EL=0.5]
"""
        cg3 = parse_circuit(desc3)
        sc3 = build_symbolic_circuit(cg3)
        T3, vc3 = compute_variable_transformation(sc3)
        @test T3 ≈ Matrix{Float64}(I, 1, 1)
        @test 1 in vc3.extended  # JJ + inductor → extended
    end

    @testset "Hierarchical diagonalization" begin
        # Two transmons coupled capacitively
        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 2, EJ=8.0, EC=0.4]
  - [C, 1, 2, EC=0.1]
"""
        circ = Circuit(desc; ncut=10)

        # Full diag eigenvalues for reference
        evals_full = eigenvals(circ; evals_count=6)

        # Hierarchical: split into 2 groups
        hs = hierarchical_diag(circ;
            system_hierarchy=[[1], [2]],
            subsystem_trunc_dims=Dict(1=>10, 2=>10))

        @test length(hs.subsystems) == 2
        @test hs.subsystems[1] isa SubCircuit
        @test hilbertdim(hs.subsystems[1]) == 10
        @test hilbertdim(hs.subsystems[2]) == 10

        # Hierarchical eigenvalues should approximate full diag
        evals_hier = eigenvals(hs; evals_count=6)
        @test length(evals_hier) == 6
        # Ground state should match well (within 5% — hierarchical approx with truncation)
        @test abs(evals_hier[1] - evals_full[1]) / abs(evals_full[1]) < 0.05

        # Error on invalid hierarchy
        @test_throws ErrorException hierarchical_diag(circ;
            system_hierarchy=[[1]], subsystem_trunc_dims=Dict(1=>5))
    end

    @testset "Hierarchical diag: cross-group capacitive coupling (factor-of-2)" begin
        # With full truncation (no truncation), hierarchical must match full diag exactly
        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 2, EJ=8.0, EC=0.4]
  - [C, 1, 2, EC=0.1]
"""
        circ = Circuit(desc; ncut=15)
        nstates = 2 * 15 + 1   # full dim for periodic mode with ncut=15

        evals_full = eigenvals(circ; evals_count=6)
        evals_full .-= evals_full[1]

        hs = hierarchical_diag(circ;
            system_hierarchy=[[1], [2]],
            subsystem_trunc_dims=Dict(1=>nstates, 2=>nstates))
        evals_hier = eigenvals(hs; evals_count=6)
        evals_hier .-= evals_hier[1]

        # With no truncation, must be exact (up to floating point)
        @test evals_hier ≈ evals_full atol=1e-8
    end

    @testset "Hierarchical diag: offset-charge regression" begin
        # Nonzero offset charges should still match full direct diagonalization
        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 2, EJ=8.0, EC=0.4]
  - [C, 1, 2, EC=0.1]
"""
        circ = Circuit(desc; ncut=10, offset_charges=[0.2, -0.15])
        evals_full = eigenvals(circ; evals_count=6)

        hs = hierarchical_diag(circ;
            system_hierarchy=[[1], [2]],
            subsystem_trunc_dims=Dict(1=>21, 2=>21))
        evals_hier = eigenvals(hs; evals_count=6)

        @test evals_hier ≈ evals_full atol=1e-8
    end

    @testset "Hierarchical diag: cross-group Josephson coupling" begin
        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 2, EJ=8.0, EC=0.4]
  - [JJ, 1, 2, EJ=2.0, EC=0.05]
"""
        circ = Circuit(desc; ncut=15)
        nstates = 2 * 15 + 1

        evals_full = eigenvals(circ; evals_count=6)
        evals_full .-= evals_full[1]

        hs = hierarchical_diag(circ;
            system_hierarchy=[[1], [2]],
            subsystem_trunc_dims=Dict(1=>nstates, 2=>nstates))
        evals_hier = eigenvals(hs; evals_count=6)
        evals_hier .-= evals_hier[1]

        # With full truncation dims, cross-group Josephson should match exactly
        @test evals_hier ≈ evals_full atol=1e-8

        # With truncated dims (15 per mode), should still be close (<5%)
        hs_trunc = hierarchical_diag(circ;
            system_hierarchy=[[1], [2]],
            subsystem_trunc_dims=Dict(1=>15, 2=>15))
        evals_trunc = eigenvals(hs_trunc; evals_count=6)
        evals_trunc .-= evals_trunc[1]
        for i in 2:6
            @test abs(evals_trunc[i] - evals_full[i]) / abs(evals_full[i]) < 0.05
        end
    end

    @testset "Hierarchical diag: cross-group inductive coupling" begin
        # Fluxonium-like modes with cross inductor
        desc = """
branches:
  - [JJ, 0, 1, EJ=8.0, EC=2.5]
  - [L, 0, 1, EL=0.5]
  - [JJ, 0, 2, EJ=6.0, EC=2.0]
  - [L, 0, 2, EL=0.4]
  - [L, 1, 2, EL=0.05]
"""
        circ = Circuit(desc; ncut=15, ext_basis=:harmonic)
        nstates_full = circ.cutoffs[first(vcat(circ.var_categories.periodic, circ.var_categories.extended))]

        evals_full = eigenvals(circ; evals_count=6)
        evals_full .-= evals_full[1]

        # With generous truncation, should be within 10%
        hs = hierarchical_diag(circ;
            system_hierarchy=[[1], [2]],
            subsystem_trunc_dims=Dict(1=>nstates_full, 2=>nstates_full))
        evals_hier = eigenvals(hs; evals_count=6)
        evals_hier .-= evals_hier[1]

        @test evals_hier ≈ evals_full atol=1e-6
    end

    @testset "Hierarchical diag: inductive external-flux regression" begin
        # A single JJ+L mode with external flux should match full direct diagonalization
        desc = """
branches:
  - [JJ, 0, 1, EJ=8.0, EC=2.5]
  - [L,  0, 1, EL=0.5]
"""
        circ = Circuit(desc; ncut=8, ext_basis=:harmonic)
        set_external_flux!(circ, 1, 0.7)
        evals_full = eigenvals(circ; evals_count=6)

        hs = hierarchical_diag(circ;
            system_hierarchy=[[1]],
            subsystem_trunc_dims=Dict(1=>circ.cutoffs[1]))
        evals_hier = eigenvals(hs; evals_count=6)

        @test evals_hier ≈ evals_full atol=1e-8
    end

    @testset "Hierarchical diag: add_operator! and extra_H_terms" begin
        # Test that add_operator! properly adds terms to Hamiltonian
        t = Transmon(EJ=10.0, EC=0.3, ng=0.0, ncut=5)
        hs = HilbertSpace([t])
        evals_before = eigenvals(hs; evals_count=4)

        # Add the bare Hamiltonian again as a perturbation (doubles it)
        add_operator!(hs, hamiltonian(t))

        evals_after = eigenvals(hs; evals_count=4)
        # Eigenvalues should be doubled (H + H = 2H)
        @test evals_after ≈ 2 .* evals_before rtol=1e-8
        @test length(hs.extra_H_terms) == 1
    end

    @testset "Hierarchical diag: nested (recursive) hierarchy" begin
        # 3-mode circuit with weak inter-mode capacitive coupling
        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 2, EJ=8.0, EC=0.4]
  - [JJ, 0, 3, EJ=6.0, EC=0.5]
  - [C, 1, 2, EC=0.08]
  - [C, 2, 3, EC=0.06]
"""
        circ = Circuit(desc; ncut=5)
        nstates = 2 * 5 + 1  # dim=11 per mode

        # Full diag reference
        evals_full = eigenvals(circ; evals_count=6)
        evals_full .-= evals_full[1]

        # Flat hierarchy: [[1], [2], [3]] with full truncation → exact
        hs_flat = hierarchical_diag(circ;
            system_hierarchy=[[1], [2], [3]],
            subsystem_trunc_dims=Dict(1=>nstates, 2=>nstates, 3=>nstates))
        evals_flat = eigenvals(hs_flat; evals_count=6)
        evals_flat .-= evals_flat[1]
        @test evals_flat ≈ evals_full atol=1e-6

        # Nested hierarchy: [[[1], [2]], [3]]
        # Full-dim leaves + generous intermediate truncation → near-exact
        hs_nested = hierarchical_diag(circ;
            system_hierarchy=[[[1], [2]], [3]],
            subsystem_trunc_dims=Dict(
                (1,1) => nstates,   # leaf [1]
                (1,2) => nstates,   # leaf [2]
                (1,)  => nstates^2, # intermediate: keep all states
                (2,)  => nstates))  # leaf [3]
        evals_nested = eigenvals(hs_nested; evals_count=6)
        evals_nested .-= evals_nested[1]
        @test evals_nested ≈ evals_full atol=1e-6

        # Truncated nested: [[[1],[2]], [3]] with real truncation
        hs_trunc = hierarchical_diag(circ;
            system_hierarchy=[[[1], [2]], [3]],
            subsystem_trunc_dims=Dict(
                (1,1) => 8,    # leaf [1]
                (1,2) => 8,    # leaf [2]
                (1,)  => 12,   # intermediate group
                (2,)  => 8))   # leaf [3]
        evals_trunc = eigenvals(hs_trunc; evals_count=6)
        evals_trunc .-= evals_trunc[1]
        for i in 2:4
            @test abs(evals_trunc[i] - evals_full[i]) / abs(evals_full[i]) < 0.2
        end
    end

    @testset "Hierarchical diag: nested with HierarchyGroup API" begin
        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 2, EJ=8.0, EC=0.4]
  - [C, 1, 2, EC=0.1]
"""
        circ = Circuit(desc; ncut=15)
        nstates = 2 * 15 + 1

        evals_full = eigenvals(circ; evals_count=6)
        evals_full .-= evals_full[1]

        # Use explicit HierarchyGroup/HierarchyLeaf types
        hier = HierarchyGroup([HierarchyLeaf([1]), HierarchyLeaf([2])])
        td = Dict((1,) => nstates, (2,) => nstates)
        hs = hierarchical_diag(circ;
            system_hierarchy=hier,
            subsystem_trunc_dims=td)
        evals_hier = eigenvals(hs; evals_count=6)
        evals_hier .-= evals_hier[1]

        @test evals_hier ≈ evals_full atol=1e-8
    end

    @testset "Hierarchical diag: single-group hierarchy" begin
        # Edge case: all modes in a single group
        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 2, EJ=8.0, EC=0.4]
  - [C, 1, 2, EC=0.1]
"""
        circ = Circuit(desc; ncut=10)

        evals_full = eigenvals(circ; evals_count=4)
        evals_full .-= evals_full[1]

        # Single leaf group with all modes
        hs = hierarchical_diag(circ;
            system_hierarchy=[[1, 2]],
            subsystem_trunc_dims=Dict(1=>20))
        evals_hier = eigenvals(hs; evals_count=4)
        evals_hier .-= evals_hier[1]

        # With enough truncation, should be exact
        @test evals_hier ≈ evals_full atol=1e-6
    end

    @testset "Hierarchical diag: Yan-style coupler flux sweep" begin
        desc = """
branches:
  - [JJ, 0, 1, EJ=4.5, EC=0.1]
  - [JJ, 1, 0, EJ=10.5, EC=0.1]
  - [C, 1, 0, EC=0.2]
  - [C, 1, 2, EC=5.0]
  - [JJ, 0, 2, EJ=30.0, EC=0.1]
  - [JJ, 2, 0, EJ=20.0, EC=0.1]
  - [C, 2, 0, EC=0.1]
  - [C, 2, 3, EC=5.0]
  - [JJ, 0, 3, EJ=4.6, EC=0.1]
  - [JJ, 3, 0, EJ=10.0, EC=0.1]
  - [C, 3, 0, EC=0.2]
  - [C, 1, 3, EC=500.0]
"""
        flux_vals = [-π, 0.0, π]

        qb1_w01 = Float64[]
        cplr_w01 = Float64[]
        qb2_w01 = Float64[]

        for phi2 in flux_vals
            circ = Circuit(desc; ncut=6)
            set_param!(circ, :Φ1, 0.0)
            set_param!(circ, :Φ2, phi2)
            set_param!(circ, :Φ3, 0.0)

            hs = hierarchical_diag(circ;
                system_hierarchy=[[1], [2], [3]],
                subsystem_trunc_dims=Dict(1=>3, 2=>3, 3=>3))

            @test length(hs.subsystems) == 3
            @test all(sub -> sub isa SubCircuit, hs.subsystems)

            push!(qb1_w01, eigenvals(hs.subsystems[1]; evals_count=2)[2] -
                            eigenvals(hs.subsystems[1]; evals_count=2)[1])
            push!(cplr_w01, eigenvals(hs.subsystems[2]; evals_count=2)[2] -
                             eigenvals(hs.subsystems[2]; evals_count=2)[1])
            push!(qb2_w01, eigenvals(hs.subsystems[3]; evals_count=2)[2] -
                            eigenvals(hs.subsystems[3]; evals_count=2)[1])
        end

        @test qb1_w01[1] ≈ qb1_w01[2] atol=1e-10
        @test qb1_w01[2] ≈ qb1_w01[3] atol=1e-10
        @test qb2_w01[1] ≈ qb2_w01[2] atol=1e-10
        @test qb2_w01[2] ≈ qb2_w01[3] atol=1e-10
        @test cplr_w01[2] > cplr_w01[1]
        @test cplr_w01[2] > cplr_w01[3]
        @test cplr_w01[1] ≈ cplr_w01[3] atol=1e-10
    end

    @testset "Symbolic accessors" begin
        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [C, 0, 1, EC=0.5]
"""
        circ = Circuit(desc; ncut=10)
        # sym_hamiltonian returns a symbolic expression (not nothing)
        @test sym_hamiltonian(circ) !== nothing
        @test sym_hamiltonian_node(circ) !== nothing
        T, vc = variable_transformation(circ)
        @test size(T) == (1, 1)
        @test 1 in vc.periodic
    end

    # ── Branch parameter sweeps (GAP-PARAMSWEEP) ───────────────────────────

    @testset "Branch parameter sweeps" begin
        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [C, 0, 1, EC=0.5]
"""
        circ = Circuit(desc; ncut=20)
        evals_orig = eigenvals(circ; evals_count=4)
        w01_orig = evals_orig[2] - evals_orig[1]

        # Test 1: EJ sweep — increasing EJ increases ω01
        set_param!(circ, :EJ, 20.0)
        @test get_param(circ, :EJ) == 20.0
        evals_high = eigenvals(circ; evals_count=4)
        w01_high = evals_high[2] - evals_high[1]
        @test w01_high > w01_orig

        # Test 2: restore original EJ
        set_param!(circ, :EJ, 10.0)
        evals_restored = eigenvals(circ; evals_count=4)
        @test evals_restored ≈ evals_orig rtol=1e-10

        # Test 3: indexed branch param (EJ on branch 1)
        set_param!(circ, :EJ_1, 15.0)
        @test get_param(circ, :EJ_1) == 15.0
        w01_15 = eigenvals(circ; evals_count=4)[2] - eigenvals(circ; evals_count=4)[1]
        @test w01_15 > w01_orig  # EJ=15 > EJ=10

        # Test 4: EC sweep — increasing EC (decreasing capacitance) increases ω01
        set_param!(circ, :EJ_1, 10.0)  # restore
        set_param!(circ, :EC, 0.5)     # set EC on first branch with EC (JJ branch)
        @test get_param(circ, :EC) == 0.5
        w01_highec = eigenvals(circ; evals_count=4)[2] - eigenvals(circ; evals_count=4)[1]
        # Higher EC means less capacitance, higher charging energy
        # For transmon, ω01 ≈ √(8EJ·EC) - EC, so higher EC gives higher ω01
        @test w01_highec > w01_orig

        # Test 5: original graph is NOT mutated
        @test circ.symbolic_circuit.graph.branches[1].parameters[:EJ] == 10.0

        # Test 6: error on invalid branch index
        @test_throws ErrorException set_param!(circ, :EJ_99, 5.0)

        # Test 7: error on unrecognized parameter
        @test_throws ErrorException set_param!(circ, :XYZ, 1.0)
    end

    @testset "Branch parameter sweep with get_spectrum_vs_paramvals" begin
        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [C, 0, 1, EC=0.5]
"""
        circ = Circuit(desc; ncut=20)

        # EJ sweep via get_spectrum_vs_paramvals
        ej_vals = collect(range(5.0, 20.0, length=5))
        sd = get_spectrum_vs_paramvals(circ, :EJ, ej_vals; evals_count=4)
        @test size(sd.eigenvalues) == (5, 4)
        # ω01 should increase with EJ (transmon regime)
        w01s = sd.eigenvalues[:, 2] .- sd.eigenvalues[:, 1]
        @test issorted(w01s)
    end

    @testset "Two-JJ SQUID indexed parameter" begin
        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [C, 0, 1, EC=0.5]
"""
        circ = Circuit(desc; ncut=15)
        evals_sym = eigenvals(circ; evals_count=4)

        # Make asymmetric: EJ_1=15, EJ_2=5
        set_param!(circ, :EJ_1, 15.0)
        set_param!(circ, :EJ_2, 5.0)
        @test get_param(circ, :EJ_1) == 15.0
        @test get_param(circ, :EJ_2) == 5.0

        # At zero flux, EJ_eff = EJ1 + EJ2 = 20 either way → same spectrum
        evals_asym = eigenvals(circ; evals_count=4)
        @test evals_asym ≈ evals_sym rtol=1e-10

        # Increasing total EJ should increase ω01
        set_param!(circ, :EJ_1, 20.0)  # total EJ = 25 now
        w01_high = eigenvals(circ; evals_count=4)[2] - eigenvals(circ; evals_count=4)[1]
        w01_sym = evals_sym[2] - evals_sym[1]
        @test w01_high > w01_sym
    end

    # ── Frozen variable handling (GAP-FROZEN) ──────────────────────────────

    @testset "Frozen variable handling" begin
        # Frozen node with ground connection — capacitively shunts node 1
        # Node 1: JJ to ground (periodic mode)
        # Node 2: capacitively coupled to node 1 AND ground (frozen, but shunts)
        desc_frozen = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [C, 1, 2, EC=0.5]
  - [C, 0, 2, EC=1.0]
"""
        circ_frozen = Circuit(desc_frozen; ncut=15)
        T, vc = variable_transformation(circ_frozen)

        # Node 2 should be classified as frozen
        @test 2 in vc.frozen
        @test 1 in vc.periodic
        @test isempty(vc.extended)

        # Should produce valid eigenvalues (only 1 active mode)
        evals = eigenvals(circ_frozen; evals_count=4)
        @test length(evals) == 4
        @test issorted(evals)

        # Compare with bare JJ circuit (no shunt from frozen node)
        desc_bare = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
"""
        circ_bare = Circuit(desc_bare; ncut=15)
        evals_bare = eigenvals(circ_bare; evals_count=4)

        # The frozen node with ground path adds shunt capacitance → lower ω01
        w01_frozen = evals[2] - evals[1]
        w01_bare = evals_bare[2] - evals_bare[1]
        @test w01_frozen < w01_bare
    end

    @testset "Frozen node to ground" begin
        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [C, 1, 2, EC=0.5]
  - [C, 0, 2, EC=1.0]
"""
        circ = Circuit(desc; ncut=15)
        T, vc = variable_transformation(circ)
        @test 2 in vc.frozen
        @test 1 in vc.periodic

        evals = eigenvals(circ; evals_count=4)
        @test length(evals) == 4
        @test issorted(evals)
    end

    @testset "No frozen nodes in standard circuit" begin
        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [C, 0, 1, EC=0.5]
"""
        circ = Circuit(desc; ncut=15)
        _, vc = variable_transformation(circ)
        @test isempty(vc.frozen)
    end

    @testset "sym_lagrangian" begin
        Sym = ScQubitsMimic.Symbolics

        # Simple transmon: L has kinetic (φ̇) and Josephson (cos) terms
        desc_transmon = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [C, 0, 1, EC=0.5]
"""
        circ1 = Circuit(desc_transmon; ncut=10)
        L_node = sym_lagrangian(circ1; vars_type=:node)
        @test L_node isa Sym.Num
        L_str = string(L_node)
        @test occursin("φ̇", L_str)
        @test occursin("cos", L_str)

        # vars_type=:new should produce θ̇ variables
        L_new = sym_lagrangian(circ1; vars_type=:new)
        @test L_new isa Sym.Num
        L_new_str = string(L_new)
        @test occursin("θ̇", L_new_str)
        @test occursin("cos", L_new_str)

        # Fluxonium-like (JJ + L): inductive terms present
        desc_flux = """
branches:
  - [JJ, 0, 1, EJ=8.0, EC=2.5]
  - [L, 0, 1, EL=0.5]
"""
        circ2 = Circuit(desc_flux; ncut=10)
        L_flux = sym_lagrangian(circ2; vars_type=:node)
        L_flux_str = string(L_flux)
        @test occursin("φ̇", L_flux_str)
        @test occursin("cos", L_flux_str)

        # External flux circuit: two JJs → Φ1 appears, no Φext
        desc_ext = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [C, 0, 1, EC=0.5]
"""
        circ3 = Circuit(desc_ext; ncut=10)
        L_ext = sym_lagrangian(circ3; vars_type=:node)
        L_ext_str = string(L_ext)
        @test occursin("Φ1", L_ext_str)
        @test !occursin("Φext", L_ext_str)

        # Multi-node: both vars_type options work
        desc_multi = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 2, EJ=8.0, EC=0.4]
  - [C, 1, 2, EC=0.1]
"""
        circ4 = Circuit(desc_multi; ncut=6)
        L4_node = sym_lagrangian(circ4; vars_type=:node)
        L4_new = sym_lagrangian(circ4; vars_type=:new)
        @test L4_node isa Sym.Num
        @test L4_new isa Sym.Num
        @test occursin("φ̇", string(L4_node))
        @test occursin("θ̇", string(L4_new))

        # Invalid vars_type throws ArgumentError
        @test_throws ArgumentError sym_lagrangian(circ1; vars_type=:invalid)

        # Kinetic coefficient consistency: coefficient of φ̇₁² should be C/2
        sc = circ1.symbolic_circuit
        C_expected = 1 / (8 * 0.3) + 1 / (8 * 0.5)
        c = Sym.coeff(Sym.expand(L_node), sc.node_dot_vars[1]^2)
        @test Float64(Sym.value(c)) ≈ C_expected / 2 rtol=1e-10
    end

    # ── Hierarchical analysis APIs (configure!, sym_hamiltonian, sym_interaction) ─

    @testset "configure! stores hierarchy state" begin
        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 2, EJ=8.0, EC=0.4]
  - [C, 1, 2, EC=0.1]
"""
        circ = Circuit(desc; ncut=10)
        configure!(circ; system_hierarchy=[[1],[2]], subsystem_trunc_dims=Dict(1=>10, 2=>10))

        @test circ._hierarchical_diagonalization == true
        @test circ._hilbert_space isa HilbertSpace
        @test circ._subsystems !== nothing
        @test length(circ._subsystems) == 2
        @test all(sub -> sub isa SubCircuit, circ._subsystems)
        @test circ._subsystem_sym_hamiltonians !== nothing
        @test circ._subsystem_interactions_sym !== nothing
    end

    @testset "sym_hamiltonian with subsystem_index" begin
        Sym = ScQubitsMimic.Symbolics

        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 2, EJ=8.0, EC=0.4]
  - [C, 1, 2, EC=0.1]
"""
        circ = Circuit(desc; ncut=10)

        # Without configure!, subsystem_index should error
        @test_throws ErrorException sym_hamiltonian(circ; subsystem_index=1)

        configure!(circ; system_hierarchy=[[1],[2]], subsystem_trunc_dims=Dict(1=>10, 2=>10))

        # Full Hamiltonian still works
        H_full = sym_hamiltonian(circ)
        @test H_full isa Sym.Num

        # Subsystem Hamiltonians
        H1 = sym_hamiltonian(circ; subsystem_index=1)
        H2 = sym_hamiltonian(circ; subsystem_index=2)
        @test H1 isa Sym.Num
        @test H2 isa Sym.Num

        # H1 should reference mode-1 variables but NOT mode-2 variables
        h1_vars = Sym.get_variables(H1)
        ref_θ = [Sym.variable(:θ, i) for i in 1:2]
        ref_nθ = [Sym.variable(:nθ, i) for i in 1:2]
        has_mode(vars, i) = any(v -> isequal(v, ref_θ[i]) || isequal(v, ref_nθ[i]), vars)
        @test has_mode(h1_vars, 1)
        @test !has_mode(h1_vars, 2)

        # H2 should reference mode-2 variables but NOT mode-1 variables
        h2_vars = Sym.get_variables(H2)
        @test has_mode(h2_vars, 2)
        @test !has_mode(h2_vars, 1)

        # Invalid subsystem_index
        @test_throws ErrorException sym_hamiltonian(circ; subsystem_index=3)
    end

    @testset "sym_interaction returns nontrivial coupling" begin
        Sym = ScQubitsMimic.Symbolics

        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 2, EJ=8.0, EC=0.4]
  - [C, 1, 2, EC=0.1]
"""
        circ = Circuit(desc; ncut=10)

        # Without configure!, should error
        @test_throws ErrorException sym_interaction(circ; subsystem_indices=(1, 2))

        configure!(circ; system_hierarchy=[[1],[2]], subsystem_trunc_dims=Dict(1=>10, 2=>10))

        H_int = sym_interaction(circ; subsystem_indices=(1, 2))
        @test H_int isa Sym.Num

        # Should be nontrivial (not zero) for capacitively coupled circuit
        @test !isequal(H_int, Sym.Num(0))

        # Should contain cross-mode terms (both mode 1 and mode 2 variables)
        int_vars = Sym.get_variables(H_int)
        ref_θ = [Sym.variable(:θ, i) for i in 1:2]
        ref_nθ = [Sym.variable(:nθ, i) for i in 1:2]
        has_mode(vars, i) = any(v -> isequal(v, ref_θ[i]) || isequal(v, ref_nθ[i]), vars)
        @test has_mode(int_vars, 1)
        @test has_mode(int_vars, 2)

        # Symmetric: (1,2) and (2,1) should give same result
        H_int_rev = sym_interaction(circ; subsystem_indices=(2, 1))
        @test isequal(H_int, H_int_rev)
    end

    @testset "Symbolic decomposition completeness" begin
        Sym = ScQubitsMimic.Symbolics

        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 2, EJ=8.0, EC=0.4]
  - [C, 1, 2, EC=0.1]
"""
        circ = Circuit(desc; ncut=10)
        configure!(circ; system_hierarchy=[[1],[2]], subsystem_trunc_dims=Dict(1=>10, 2=>10))

        H_full = sym_hamiltonian(circ)
        H1 = sym_hamiltonian(circ; subsystem_index=1)
        H2 = sym_hamiltonian(circ; subsystem_index=2)
        H_int = sym_interaction(circ; subsystem_indices=(1, 2))

        # H1 + H2 + H_int should equal H_full (symbolic difference simplifies to 0)
        diff = Sym.simplify(Sym.expand(H1 + H2 + H_int - H_full))
        @test isequal(diff, Sym.Num(0)) || isequal(diff, 0)
    end

    @testset "Yan-style 3-loop coupler with configure!" begin
        desc = """
branches:
  - [JJ, 0, 1, EJ=4.5, EC=0.1]
  - [JJ, 1, 0, EJ=10.5, EC=0.1]
  - [C, 1, 0, EC=0.2]
  - [C, 1, 2, EC=5.0]
  - [JJ, 0, 2, EJ=30.0, EC=0.1]
  - [JJ, 2, 0, EJ=20.0, EC=0.1]
  - [C, 2, 0, EC=0.1]
  - [C, 2, 3, EC=5.0]
  - [JJ, 0, 3, EJ=4.6, EC=0.1]
  - [JJ, 3, 0, EJ=10.0, EC=0.1]
  - [C, 3, 0, EC=0.2]
  - [C, 1, 3, EC=500.0]
"""
        circ = Circuit(desc; ncut=6)
        set_param!(circ, :Φ1, 0.0)
        set_param!(circ, :Φ2, 0.0)
        set_param!(circ, :Φ3, 0.0)

        configure!(circ;
            system_hierarchy=[[1], [2], [3]],
            subsystem_trunc_dims=Dict(1=>3, 2=>3, 3=>3))

        # 3 subsystems
        @test length(circ._subsystems) == 3
        @test all(sub -> sub isa SubCircuit, circ._subsystems)
        @test circ._hilbert_space isa HilbertSpace

        # Each subsystem has its own symbolic Hamiltonian
        for i in 1:3
            Hi = sym_hamiltonian(circ; subsystem_index=i)
            @test Hi isa ScQubitsMimic.Symbolics.Num
        end

        # Interactions between all pairs should exist (capacitive coupling)
        for (i, j) in [(1,2), (2,3), (1,3)]
            H_int = sym_interaction(circ; subsystem_indices=(i, j))
            @test H_int isa ScQubitsMimic.Symbolics.Num
            # At least some pairs should have nontrivial coupling
        end

        # Check that (1,2) and (2,3) have nontrivial coupling (direct capacitive)
        Sym = ScQubitsMimic.Symbolics
        interaction_strength(expr) = begin
            coeff = split(replace(string(expr), " " => ""), "nθ")[1]
            abs(parse(Float64, coeff))
        end

        H12 = sym_interaction(circ; subsystem_indices=(1, 2))
        H23 = sym_interaction(circ; subsystem_indices=(2, 3))
        H13 = sym_interaction(circ; subsystem_indices=(1, 3))

        @test !isequal(H12, Sym.Num(0))
        @test !isequal(H23, Sym.Num(0))
        @test !isequal(H13, Sym.Num(0))
        @test interaction_strength(H13) > 1e-4
        @test interaction_strength(H12) > 10 * interaction_strength(H13)
        @test interaction_strength(H23) > 10 * interaction_strength(H13)
    end

    @testset "invalidate_cache! clears numerical but keeps symbolic" begin
        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 2, EJ=8.0, EC=0.4]
  - [C, 1, 2, EC=0.1]
"""
        circ = Circuit(desc; ncut=10)
        configure!(circ; system_hierarchy=[[1],[2]], subsystem_trunc_dims=Dict(1=>10, 2=>10))

        @test circ._hilbert_space !== nothing
        @test circ._subsystem_sym_hamiltonians !== nothing

        # After parameter change, numerical results cleared but symbolic stays
        set_param!(circ, :EJ, 12.0)
        @test circ._hilbert_space === nothing
        @test circ._subsystems === nothing
        @test circ._subsystem_sym_hamiltonians !== nothing  # symbolic still valid
        @test circ._hierarchical_diagonalization == true     # flag still set
    end

    @testset "Multi-mode subsystem grouping" begin
        Sym = ScQubitsMimic.Symbolics

        # 3-mode circuit with grouping [[1,2],[3]]
        desc = """
branches:
  - [JJ, 0, 1, EJ=10.0, EC=0.3]
  - [JJ, 0, 2, EJ=8.0, EC=0.4]
  - [JJ, 0, 3, EJ=6.0, EC=0.5]
  - [C, 1, 2, EC=0.1]
  - [C, 2, 3, EC=0.15]
"""
        circ = Circuit(desc; ncut=5)
        configure!(circ; system_hierarchy=[[1,2],[3]],
                  subsystem_trunc_dims=Dict(1=>20, 2=>10))

        H1 = sym_hamiltonian(circ; subsystem_index=1)
        H2 = sym_hamiltonian(circ; subsystem_index=2)

        # H1 should contain modes 1 and/or 2 but not mode 3
        h1_vars = Sym.get_variables(H1)
        ref_θ = [Sym.variable(:θ, i) for i in 1:3]
        ref_nθ = [Sym.variable(:nθ, i) for i in 1:3]
        has_mode(vars, i) = any(v -> isequal(v, ref_θ[i]) || isequal(v, ref_nθ[i]), vars)
        @test has_mode(h1_vars, 1) || has_mode(h1_vars, 2)
        @test !has_mode(h1_vars, 3)

        # H2 should contain mode 3 only
        h2_vars = Sym.get_variables(H2)
        @test has_mode(h2_vars, 3)
        @test !has_mode(h2_vars, 1) && !has_mode(h2_vars, 2)
    end

end
