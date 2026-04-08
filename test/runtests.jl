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
        # H = E_osc * (n + 0.5), so E_k = E_osc * (k + 0.5)
        for k in 0:4
            @test evals[k + 1] ≈ 6.0 * (k + 0.5) rtol=1e-10
        end
    end

    @testset "KerrOscillator" begin
        kosc = KerrOscillator(E_osc=6.0, K=-0.2, truncated_dim=10)
        evals = eigenvals(kosc; evals_count=4)
        # H = E_osc * n + K/2 * n*(n-1), so E_n = E_osc * n + K/2 * n*(n-1)
        for n in 0:3
            expected = 6.0 * n + (-0.2 / 2) * n * (n - 1)
            @test evals[n + 1] ≈ expected rtol=1e-10
        end
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
        sd = get_spectrum_vs_paramvals(circ, :flux, [0.0, Float64(π)]; evals_count=3)
        # At Φ=0, full EJ; at Φ=π, EJ→0 so ω01 → small
        w01_0 = sd.eigenvalues[1, 2] - sd.eigenvalues[1, 1]
        w01_pi = sd.eigenvalues[2, 2] - sd.eigenvalues[2, 1]
        @test w01_0 > w01_pi
        @test w01_0 > 5.0   # should be well in transmon regime
        @test w01_pi < 3.0   # near charge regime at frustration
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

end
