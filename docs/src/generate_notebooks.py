#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS_SRC = ROOT / "docs" / "src"
NOTEBOOKS_DIR = DOCS_SRC / "notebooks"
TEMPLATE_PATH = Path.home() / ".codex" / "skills" / "jupyter-notebook" / "assets" / "tutorial-template.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def load_template() -> dict:
    with TEMPLATE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def notebook(title: str, cells: list[dict]) -> dict:
    nb = load_template()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {
        "display_name": "Julia 1.12.5",
        "language": "julia",
        "name": "julia-1.12",
    }
    nb["metadata"]["language_info"] = {
        "name": "julia",
        "version": "1.12.5",
        "file_extension": ".jl",
        "mimetype": "application/julia",
    }
    nb["metadata"]["title"] = title
    return nb


def write_nb(path: Path, title: str, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(notebook(title, cells), f, indent=1, ensure_ascii=False)
        f.write("\n")


def export_names() -> list[str]:
    cmd = [
        "julia",
        "--project=.",
        "-e",
        (
            "using ScQubitsMimic; "
            "for name in sort!(string.(names(ScQubitsMimic; all=false, imported=false))); "
            "println(name); "
            "end"
        ),
    ]
    result = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def build_index(
    notebook_specs: list[tuple[str, str, str]],
    api_specs: list[tuple[str, str, str]],
) -> str:
    lines = [
        "# Notebook-First Documentation",
        "",
        "`ScQubitsMimic.jl` mirrors the scqubits workflow in Julia for the subset of APIs implemented and tested in this repository.",
        "",
        "This v1 documentation has two layers:",
        "- tutorial notebooks live in [`docs/src/notebooks`](./notebooks)",
        "- hand-curated API reference pages live in [`docs/src/api`](./api/index.md)",
        "- examples in [`examples`](../../examples) remain scratch or development artifacts",
        "- coverage is limited to the tested public API exported by `ScQubitsMimic`",
        "",
        "## What Is Implemented",
        "",
        "- Core spectral utilities for simple quantum systems",
        "- `Transmon` and `TunableTransmon` workflows",
        "- Circuit parsing, symbolic analysis, and quantization",
        "- Composite `HilbertSpace` analysis with bare/dressed lookup tables",
        "- Single-system and coupled-system parameter sweeps",
        "- Dispersive post-processing and hierarchical circuit diagonalization",
        "",
        "## Not Yet Ported From Python scqubits",
        "",
        "- The broader family of built-in qubit models such as `Fluxonium`, `FluxQubit`, and `ZeroPi`",
        "- GUI workflows and explorer tooling",
        "- The full Python plotting/noise/coherence API surface",
        "- The full scqubits preslicing and container interface around `ParameterSweep`",
        "",
        "## Notebook Guide",
        "",
        "| Notebook | Focus |",
        "| --- | --- |",
    ]
    for path, title, summary in notebook_specs:
        lines.append(f"| [`{path}`](./{path}) | {summary} |")
    lines.extend(
        [
            "",
            "## API Reference",
            "",
            "| Page | Focus |",
            "| --- | --- |",
        ]
    )
    for path, title, summary in api_specs:
        lines.append(f"| [`{path}`](./{path}) | {summary} |")
    lines.extend(
        [
            "",
            "## Coverage Map",
            "",
            "See [`api-coverage.md`](./api-coverage.md) for the full export-to-notebook mapping, including appendix-only APIs.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_api_coverage(
    mapping: dict[str, tuple[str, str]],
    appendix: dict[str, list[str]],
    section_specs: list[tuple[str, str, str]],
) -> str:
    grouped: dict[str, list[tuple[str, str]]] = {}
    for symbol, (path, note) in mapping.items():
        grouped.setdefault(path, []).append((symbol, note))

    lines = [
        "# API Coverage",
        "",
        "Every public export from `ScQubitsMimic` is assigned to either a primary tutorial notebook, a primary API reference page, or an appendix-only reference bucket.",
        "",
        "Circuit-related exports use the hand-curated API reference pages as the primary reference. The notebooks remain the runnable tutorial companions.",
        "",
    ]
    for path, title, _ in section_specs:
        entries = grouped.get(path, [])
        if not entries:
            continue
        lines.extend(
            [
                f"## [`{title}`](./{path})",
                "",
                "| Symbol | Notes |",
                "| --- | --- |",
            ]
        )
        for symbol, note in sorted(entries, key=lambda item: item[0].lower()):
            lines.append(f"| `{symbol}` | {note} |")
        lines.append("")

    lines.append("## Appendix-Only Reference")
    lines.append("")
    for section, symbols in appendix.items():
        lines.append(f"### {section}")
        lines.append("")
        lines.append(", ".join(f"`{symbol}`" for symbol in symbols))
        lines.append("")

    return "\n".join(lines)


NOTEBOOK_SPECS = [
    ("notebooks/01-core-api.ipynb", "01 Core API", "Shared conventions, constants, grids, simple systems, and common spectral routines."),
    ("notebooks/02-transmon.ipynb", "02 Transmon", "Fixed-frequency transmon workflows, operators, sweeps, matrix elements, and wavefunction interpretation."),
    ("notebooks/03-tunable-transmon.ipynb", "03 Tunable Transmon", "Flux-tunable transmon behavior and parity with the circuit-derived SQUID workflow."),
    ("notebooks/04-circuit-quantization.ipynb", "04 Circuit Quantization", "Circuit graphs, symbolic analysis, topology helpers, and the main `Circuit` API."),
    ("notebooks/05-hilbertspace-and-lookup.ipynb", "05 HilbertSpace And Lookup", "Composite systems, interactions, dressed spectra, and bare/dressed lookup tables."),
    ("notebooks/06-sweeps-and-dispersive.ipynb", "06 Sweeps And Dispersive", "Single-system and coupled-system sweeps, lookup-aware analysis, and dispersive summaries."),
    ("notebooks/07-advanced-circuit-hierarchy.ipynb", "07 Advanced Circuit Hierarchy", "Hierarchical diagonalization, truncation templates, subsystem extraction, and interaction decomposition."),
]

API_REFERENCE_SPECS = [
    ("api/index.md", "API Reference Index", "Landing page for the hand-curated API reference layer."),
    ("api/circuit.md", "Circuit", "Reference for `Circuit`, topology helpers, symbolic inspection, and parameter mutation APIs."),
    ("api/symbolic-circuit.md", "SymbolicCircuit", "Reference for `SymbolicCircuit`, its grouped fields, and `build_symbolic_circuit`."),
    ("api/circuit-hierarchy.md", "Circuit Hierarchy", "Reference for `configure!`, `hierarchical_diag`, hierarchy node types, and symbolic subsystem interactions."),
]

COVERAGE_SECTION_SPECS = [
    ("notebooks/01-core-api.ipynb", "01 Core API", ""),
    ("notebooks/02-transmon.ipynb", "02 Transmon", ""),
    ("notebooks/03-tunable-transmon.ipynb", "03 Tunable Transmon", ""),
    ("api/circuit.md", "Circuit API", ""),
    ("api/symbolic-circuit.md", "SymbolicCircuit API", ""),
    ("notebooks/05-hilbertspace-and-lookup.ipynb", "05 HilbertSpace And Lookup", ""),
    ("notebooks/06-sweeps-and-dispersive.ipynb", "06 Sweeps And Dispersive", ""),
    ("api/circuit-hierarchy.md", "Circuit Hierarchy API", ""),
]


API_MAPPING = {
    "AbstractOscillator": ("notebooks/01-core-api.ipynb", "Type hierarchy and dispatch context for oscillator-like systems."),
    "AbstractQuantumSystem": ("notebooks/01-core-api.ipynb", "Base abstraction used by the shared spectral routines."),
    "AbstractQubit": ("notebooks/01-core-api.ipynb", "Type hierarchy overview for qubit-like systems."),
    "AbstractQubit1d": ("notebooks/01-core-api.ipynb", "1D qubit subtype used by the wavefunction plotting API."),
    "AbstractQubitNd": ("notebooks/01-core-api.ipynb", "Higher-dimensional qubit subtype noted for future parity."),
    "GenericQubit": ("notebooks/01-core-api.ipynb", "Minimal two-level system used to illustrate the common API."),
    "Grid1d": ("notebooks/01-core-api.ipynb", "Uniform grid helper for phase-space discretization."),
    "KerrOscillator": ("notebooks/01-core-api.ipynb", "Simple anharmonic oscillator example."),
    "Oscillator": ("notebooks/01-core-api.ipynb", "Simple harmonic mode example."),
    "PhysicalConstants": ("notebooks/01-core-api.ipynb", "Namespace of physical constants used by unit conversions."),
    "ScQubitsMimic": ("notebooks/01-core-api.ipynb", "Package namespace and import surface."),
    "SpectrumData": ("notebooks/01-core-api.ipynb", "Shared container returned by single-system spectral sweeps."),
    "annihilation_operator": ("notebooks/01-core-api.ipynb", "Oscillator lowering operator."),
    "convert_units": ("notebooks/01-core-api.ipynb", "Energy unit conversion helper."),
    "creation_operator": ("notebooks/01-core-api.ipynb", "Oscillator raising operator."),
    "eigensys": ("notebooks/01-core-api.ipynb", "Shared eigensystem interface."),
    "eigenvals": ("notebooks/01-core-api.ipynb", "Shared eigenvalue interface."),
    "get_spectrum_vs_paramvals": ("notebooks/01-core-api.ipynb", "Single-system spectral sweep entry point."),
    "grid_points": ("notebooks/01-core-api.ipynb", "Grid sampling helper."),
    "grid_spacing": ("notebooks/01-core-api.ipynb", "Grid spacing helper."),
    "hamiltonian": ("notebooks/01-core-api.ipynb", "Shared Hamiltonian interface."),
    "hilbertdim": ("notebooks/01-core-api.ipynb", "Shared Hilbert-space dimension interface."),
    "matrixelement": ("notebooks/01-core-api.ipynb", "Single matrix element query in an eigenbasis."),
    "matrixelement_table": ("notebooks/01-core-api.ipynb", "Dense matrix-element table helper."),
    "number_operator": ("notebooks/01-core-api.ipynb", "Oscillator number operator."),
    "Transmon": ("notebooks/02-transmon.ipynb", "Fixed-frequency transmon construction and analysis."),
    "cos_phi_operator": ("notebooks/02-transmon.ipynb", "Transmon charge-basis cosine operator."),
    "exp_i_phi_operator": ("notebooks/02-transmon.ipynb", "Transmon phase-step operator."),
    "n_operator": ("notebooks/02-transmon.ipynb", "Transmon/TunableTransmon number operator."),
    "plot_matrixelements": ("notebooks/02-transmon.ipynb", "Makie extension helper for operator heatmaps."),
    "plot_wavefunction": ("notebooks/02-transmon.ipynb", "Makie extension helper for 1D qubit wavefunctions."),
    "potential": ("notebooks/02-transmon.ipynb", "Transmon potential-energy helper."),
    "sin_phi_operator": ("notebooks/02-transmon.ipynb", "Transmon charge-basis sine operator."),
    "TunableTransmon": ("notebooks/03-tunable-transmon.ipynb", "Flux-tunable transmon API."),
    "ej_effective": ("notebooks/03-tunable-transmon.ipynb", "Effective Josephson energy for a SQUID transmon."),
    "Branch": ("api/circuit.md", "Parsed circuit-graph branch object. Tutorial companion: [`04-circuit-quantization.ipynb`](./notebooks/04-circuit-quantization.ipynb)."),
    "BranchType": ("api/circuit.md", "Enum of supported branch kinds. Tutorial companion: [`04-circuit-quantization.ipynb`](./notebooks/04-circuit-quantization.ipynb)."),
    "CJ_branch": ("api/circuit.md", "Capacitive-JJ branch tag included in the circuit graph helpers."),
    "C_branch": ("api/circuit.md", "Capacitive branch tag used in circuit graph descriptions."),
    "Circuit": ("api/circuit.md", "Main user-facing circuit quantization type."),
    "CircuitGraph": ("api/circuit.md", "Graph-level circuit representation used before symbolic analysis."),
    "JJ_branch": ("api/circuit.md", "Josephson-junction branch tag used in circuit graph descriptions."),
    "L_branch": ("api/circuit.md", "Inductive branch tag used in circuit graph descriptions."),
    "SymbolicCircuit": ("api/symbolic-circuit.md", "Symbolic circuit-analysis result type."),
    "VarCategories": ("api/symbolic-circuit.md", "Mode classification result from symbolic mode decomposition."),
    "build_symbolic_circuit": ("api/symbolic-circuit.md", "Direct symbolic-analysis helper from a parsed circuit graph."),
    "compute_variable_transformation": ("api/symbolic-circuit.md", "Mode decomposition helper for symbolic circuits."),
    "external_fluxes": ("api/circuit.md", "Returns the symbolic external-flux variables attached to a `Circuit`."),
    "find_closure_branches": ("api/circuit.md", "Circuit-topology helper for cotree branches."),
    "find_fundamental_loops": ("api/circuit.md", "Circuit-topology helper for branch-level loops."),
    "find_spanning_tree": ("api/circuit.md", "Circuit-topology helper for choosing a spanning tree."),
    "find_superconducting_loops": ("api/circuit.md", "Circuit-topology helper specialized to superconducting flux loops."),
    "get_param": ("api/circuit.md", "Generic circuit parameter getter for fluxes, offset charges, and branch parameters."),
    "invalidate_cache!": ("api/circuit.md", "Clears the cached numerical Hamiltonian and configured hierarchy state."),
    "list_branch_params": ("api/circuit.md", "Prints effective branch parameters and their current values."),
    "offset_charge_transformation": ("api/circuit.md", "Returns symbolic equations mapping periodic-mode offset charges to node-charge placeholders."),
    "offset_charges": ("api/circuit.md", "Returns the symbolic periodic-mode offset-charge labels for a `Circuit`."),
    "parse_circuit": ("api/circuit.md", "Parser for scqubits-style YAML circuit descriptions."),
    "set_external_flux!": ("api/circuit.md", "Sets one external flux value on a circuit."),
    "set_offset_charge!": ("api/circuit.md", "Sets one periodic-mode offset charge on a circuit."),
    "set_param!": ("api/circuit.md", "Unified circuit parameter setter used by sweeps and manual updates."),
    "sym_external_fluxes": ("api/circuit.md", "Maps symbolic flux variables to their closure branches and loops."),
    "sym_hamiltonian": ("api/circuit.md", "Formatted symbolic Hamiltonian accessor; subsystem mode is covered on the hierarchy page."),
    "sym_hamiltonian_node": ("api/circuit.md", "Returns the symbolic Hamiltonian in node variables."),
    "sym_lagrangian": ("api/circuit.md", "Returns the symbolic Lagrangian in node or transformed variables."),
    "variable_transformation": ("api/circuit.md", "Convenience accessor for the stored mode transformation and categories."),
    "HilbertSpace": ("notebooks/05-hilbertspace-and-lookup.ipynb", "Composite-system container."),
    "InteractionTerm": ("notebooks/05-hilbertspace-and-lookup.ipynb", "Stored interaction descriptor used by `HilbertSpace`."),
    "OVERLAP_THRESHOLD": ("notebooks/05-hilbertspace-and-lookup.ipynb", "Default overlap threshold for bare/dressed labeling."),
    "SpectrumLookup": ("notebooks/05-hilbertspace-and-lookup.ipynb", "Bare/dressed lookup container."),
    "add_interaction!": ("notebooks/05-hilbertspace-and-lookup.ipynb", "Adds operator-factorized interaction terms."),
    "add_operator!": ("notebooks/05-hilbertspace-and-lookup.ipynb", "Adds pre-built operators directly to a `HilbertSpace`."),
    "bare_index": ("notebooks/05-hilbertspace-and-lookup.ipynb", "Lookup from dressed to bare labels."),
    "diag!": ("notebooks/05-hilbertspace-and-lookup.ipynb", "Convenience diagonalization helper."),
    "dressed_index": ("notebooks/05-hilbertspace-and-lookup.ipynb", "Lookup from bare to dressed labels."),
    "energy_by_bare_index": ("notebooks/05-hilbertspace-and-lookup.ipynb", "Bare-label energy accessor."),
    "energy_by_dressed_index": ("notebooks/05-hilbertspace-and-lookup.ipynb", "Dressed-index energy accessor."),
    "generate_lookup!": ("notebooks/05-hilbertspace-and-lookup.ipynb", "Builds lookup tables on `HilbertSpace` or `ParameterSweep`."),
    "identity_wrap": ("notebooks/05-hilbertspace-and-lookup.ipynb", "Embeds subsystem operators into the full tensor-product space."),
    "lookup_exists": ("notebooks/05-hilbertspace-and-lookup.ipynb", "Checks whether lookup data have already been built."),
    "op_in_dressed_eigenbasis": ("notebooks/05-hilbertspace-and-lookup.ipynb", "Projects operators into the dressed eigenbasis."),
    "ParameterSweep": ("notebooks/06-sweeps-and-dispersive.ipynb", "Coupled-system sweep API."),
    "SingleSystemSweep": ("notebooks/06-sweeps-and-dispersive.ipynb", "Single-system sweep helper."),
    "chi_matrix": ("notebooks/06-sweeps-and-dispersive.ipynb", "Cross-Kerr/dispersive summary from lookups."),
    "lamb_shift": ("notebooks/06-sweeps-and-dispersive.ipynb", "Lamb-shift extraction from lookup data."),
    "plot_chi_vs_paramvals": ("notebooks/06-sweeps-and-dispersive.ipynb", "Makie helper for plotting dispersive shifts across a sweep."),
    "plot_evals_vs_paramvals": ("notebooks/06-sweeps-and-dispersive.ipynb", "Makie helper for plotting spectra from sweeps."),
    "run!": ("notebooks/06-sweeps-and-dispersive.ipynb", "Explicit sweep execution for `ParameterSweep`."),
    "self_kerr": ("notebooks/06-sweeps-and-dispersive.ipynb", "Self-Kerr/anharmonicity extraction from lookup data."),
    "HierarchyGroup": ("api/circuit-hierarchy.md", "Typed nested-group hierarchy node. Tutorial companion: [`07-advanced-circuit-hierarchy.ipynb`](./notebooks/07-advanced-circuit-hierarchy.ipynb)."),
    "HierarchyLeaf": ("api/circuit-hierarchy.md", "Typed leaf hierarchy node. Tutorial companion: [`07-advanced-circuit-hierarchy.ipynb`](./notebooks/07-advanced-circuit-hierarchy.ipynb)."),
    "HierarchyNode": ("api/circuit-hierarchy.md", "Union alias for hierarchy-node types."),
    "SubCircuit": ("api/circuit-hierarchy.md", "Subsystem object created by hierarchical diagonalization."),
    "configure!": ("api/circuit-hierarchy.md", "Stores hierarchy state on a `Circuit` and computes subsystem decompositions."),
    "hierarchical_diag": ("api/circuit-hierarchy.md", "One-shot hierarchical diagonalization helper."),
    "sym_interaction": ("api/circuit-hierarchy.md", "Symbolic interaction decomposition between configured subsystems."),
    "truncation_template": ("api/circuit-hierarchy.md", "Helper for scqubits-style hierarchy truncation templates."),
}


APPENDIX_ONLY = {
    "Low-Level Basis Operators": [
        "cos_phi_operator_grid",
        "cos_phi_operator_ho",
        "cos_theta_operator",
        "d2_dphi2_operator_grid",
        "d_dphi_operator_grid",
        "exp_i_theta_operator",
        "n_operator_ho",
        "n_operator_periodic",
        "phi_operator_grid",
        "phi_operator_ho",
        "sin_phi_operator_grid",
        "sin_phi_operator_ho",
        "sin_theta_operator",
    ],
}


def build_notebooks() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    write_nb(
        NOTEBOOKS_DIR / "01-core-api.ipynb",
        "Core API",
        [
            md(
                """
                # Core API

                This notebook introduces the shared conventions used across `ScQubitsMimic.jl`: energy units, grid helpers, simple systems, and the spectral routines that all higher-level objects build on.
                """
            ),
            md(
                """
                **Audience:** readers who want a compact tour of the common API before working with qubits or circuits.

                **Prerequisites:** Julia basics and a rough idea of Hamiltonians/eigenvalues.

                **Learning goals:** after this notebook you should be able to construct simple systems, inspect spectra, compute matrix elements, and interpret the package's shared return types.
                """
            ),
            md(
                """
                ## Outline

                1. Inspect physical constants and convert between common energy units.
                2. Build a `Grid1d` and inspect its sampling helpers.
                3. Use `GenericQubit`, `Oscillator`, and `KerrOscillator` as minimal examples.
                4. Apply the common spectral routines: `hamiltonian`, `hilbertdim`, `eigenvals`, `eigensys`, `matrixelement`, `matrixelement_table`, and `get_spectrum_vs_paramvals`.
                """
            ),
            code(
                """
                using ScQubitsMimic
                using LinearAlgebra
                """
            ),
            code(
                """
                (
                    elementary_charge_C = PhysicalConstants.e,
                    flux_quantum_Wb = PhysicalConstants.Phi_0,
                    five_GHz_in_K = round(convert_units(5.0, :GHz, :K), digits=6),
                    five_GHz_in_eV = round(convert_units(5.0, :GHz, :eV), sigdigits=6),
                )
                """
            ),
            code(
                """
                grid = Grid1d(-π, π, 9)
                (
                    npoints = grid.npoints,
                    spacing = round(grid_spacing(grid), digits=6),
                    first_three_points = round.(collect(grid_points(grid))[1:3], digits=4),
                    last_three_points = round.(collect(grid_points(grid))[end-2:end], digits=4),
                )
                """
            ),
            code(
                """
                generic = GenericQubit(E=5.0)
                osc = Oscillator(E_osc=6.0, truncated_dim=8)
                kerr = KerrOscillator(E_osc=6.0, K=0.2, truncated_dim=8)

                (
                    generic_evals = round.(eigenvals(generic; evals_count=2), digits=6),
                    oscillator_evals = round.(eigenvals(osc; evals_count=5), digits=6),
                    kerr_evals = round.(eigenvals(kerr; evals_count=5), digits=6),
                )
                """
            ),
            code(
                """
                H = hamiltonian(osc)
                vals, vecs = eigensys(osc; evals_count=4)
                n_op = number_operator(osc)

                (
                    hilbert_dimension = hilbertdim(osc),
                    hamiltonian_shape = size(H.data),
                    first_four_evals = round.(vals, digits=6),
                    eigvec_shape = size(vecs),
                    n01 = round(matrixelement(osc, n_op, 1, 2), digits=6),
                    n_table = round.(matrixelement_table(osc, n_op; evals_count=4), digits=3),
                )
                """
            ),
            code(
                """
                sd = get_spectrum_vs_paramvals(generic, :E, [2.0, 3.0, 4.0, 5.0]; evals_count=2)
                (
                    param_vals = sd.param_vals,
                    eigenvalue_table = round.(sd.eigenvalues, digits=4),
                )
                """
            ),
            md(
                """
                ## Exercise

                Change the oscillator frequency from `6.0` GHz to `4.5` GHz and verify that the first three eigenvalues still follow the equally spaced harmonic sequence.
                """
            ),
            code(
                """
                osc_exercise = Oscillator(E_osc=4.5, truncated_dim=6)
                round.(eigenvals(osc_exercise; evals_count=3), digits=6)
                """
            ),
            md(
                """
                ## Pitfalls And Extensions

                A `Grid1d` is only a sampling helper; it does not quantize a system by itself. For simple qubit and oscillator examples the basis is built into the system type.

                The shared spectral routines use Julia's 1-based indexing. A matrix element such as `matrixelement(sys, op, 1, 2)` refers to the ground-to-first-excited transition.
                """
            ),
            md(
                """
                ## API Covered

                `ScQubitsMimic`, `PhysicalConstants`, `convert_units`, `Grid1d`, `grid_points`, `grid_spacing`, `SpectrumData`, `GenericQubit`, `Oscillator`, `KerrOscillator`, `annihilation_operator`, `creation_operator`, `number_operator`, `hamiltonian`, `hilbertdim`, `eigenvals`, `eigensys`, `matrixelement`, `matrixelement_table`, `get_spectrum_vs_paramvals`, and the exported abstract system types.
                """
            ),
        ],
    )

    write_nb(
        NOTEBOOKS_DIR / "02-transmon.ipynb",
        "Transmon",
        [
            md(
                """
                # Transmon

                This notebook documents the fixed-frequency `Transmon` workflow in `ScQubitsMimic.jl`, focusing on construction, operators, spectrum trends, matrix elements, and wavefunction interpretation.
                """
            ),
            md(
                """
                **Audience:** readers already comfortable with the shared API from notebook 01.

                **Prerequisites:** basic transmon physics (`EJ`, `EC`, and offset charge `ng`).

                **Learning goals:** create a `Transmon`, inspect charge-basis operators, sweep `ng` and `EJ`, and visualize matrix elements and wavefunctions with the Makie extension.
                """
            ),
            md(
                """
                ## Outline

                1. Build a transmon and compare `ω01` against the transmon-limit approximation.
                2. Inspect the exported charge-basis operators.
                3. Sweep `ng` and `EJ` to see charge dispersion and frequency scaling.
                4. Plot wavefunctions and matrix elements using the extension API.
                """
            ),
            code(
                """
                using ScQubitsMimic
                using CairoMakie
                """
            ),
            code(
                """
                tmon = Transmon(EJ=30.0, EC=1.2, ng=0.0, ncut=30, truncated_dim=6)
                evals, _ = eigensys(tmon; evals_count=6)
                ω01 = evals[2] - evals[1]
                ω12 = evals[3] - evals[2]
                ω01_approx = sqrt(8 * tmon.EJ * tmon.EC) - tmon.EC

                (
                    EJ_over_EC = round(tmon.EJ / tmon.EC, digits=3),
                    first_six_levels = round.(evals, digits=6),
                    ω01_GHz = round(ω01, digits=6),
                    anharmonicity_GHz = round(ω12 - ω01, digits=6),
                    transmon_limit_estimate_GHz = round(ω01_approx, digits=6),
                    approximation_error_percent = round(abs(ω01 - ω01_approx) / ω01 * 100, digits=3),
                )
                """
            ),
            code(
                """
                ϕ_vals = range(-π, π; length=5)
                (
                    n_shape = size(n_operator(tmon).data),
                    exp_iϕ_shape = size(exp_i_phi_operator(tmon).data),
                    cosϕ_shape = size(cos_phi_operator(tmon).data),
                    sinϕ_shape = size(sin_phi_operator(tmon).data),
                    potential_samples = round.(potential.(Ref(tmon), ϕ_vals), digits=4),
                )
                """
            ),
            code(
                """
                ng_vals = collect(range(-0.5, 0.5; length=9))
                ng_spectrum = get_spectrum_vs_paramvals(tmon, :ng, ng_vals; evals_count=3)

                (
                    ng_points = ng_spectrum.param_vals,
                    ω01_vs_ng = round.(ng_spectrum.eigenvalues[:, 2] .- ng_spectrum.eigenvalues[:, 1], digits=6),
                )
                """
            ),
            code(
                """
                ej_vals = collect(range(10.0, 50.0; length=7))
                ej_spectrum = get_spectrum_vs_paramvals(tmon, :EJ, ej_vals; evals_count=3)

                (
                    EJ_points = ej_spectrum.param_vals,
                    ω01_vs_EJ = round.(ej_spectrum.eigenvalues[:, 2] .- ej_spectrum.eigenvalues[:, 1], digits=6),
                )
                """
            ),
            code(
                """
                round.(matrixelement_table(tmon, n_operator(tmon); evals_count=4), digits=6)
                """
            ),
            code(
                """
                plot_wavefunction(tmon, [1, 2, 3]; mode=:abs_sqr)
                """
            ),
            code(
                """
                plot_matrixelements(tmon, s -> n_operator(s); evals_count=8, mode=:abs)
                """
            ),
            md(
                """
                ## Exercise

                Lower `EJ` to `15.0` GHz while keeping `EC=1.2` GHz. Recompute `ω01` and compare the new anharmonicity against `-EC`.
                """
            ),
            code(
                """
                tmon_exercise = Transmon(EJ=15.0, EC=1.2, ng=0.0, ncut=30, truncated_dim=4)
                evals_exercise = eigenvals(tmon_exercise; evals_count=3)
                (
                    ω01_GHz = round(evals_exercise[2] - evals_exercise[1], digits=6),
                    anharmonicity_GHz = round((evals_exercise[3] - evals_exercise[2]) - (evals_exercise[2] - evals_exercise[1]), digits=6),
                )
                """
            ),
            md(
                """
                ## Pitfalls And Extensions

                `plot_wavefunction` uses eigenstate indices in Julia's 1-based convention: `1` is the ground state, `2` is the first excited state, and so on.

                The built-in plotting API currently visualizes the charge-basis wavefunction directly. If you need phase-basis reconstructions beyond the exported API, keep that logic notebook-local and label it clearly as a derived analysis rather than a package function.
                """
            ),
            md(
                """
                ## API Covered

                `Transmon`, `n_operator`, `exp_i_phi_operator`, `cos_phi_operator`, `sin_phi_operator`, `potential`, `plot_wavefunction`, and `plot_matrixelements`.
                """
            ),
        ],
    )

    write_nb(
        NOTEBOOKS_DIR / "03-tunable-transmon.ipynb",
        "Tunable Transmon",
        [
            md(
                """
                # Tunable Transmon

                This notebook covers the flux-tunable `TunableTransmon` API and compares it with an equivalent SQUID built from the generic circuit pipeline.
                """
            ),
            md(
                """
                **Audience:** readers familiar with the fixed-frequency transmon notebook.

                **Prerequisites:** the meaning of `EJmax`, asymmetry `d`, and external flux in units of `Φ₀`.

                **Learning goals:** compute `EJ_eff(Φ)`, inspect flux dependence in the spectrum, compare symmetric and asymmetric devices, and validate against a `Circuit`-derived SQUID.
                """
            ),
            md(
                """
                ## Outline

                1. Construct a `TunableTransmon` and inspect `ej_effective`.
                2. Sweep flux through the symmetric and asymmetric SQUID cases.
                3. Compare the hardcoded model with a circuit-derived SQUID.
                """
            ),
            code(
                """
                using ScQubitsMimic
                """
            ),
            code(
                """
                tt = TunableTransmon(EJmax=15.0, EC=0.6, d=0.1, flux=0.0, ng=0.0, ncut=30, truncated_dim=6)
                flux_vals = [0.0, 0.1, 0.25, 0.5]

                (
                    parameters = (EJmax=tt.EJmax, EC=tt.EC, d=tt.d, ncut=tt.ncut),
                    EJ_eff_vs_flux = [(ϕ, round((tt.flux = ϕ; ej_effective(tt)), digits=6)) for ϕ in flux_vals],
                )
                """
            ),
            code(
                """
                tt.flux = 0.0
                sd_flux = get_spectrum_vs_paramvals(tt, :flux, collect(range(0.0, 0.5; length=7)); evals_count=3)
                (
                    flux_points = sd_flux.param_vals,
                    ω01_vs_flux = round.(sd_flux.eigenvalues[:, 2] .- sd_flux.eigenvalues[:, 1], digits=6),
                    ω12_vs_flux = round.(sd_flux.eigenvalues[:, 3] .- sd_flux.eigenvalues[:, 2], digits=6),
                )
                """
            ),
            code(
                """
                tt_sym = TunableTransmon(EJmax=15.0, EC=0.6, d=0.0, flux=0.5, ncut=30, truncated_dim=4)
                tt_asym = TunableTransmon(EJmax=15.0, EC=0.6, d=0.1, flux=0.5, ncut=30, truncated_dim=4)

                (
                    symmetric_EJ_eff = round(ej_effective(tt_sym), digits=8),
                    asymmetric_EJ_eff = round(ej_effective(tt_asym), digits=8),
                    symmetric_ω01 = round(diff(eigenvals(tt_sym; evals_count=2))[1], digits=6),
                    asymmetric_ω01 = round(diff(eigenvals(tt_asym; evals_count=2))[1], digits=6),
                )
                """
            ),
            code(
                '''
                desc = """
                branches:
                  - [JJ, 0, 1, EJ=7.5, EC=0.3]
                  - [JJ, 0, 1, EJ=7.5, EC=0.3]
                """

                circ = Circuit(desc; ncut=30)
                comparison = []
                for flux in (0.0, 0.25)
                    tt_cmp = TunableTransmon(EJmax=15.0, EC=0.6, d=0.0, flux=flux, ncut=30, truncated_dim=3)
                    set_external_flux!(circ, 1, 2π * flux)
                    push!(comparison, (
                        flux = flux,
                        tunable_ω01 = round(diff(eigenvals(tt_cmp; evals_count=2))[1], digits=6),
                        circuit_ω01 = round(diff(eigenvals(circ; evals_count=2))[1], digits=6),
                    ))
                end
                comparison
                '''
            ),
            md(
                """
                ## Exercise

                Set `d=0.2` and evaluate the device at `flux=0.5`. How much residual `EJ_eff` remains compared with the symmetric device?
                """
            ),
            code(
                """
                tt_exercise = TunableTransmon(EJmax=15.0, EC=0.6, d=0.2, flux=0.5, ncut=30, truncated_dim=3)
                round(ej_effective(tt_exercise), digits=6)
                """
            ),
            md(
                """
                ## Pitfalls And Extensions

                `TunableTransmon.flux` is expressed in units of `Φ₀`, while the circuit pipeline uses external flux variables in radians. That is why the parity comparison multiplies by `2π` before calling `set_external_flux!`.

                The hardcoded and circuit-derived models agree most closely when the circuit capacitance bookkeeping is chosen to reproduce the same effective charging energy.
                """
            ),
            md(
                """
                ## API Covered

                `TunableTransmon` and `ej_effective`.
                """
            ),
        ],
    )

    write_nb(
        NOTEBOOKS_DIR / "04-circuit-quantization.ipynb",
        "Circuit Quantization",
        [
            md(
                """
                # Circuit Quantization

                This notebook documents the graph → symbolic circuit → mode decomposition → numerical quantization workflow exposed by `Circuit` and its supporting APIs.
                """
            ),
            md(
                """
                **Audience:** readers who want the Julia equivalent of the scqubits custom-circuit path.

                **Prerequisites:** familiarity with branch-based superconducting circuit descriptions and symbolic Hamiltonians.

                **Learning goals:** parse a circuit, inspect its topology, build a `SymbolicCircuit`, construct a quantized `Circuit`, inspect symbolic expressions, and sweep flux or branch parameters.
                """
            ),
            md(
                """
                ## Outline

                1. Parse a YAML-style circuit description into a `CircuitGraph`.
                2. Inspect spanning trees, closure branches, and superconducting loops.
                3. Build the symbolic circuit and examine the variable transformation.
                4. Construct a quantized `Circuit`, inspect symbolic accessors, and update parameters.
                """
            ),
            code(
                '''
                using ScQubitsMimic

                desc = """
                branches:
                  - [JJ, 0, 1, EJ=10.0, EC=0.3]
                  - [JJ, 0, 1, EJ=10.0, EC=0.3]
                  - [C, 0, 1, EC=0.5]
                """
                '''
            ),
            code(
                """
                cg = parse_circuit(desc)
                (
                    branch_types = [branch.branch_type for branch in cg.branches],
                    num_nodes = cg.num_nodes,
                    has_ground = cg.has_ground,
                )
                """
            ),
            code(
                """
                tree = find_spanning_tree(cg)
                closure = find_closure_branches(cg, tree)
                sc_closure, sc_loops = find_superconducting_loops(cg)

                (
                    spanning_tree = tree,
                    closure_branches = closure,
                    superconducting_closure_branches = sc_closure,
                    superconducting_loops = sc_loops,
                    fundamental_loops = find_fundamental_loops(cg, tree),
                )
                """
            ),
            code(
                """
                sc = build_symbolic_circuit(cg)
                T, categories = compute_variable_transformation(sc)
                (
                    symbolic_fluxes = string.(sc.external_fluxes),
                    symbolic_offset_charges = string.(sc.offset_charges),
                    transform = round.(T, digits=3),
                    categories = categories,
                )
                """
            ),
            code(
                """
                circ = Circuit(desc; ncut=30)
                (
                    mode_categories = circ.var_categories,
                    cutoffs = circ.cutoffs,
                    symbolic_hamiltonian = sym_hamiltonian(circ; return_expr=true),
                    symbolic_hamiltonian_node = sym_hamiltonian_node(circ),
                    symbolic_lagrangian_node = sym_lagrangian(circ),
                    symbolic_lagrangian_mode = sym_lagrangian(circ; vars_type=:new),
                )
                """
            ),
            code(
                """
                (
                    transformation = round.(first(variable_transformation(circ)), digits=3),
                    offset_charge_map = offset_charge_transformation(circ),
                    loop_to_flux_map = sym_external_fluxes(circ),
                    symbolic_offset_charge_labels = string.(offset_charges(circ)),
                )
                """
            ),
            code(
                """
                set_external_flux!(circ, 1, π / 2)
                set_offset_charge!(circ, 1, 0.125)
                before_override = (
                    Φ1 = get_param(circ, Symbol("Φ1")),
                    ng1 = get_param(circ, :ng1),
                    EJ_branch_1 = get_param(circ, :EJ_1),
                )

                set_param!(circ, :EJ_1, 12.0)
                after_override = (
                    Φ1 = get_param(circ, Symbol("Φ1")),
                    ng1 = get_param(circ, :ng1),
                    EJ_branch_1 = get_param(circ, :EJ_1),
                    listed_params = list_branch_params(circ),
                )

                (before_override = before_override, after_override = after_override)
                """
            ),
            code(
                """
                sd = get_spectrum_vs_paramvals(circ, Symbol("Φ1"), collect(range(0.0, π; length=5)); evals_count=3)
                (
                    Φ1_points = round.(sd.param_vals, digits=4),
                    ω01_vs_flux = round.(sd.eigenvalues[:, 2] .- sd.eigenvalues[:, 1], digits=6),
                )
                """
            ),
            code(
                """
                invalidate_cache!(circ)
                typeof(hamiltonian(circ))
                """
            ),
            md(
                """
                ## Exercise

                Add a shunt capacitor branch `[C, 0, 1, EC=1.0]` to the circuit description and compare the resulting `ω01` against the unshunted device.
                """
            ),
            code(
                '''
                desc_exercise = """
                branches:
                  - [JJ, 0, 1, EJ=10.0, EC=0.3]
                  - [JJ, 0, 1, EJ=10.0, EC=0.3]
                  - [C, 0, 1, EC=0.5]
                  - [C, 0, 1, EC=1.0]
                """
                circ_exercise = Circuit(desc_exercise; ncut=30)
                round(diff(eigenvals(circ_exercise; evals_count=2))[1], digits=6)
                '''
            ),
            md(
                """
                ## Pitfalls And Extensions

                Circuit external flux symbols such as `Φ1` live in radians, not in units of `Φ₀`. This differs from the hardcoded `TunableTransmon` API.

                `invalidate_cache!` clears the numerical Hamiltonian cache. If hierarchical diagonalization has already been configured, the symbolic decomposition remains available while the numerical hierarchy is rebuilt on demand.
                """
            ),
            md(
                """
                ## API Covered

                `BranchType`, `C_branch`, `L_branch`, `JJ_branch`, `CJ_branch`, `Branch`, `CircuitGraph`, `parse_circuit`, `find_spanning_tree`, `find_closure_branches`, `find_fundamental_loops`, `find_superconducting_loops`, `SymbolicCircuit`, `build_symbolic_circuit`, `compute_variable_transformation`, `VarCategories`, `Circuit`, `variable_transformation`, `offset_charge_transformation`, `external_fluxes`, `sym_external_fluxes`, `offset_charges`, `sym_hamiltonian`, `sym_hamiltonian_node`, `sym_lagrangian`, `set_external_flux!`, `set_offset_charge!`, `set_param!`, `get_param`, `list_branch_params`, and `invalidate_cache!`.
                """
            ),
        ],
    )

    write_nb(
        NOTEBOOKS_DIR / "05-hilbertspace-and-lookup.ipynb",
        "HilbertSpace And Lookup",
        [
            md(
                """
                # HilbertSpace And Lookup

                This notebook documents how `HilbertSpace` combines independently defined subsystems, how interactions are added, and how bare/dressed lookup tables are built and queried.
                """
            ),
            md(
                """
                **Audience:** readers ready to move from single systems to coupled systems.

                **Prerequisites:** the shared API from notebook 01 and the meaning of dressed versus bare states.

                **Learning goals:** assemble a composite Hamiltonian, add interactions and direct operator terms, diagonalize the coupled system, build a `SpectrumLookup`, and move operators into the dressed basis.
                """
            ),
            md(
                """
                ## Outline

                1. Build a coupled pair of Kerr oscillators.
                2. Add interaction terms and an extra operator directly to the full Hilbert space.
                3. Diagonalize the system and generate bare/dressed lookup data.
                4. Transform an operator into the dressed basis.
                """
            ),
            code(
                """
                using ScQubitsMimic
                using LinearAlgebra
                """
            ),
            code(
                """
                kosc1 = KerrOscillator(E_osc=4.5, K=0.05, truncated_dim=6)
                kosc2 = KerrOscillator(E_osc=6.0, K=0.03, truncated_dim=6)
                hs = HilbertSpace([kosc1, kosc2])
                dims = [hilbertdim(s) for s in hs.subsystems]

                bare_summary = (
                    subsystem_dims = dims,
                    bare_dimension = hilbertdim(hs),
                    bare_evals = round.(eigenvals(hs; evals_count=6), digits=6),
                )
                bare_summary
                """
            ),
            code(
                """
                g = 0.1
                add_interaction!(hs, g, [kosc1, kosc2], [s -> creation_operator(s), s -> annihilation_operator(s)])
                add_interaction!(hs, g, [kosc1, kosc2], [s -> annihilation_operator(s), s -> creation_operator(s)])

                detuning_term = 0.02 * identity_wrap(number_operator(kosc2), 2, dims)
                add_operator!(hs, detuning_term)

                (
                    interaction_count = length(hs.interactions),
                    extra_operator_count = length(hs.extra_H_terms),
                    dressed_evals = round.(diag!(hs; evals_count=8)[1], digits=6),
                )
                """
            ),
            code(
                """
                lookup = generate_lookup!(hs; evals_count=12)
                (
                    lookup_built = lookup_exists(hs),
                    overlap_threshold = OVERLAP_THRESHOLD,
                    dressed_1_to_bare = bare_index(hs, 1),
                    bare_2_1_to_dressed = dressed_index(hs, 2, 1),
                    energy_dressed_2 = round(energy_by_dressed_index(hs, 2), digits=6),
                    energy_bare_1_2 = round(energy_by_bare_index(hs, 1, 2), digits=6),
                )
                """
            ),
            code(
                """
                n1_full = identity_wrap(number_operator(kosc1), 1, dims)
                n1_dressed = op_in_dressed_eigenbasis(hs, n1_full; truncated_dim=5)
                round.(n1_dressed, digits=4)
                """
            ),
            md(
                """
                ## Exercise

                Replace the two Kerr oscillators with a `Transmon` and an `Oscillator`. Rebuild the lookup and identify which dressed state most closely matches bare state `(2, 1)`.
                """
            ),
            code(
                """
                tmon_exercise = Transmon(EJ=30.0, EC=1.2, ng=0.0, ncut=12, truncated_dim=4)
                osc_exercise = Oscillator(E_osc=7.0, truncated_dim=6)
                hs_exercise = HilbertSpace([tmon_exercise, osc_exercise])
                add_interaction!(hs_exercise, 0.1, [tmon_exercise, osc_exercise],
                    [s -> n_operator(s), s -> annihilation_operator(s) + creation_operator(s)])
                generate_lookup!(hs_exercise; evals_count=12)
                dressed_index(hs_exercise, 2, 1)
                """
            ),
            md(
                """
                ## Pitfalls And Extensions

                `generate_lookup!` needs enough dressed states to label the bare states you care about. If a lookup query fails, raise `evals_count` and rebuild the lookup.

                `identity_wrap` is useful whenever you want to add a hand-built operator to the composite system. The helper follows the subsystem order stored on the `HilbertSpace`.
                """
            ),
            md(
                """
                ## API Covered

                `HilbertSpace`, `InteractionTerm`, `SpectrumLookup`, `OVERLAP_THRESHOLD`, `add_interaction!`, `add_operator!`, `diag!`, `identity_wrap`, `generate_lookup!`, `lookup_exists`, `bare_index`, `dressed_index`, `energy_by_dressed_index`, `energy_by_bare_index`, and `op_in_dressed_eigenbasis`.
                """
            ),
        ],
    )

    write_nb(
        NOTEBOOKS_DIR / "06-sweeps-and-dispersive.ipynb",
        "Sweeps And Dispersive",
        [
            md(
                """
                # Sweeps And Dispersive

                This notebook documents the sweep APIs for both single systems and coupled systems, then uses lookup-aware post-processing to extract dispersive information.
                """
            ),
            md(
                """
                **Audience:** readers who want to study parameter-dependent spectra and dispersive trends.

                **Prerequisites:** the `HilbertSpace` and lookup workflow from notebook 05.

                **Learning goals:** use `SingleSystemSweep`, construct a `ParameterSweep`, run it explicitly, build lookup-aware summaries, and use the Makie sweep-plotting helpers.
                """
            ),
            md(
                """
                ## Outline

                1. Sweep a single `TunableTransmon` with `SingleSystemSweep`.
                2. Build a coupled-system `ParameterSweep` with explicit `run!`.
                3. Query dressed energies and bare/dressed labels at individual parameter points.
                4. Compute `chi_matrix`, `self_kerr`, and `lamb_shift`, and plot sweep summaries.
                """
            ),
            code(
                """
                using ScQubitsMimic
                using CairoMakie
                """
            ),
            code(
                """
                tmon = TunableTransmon(EJmax=20.0, EC=0.3, d=0.1, flux=0.0, ng=0.0, ncut=15, truncated_dim=4)
                single_sweep = SingleSystemSweep(tmon, :flux, collect(range(0.0, 0.5; length=7)); evals_count=4)

                (
                    param_name = single_sweep.param_name,
                    param_vals = single_sweep.param_vals,
                    ω01_vs_flux = round.(single_sweep.spectrum.eigenvalues[:, 2] .- single_sweep.spectrum.eigenvalues[:, 1], digits=6),
                )
                """
            ),
            code(
                """
                plot_evals_vs_paramvals(single_sweep; subtract_ground=true, evals_count=4)
                """
            ),
            code(
                """
                coupled_tmon = TunableTransmon(EJmax=20.0, EC=0.3, d=0.1, flux=0.0, ng=0.0, ncut=10, truncated_dim=5)
                resonator = Oscillator(E_osc=5.5, truncated_dim=8)
                hs = HilbertSpace([coupled_tmon, resonator])
                add_interaction!(hs, 0.05, [coupled_tmon, resonator],
                    [s -> n_operator(s), s -> annihilation_operator(s) + creation_operator(s)])

                sweep = ParameterSweep(
                    hs,
                    Dict(:flux => collect(range(0.0, 0.3; length=4))),
                    (hs, vals) -> begin
                        hs.subsystems[1].flux = vals[:flux]
                    end;
                    evals_count=20,
                    store_lookups=true,
                    ignore_low_overlap=true,
                    autorun=false,
                )

                run!(sweep)
                (
                    param_order = sweep.param_order,
                    dressed_shape = size(sweep.dressed_evals),
                    lookup_built = lookup_exists(sweep),
                )
                """
            ),
            code(
                """
                (
                    point_2_bare_of_dressed_3 = bare_index(sweep, 3; param_indices=(2,)),
                    point_2_dressed_of_bare_2_1 = dressed_index(sweep, (2, 1); param_indices=(2,)),
                    point_2_energy_of_dressed_3 = round(energy_by_dressed_index(sweep, 3; param_indices=(2,)), digits=6),
                    point_2_energy_of_bare_2_1 = round(energy_by_bare_index(sweep, (2, 1); param_indices=(2,)), digits=6),
                )
                """
            ),
            code(
                """
                chi = chi_matrix(sweep)
                (
                    chi_shape = size(chi),
                    chi_12_MHz = round.(chi[:, 1, 2] .* 1000, digits=3),
                    self_kerr_q1_MHz = round.(self_kerr(sweep, 1) .* 1000, digits=3),
                    lamb_shift_q1_MHz = round.(lamb_shift(sweep, 1) .* 1000, digits=3),
                )
                """
            ),
            code(
                """
                plot_chi_vs_paramvals(sweep; subsys_pair=(1, 2))
                """
            ),
            md(
                """
                ## Exercise

                Increase the coupling from `0.05` to `0.08` GHz, rerun the sweep, and compare the new `χ12` values against the original sweep.
                """
            ),
            code(
                """
                hs_exercise = HilbertSpace([deepcopy(coupled_tmon), Oscillator(E_osc=5.5, truncated_dim=8)])
                add_interaction!(hs_exercise, 0.08, hs_exercise.subsystems,
                    [s -> n_operator(s), s -> annihilation_operator(s) + creation_operator(s)])

                sweep_exercise = ParameterSweep(
                    hs_exercise,
                    Dict(:flux => collect(range(0.0, 0.3; length=4))),
                    (hs, vals) -> begin
                        hs.subsystems[1].flux = vals[:flux]
                    end;
                    evals_count=20,
                    store_lookups=true,
                    ignore_low_overlap=true,
                )

                round.(chi_matrix(sweep_exercise)[:, 1, 2] .* 1000, digits=3)
                """
            ),
            md(
                """
                ## Pitfalls And Extensions

                `plot_chi_vs_paramvals` depends on `chi_matrix(sweep)`, which in turn needs lookup data rich enough to resolve the first two excitations of each subsystem. If the plot fails, increase `evals_count` and rebuild the sweep.

                `ParameterSweep` defaults to `autorun=true`. Using `autorun=false` plus `run!` is often clearer in documentation because it makes the execution boundary explicit.
                """
            ),
            md(
                """
                ## API Covered

                `SingleSystemSweep`, `ParameterSweep`, `run!`, `plot_evals_vs_paramvals`, `chi_matrix`, `self_kerr`, `lamb_shift`, and `plot_chi_vs_paramvals`.
                """
            ),
        ],
    )

    write_nb(
        NOTEBOOKS_DIR / "07-advanced-circuit-hierarchy.ipynb",
        "Advanced Circuit Hierarchy",
        [
            md(
                """
                # Advanced Circuit Hierarchy

                This notebook documents the hierarchical circuit APIs used to split a multi-mode circuit into diagonalized subsystems with symbolic interaction terms.
                """
            ),
            md(
                """
                **Audience:** readers already comfortable with the basic `Circuit` workflow.

                **Prerequisites:** symbolic circuit decomposition, subsystem truncation, and the `HilbertSpace` concepts from notebook 05.

                **Learning goals:** build flat and nested hierarchy specifications, generate truncation templates, use `hierarchical_diag` directly, and use `configure!` to cache subsystem decompositions and symbolic interaction terms on a circuit.
                """
            ),
            md(
                """
                ## Outline

                1. Build typed hierarchy objects with `HierarchyLeaf` and `HierarchyGroup`.
                2. Generate scqubits-style truncation templates.
                3. Compare direct `hierarchical_diag` against `configure!`.
                4. Inspect `SubCircuit` objects and symbolic subsystem interactions.
                """
            ),
            code(
                """
                using ScQubitsMimic

                typed_hierarchy = HierarchyGroup([
                    HierarchyGroup([HierarchyLeaf([1]), HierarchyLeaf([2])]),
                    HierarchyLeaf([3]),
                ])

                (
                    typed_hierarchy isa HierarchyNode,
                    truncation_template(typed_hierarchy),
                )
                """
            ),
            code(
                '''
                desc_two_mode = """
                branches:
                  - [JJ, 0, 1, EJ=10.0, EC=0.3]
                  - [JJ, 0, 2, EJ=8.0, EC=0.4]
                  - [C, 1, 2, EC=0.1]
                """

                circ_two_mode = Circuit(desc_two_mode; ncut=8)
                hs_hd = hierarchical_diag(circ_two_mode;
                    system_hierarchy=[[1], [2]],
                    subsystem_trunc_dims=[8, 8],
                )

                (
                    subsystem_types = [typeof(sub) for sub in hs_hd.subsystems],
                    subsystem_dims = [hilbertdim(sub) for sub in hs_hd.subsystems],
                    dressed_evals = round.(eigenvals(hs_hd; evals_count=6), digits=6),
                )
                '''
            ),
            code(
                """
                configure!(circ_two_mode; system_hierarchy=[[1], [2]], subsystem_trunc_dims=[8, 8])
                (
                    hierarchical_flag = circ_two_mode._hierarchical_diagonalization,
                    stored_subsystems = [typeof(sub) for sub in circ_two_mode._subsystems],
                    subsystem_hamiltonian_1 = sym_hamiltonian(circ_two_mode; subsystem_index=1, return_expr=true),
                    subsystem_hamiltonian_2 = sym_hamiltonian(circ_two_mode; subsystem_index=2, return_expr=true),
                    interaction_12 = sym_interaction(circ_two_mode; subsystem_indices=(1, 2), return_expr=true),
                )
                """
            ),
            code(
                '''
                desc_nested = """
                branches:
                  - [JJ, 0, 1, EJ=10.0, EC=0.3]
                  - [JJ, 0, 2, EJ=8.0, EC=0.4]
                  - [JJ, 0, 3, EJ=6.0, EC=0.5]
                  - [C, 1, 2, EC=0.08]
                  - [C, 2, 3, EC=0.06]
                """

                circ_nested = Circuit(desc_nested; ncut=5)
                nested_hierarchy = [[[1], [2]], [3]]
                nested_truncation = truncation_template(nested_hierarchy; individual_trunc_dim=5, combined_trunc_dim=20)
                configure!(circ_nested; system_hierarchy=nested_hierarchy, subsystem_trunc_dims=nested_truncation)

                (
                    nested_truncation = nested_truncation,
                    top_level_subsystems = [typeof(sub) for sub in circ_nested._subsystems],
                    top_level_dims = [hilbertdim(sub) for sub in circ_nested._subsystems],
                    interaction_12 = sym_interaction(circ_nested; subsystem_indices=(1, 2), return_expr=true),
                )
                '''
            ),
            md(
                """
                ## Exercise

                Starting from the two-mode circuit, change the hierarchy to a single grouped subsystem `[[1, 2]]` and compare the returned Hilbert-space dimension with the two-subsystem decomposition.
                """
            ),
            code(
                """
                circ_exercise = Circuit(desc_two_mode; ncut=8)
                hs_single_group = hierarchical_diag(circ_exercise;
                    system_hierarchy=[[1, 2]],
                    subsystem_trunc_dims=[12],
                )
                [hilbertdim(sub) for sub in hs_single_group.subsystems]
                """
            ),
            md(
                """
                ## Pitfalls And Extensions

                `configure!` is strict about `subsystem_trunc_dims`: if you omit it, the call fails by design. Use `truncation_template(system_hierarchy)` as the starting point and then tighten the truncation manually.

                `SubCircuit` objects are produced by the hierarchy workflow; they are not meant to be constructed directly. Use `hierarchical_diag` for one-shot workflows and `configure!` when you want the circuit to keep cached subsystem state and symbolic decompositions.
                """
            ),
            md(
                """
                ## API Covered

                `HierarchyLeaf`, `HierarchyGroup`, `HierarchyNode`, `SubCircuit`, `truncation_template`, `hierarchical_diag`, `configure!`, and `sym_interaction`.
                """
            ),
        ],
    )


def main() -> None:
    build_notebooks()

    exported = export_names()
    appendix_symbols = {symbol for symbols in APPENDIX_ONLY.values() for symbol in symbols}
    covered_symbols = set(API_MAPPING) | appendix_symbols
    missing = [symbol for symbol in exported if symbol not in covered_symbols]
    extra = [symbol for symbol in covered_symbols if symbol not in exported]

    if missing:
        raise SystemExit(f"Missing exported symbols in API mapping: {missing}")
    if extra:
        raise SystemExit(f"Symbols in mapping but not exported: {extra}")

    (DOCS_SRC / "index.md").write_text(
        build_index(NOTEBOOK_SPECS, API_REFERENCE_SPECS),
        encoding="utf-8",
    )
    (DOCS_SRC / "api-coverage.md").write_text(
        build_api_coverage(API_MAPPING, APPENDIX_ONLY, COVERAGE_SECTION_SPECS),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
