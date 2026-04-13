using Pkg

Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()

using Documenter
using CairoMakie
using ScQubitsMimic

DocMeta.setdocmeta!(ScQubitsMimic, :DocTestSetup, :(using CairoMakie, ScQubitsMimic); recursive=true)

makedocs(
    sitename="ScQubitsMimic.jl",
    modules=[ScQubitsMimic],
    format=Documenter.HTML(
        prettyurls=false,
        collapselevel=1,
    ),
    checkdocs=:exports,
    warnonly=false,
    pages=[
        "Home" => "index.md",
        "Guides" => [
            "Overview" => "guides/index.md",
            "Core API" => "guides/core-api.md",
            "Transmon" => "guides/transmon.md",
            "Tunable Transmon" => "guides/tunable-transmon.md",
            "Circuit Quantization" => "guides/circuit-quantization.md",
            "HilbertSpace And Lookup" => "guides/hilbertspace-and-lookup.md",
            "Sweeps And Dispersive" => "guides/sweeps-and-dispersive.md",
            "Advanced Circuit Hierarchy" => "guides/advanced-circuit-hierarchy.md",
        ],
        "API Reference" => [
            "Overview" => "api/index.md",
            "Core" => "api/core.md",
            "Qubits" => "api/qubits.md",
            "Oscillators" => "api/oscillators.md",
            "Circuit" => "api/circuit.md",
            "Symbolic Circuit" => "api/symbolic-circuit.md",
            "Circuit Operators" => "api/circuit-operators.md",
            "Circuit Hierarchy" => "api/circuit-hierarchy.md",
            "HilbertSpace" => "api/hilbert-space.md",
            "Sweeps And Lookup" => "api/sweeps-and-lookup.md",
            "Analysis And Transitions" => "api/analysis-and-transitions.md",
            "Plotting" => "api/plotting.md",
        ],
        "API Audit" => "api-coverage.md",
    ],
)
