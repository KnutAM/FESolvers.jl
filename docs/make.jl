using FESolvers
using Documenter

const is_ci = get(ENV, "CI", "false") == "true"

include("generate.jl")
examples = ["plasticity.jl", "transient_heat.jl"]
GENERATEDEXAMPLES = [joinpath("examples", replace(f, ".jl"=>".md")) for f in examples]

build_examples(examples)

DocMeta.setdocmeta!(FESolvers, :DocTestSetup, :(using FESolvers); recursive=true)

makedocs(;
    modules=[FESolvers],
    authors="Knut Andreas Meyer and contributors",
    repo="https://github.com/KnutAM/FESolvers.jl/blob/{commit}{path}#{line}",
    sitename="FESolvers.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://KnutAM.github.io/FESolvers.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => [#"solvers.md",
                     "userfunctions.md",
                     "nlsolvers.md",
                     "timesteppers.md",
                     "linearsolvers.md"],
        "Examples" => GENERATEDEXAMPLES,   
    ],
)

deploydocs(;
    repo="github.com/KnutAM/FESolvers.jl",
    devbranch="main",
    push_preview=true,
)
