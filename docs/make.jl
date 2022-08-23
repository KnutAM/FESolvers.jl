using FerriteSolvers
using Documenter

const is_ci = get(ENV, "CI", "false") == "true"

include("generate.jl")
examples = ["plasticity.jl",]
GENERATEDEXAMPLES = [joinpath("examples", replace(f, ".jl"=>".md")) for f in examples]

build_examples(examples)

DocMeta.setdocmeta!(FerriteSolvers, :DocTestSetup, :(using FerriteSolvers); recursive=true)

makedocs(;
    modules=[FerriteSolvers],
    authors="Knut Andreas Meyer and Maximilian Köhler",
    repo="https://github.com/KnutAM/FerriteSolvers.jl/blob/{commit}{path}#{line}",
    sitename="FerriteSolvers.jl",
    format=Documenter.HTML(;
        prettyurls=is_ci,
        canonical="https://KnutAM.github.io/FerriteSolvers.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => ["userfunctions.md",
                     "nlsolvers.md",
                     "timesteppers.md",
                     "linearsolvers.md"],
        "Examples" => GENERATEDEXAMPLES,   
    ],
)

deploydocs(;
    repo="github.com/KnutAM/FerriteSolvers.jl",
    devbranch="main",
    push_preview=true,
)