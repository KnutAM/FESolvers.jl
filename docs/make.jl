using FerriteSolvers
using Documenter

DocMeta.setdocmeta!(FerriteSolvers, :DocTestSetup, :(using FerriteSolvers); recursive=true)

makedocs(;
    modules=[FerriteSolvers],
    authors="Knut Andreas Meyer and contributors",
    repo="https://github.com/KnutAM/FerriteSolvers.jl/blob/{commit}{path}#{line}",
    sitename="FerriteSolvers.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://KnutAM.github.io/FerriteSolvers.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/KnutAM/FerriteSolvers.jl",
    devbranch="main",
)
