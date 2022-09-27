using FESolvers
using Documenter

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
        "Solvers" => "solvers.md",
        "User problem" => "userfunctions.md",
        "Nonlinear solvers" => "nlsolvers.md",
        "Time steppers" => "timesteppers.md",
        "Linear solvers" => "linearsolvers.md"
    ],
)

deploydocs(;
    repo="github.com/KnutAM/FESolvers.jl",
    devbranch="main",
    push_preview=true,
)
