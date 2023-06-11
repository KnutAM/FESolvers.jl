module FESolvers
import LinearAlgebra
using Requires
export solve_problem!

export QuasiStaticSolver
export LinearProblemSolver
export NewtonSolver, SteepestDescent
export NoLineSearch, ArmijoGoldstein
export BackslashSolver
export FixedTimeStepper, AdaptiveTimeStepper

struct ConvergenceError <: Exception
    msg::String
end

abstract type FESolver end

include("problem.jl")
include("linearsolvers.jl")
include("linesearchers.jl")

include("nlsolvers.jl")
include("nlsolvers/Newton.jl")
include("nlsolvers/SteepestDescent.jl")
include("nlsolvers/AdaptiveNewton.jl")
include("nlsolvers/LinearProblemSolver.jl")

include("timesteppers.jl")
include("QuasiStaticSolver.jl")

"""
    solve_problem!(problem, solver)

Solve a given user `problem` using the chosen `solver`

For details on the functions that should be defined for `problem`,
see [User problem](@ref)
"""
function solve_problem! end

function __init__()
    @require(
        LinearSolve="7ed4a6bd-45f5-4d41-b270-4a48e9bafcae",
        (using .LinearSolve;
        function solve_linear!(Δx, K, r, alg::Union{LinearSolve.SciMLLinearSolveAlgorithm,Nothing})
            map!(-, Δx, solve(LinearProblem(K, copy!(Δx,r)), alg; alias_b=false).u)
        end)
    )
end

end