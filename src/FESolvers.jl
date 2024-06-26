module FESolvers

import LinearAlgebra
export solve_problem!

export QuasiStaticSolver
export LinearProblemSolver
export NewtonSolver, SteepestDescent
export NoLineSearch, ArmijoGoldstein
export BackslashSolver, LinearSolveSolver
export FixedTimeStepper, AdaptiveTimeStepper

include("Convergence.jl")

include("FESolver.jl")

include("problem.jl")
include("linearsolvers.jl")
include("linesearchers.jl")

include("nlsolvers.jl")
include("nlsolvers/Newton.jl")
include("nlsolvers/AdaptiveNLSolver.jl")
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
function solve_problem!(problem, solver)
    try
        _solve_problem!(problem, solver)
    finally
        close_problem(problem)
    end
end


end