module FESolvers
import LinearAlgebra
export solve_problem!

export QuasiStaticSolver
export NewtonSolver, SteepestDescent
export NoLineSearch, ArmijoGoldstein
export BackslashSolver
export FixedTimeStepper, AdaptiveTimeStepper

struct ConvergenceError <: Exception
    msg::String
end

abstract type FESolver end

include("utils.jl")
include("userfunctions.jl")
include("linearsolvers.jl")
include("linesearchers.jl")
include("nlsolvers.jl")
include("timesteppers.jl")
include("QuasiStaticSolver.jl")

"""
    solve_problem!(solver, problem)

Solve a given user `problem` using the chosen `solver`

For details on the functions that should be defined for `problem`,
see [User problem](@ref)
"""
function solve_problem! end

end
