module FerriteSolvers
import LinearAlgebra
export solve_ferrite_problem!

export FerriteSolver
export NewtonSolver, SteepestDescent
export NoLineSearch, ArmijoGoldstein
export BackslashSolver
export FixedTimeStepper

struct ConvergenceError <: Exception
    msg::String
end

"""
    FerriteSolver(nlsolver, timestepper, linsolver=BackslashSolver())

The standard `solver`, with two parts: A nonlinear solver 
(see [Nonlinear solvers](@ref)) and a time stepper (see [Time steppers](@ref)). 
"""
struct FerriteSolver{NLS,TS}
    nlsolver::NLS 
    timestepper::TS
end
FerriteSolver(;nlsolver, timestepper) = FerriteSolver(nlsolver, timestepper)

include("userfunctions.jl")
include("linearsolvers.jl")
include("linesearchers.jl")
include("nlsolvers.jl")
include("timesteppers.jl")

"""
    solve_ferrite_problem!(solver, problem)

Solve a time-dependent problem `r(x(t),t)=0` for `x(t)`, 
stepping throught the time `t`, using the `solver`.

For details on the functions that should be defined for `problem`,
see [User problem](@ref)
"""
function solve_ferrite_problem!(solver::FerriteSolver, problem)
    t = initial_time(solver.timestepper)
    step = 1
    converged = true
    while !islaststep(solver.timestepper, t, step)
        t, step = update_time(solver, t, step, converged)
        update_to_next_step!(problem, t)
        # Can make improved guess here based on previous values and send to update_problem! 
        # Alternatively, this can be done inside update_to_next_step! by the user if desired. 
        converged = solve_nonlinear!(solver.nlsolver, problem)
        if converged
            handle_converged!(problem)
            postprocess!(problem, step)
        else    # Should be an option if this should print or not...
            println("the nonlinear solver didn't converge")
        end
    end
end

end
