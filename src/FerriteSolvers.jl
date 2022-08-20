module FerriteSolvers
export solve_ferrite_problem!

export FerriteSolver
export NewtonSolver
export FixedTimeStepper, AdaptiveTimeStepper
export BackslashSolver

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
FerriteSolver(nlsolver, timestepper) = FerriteSolver(;nlsolver, timestepper)

include("utils.jl")
include("userfunctions.jl")
include("linearsolvers.jl")
include("nlsolvers.jl")
include("timesteppers.jl")

"""
    solve_ferrite_problem!(solver, problem)

Solve a time-dependent problem `r(x(t),t)=0` for `x(t)`, 
stepping throught the time `t`, using the `solver`.
The following functions must be defined for the 
user-defined `problem`:

- `getunknowns(problem)`
- `getresidual(problem)`
- `getjacobian(problem)`
- `update_to_next_step!(problem, t)`
- `update_problem!(problem, Δx)`
- `calculate_residualnorm(problem)`
- `handle_converged!(problem)`

Additionally, one can define `postprocess!(problem, step)`
For details on each function above, please see the respective
function's documentation under [User problem](@ref)
"""
function solve_ferrite_problem!(solver::FerriteSolver, problem)
    t = initial_time(solver.timestepper)
    step = 1
    converged = true
    xold = deepcopy(getunknowns(problem))
    while !islaststep(solver.timestepper, t, step)
        t, step = update_time(solver, t, step, converged)
        update_to_next_step!(problem, t)
        update_problem!(problem)
        converged = solve_nonlinear!(solver, problem)
        if converged
            copy!(xold, getunknowns(problem))   # Would be safer with deepcopy here for custom arrays?
            handle_converged!(problem)
            postprocess!(problem, step)
        else
            setunknowns!(problem, xold)         # Reset unknowns if it didn't converge
            # TODO: Printing should be an option
            println("the nonlinear solver didn't converge")
            show(solver.nlsolver)
        end
    end
end

end
