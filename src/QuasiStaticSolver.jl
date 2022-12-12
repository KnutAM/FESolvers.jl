"""
    QuasiStaticSolver(nlsolver, timestepper)

A quasi-static `solver` that solves problems of the type R(X(t),t)=0. 
It has two parts: A nonlinear solver (see [Nonlinear solvers](@ref)) 
and a time stepper (see [Time steppers](@ref)). 
"""
struct QuasiStaticSolver{NLS,TS} <: FESolver
    nlsolver::NLS 
    timestepper::TS
end
QuasiStaticSolver(;nlsolver, timestepper) = QuasiStaticSolver(nlsolver, timestepper)

getnlsolver(s::QuasiStaticSolver) = s.nlsolver
gettimestepper(s::QuasiStaticSolver) = s.timestepper

"""
    solve_problem!(solver, problem)

Solve a time-dependent problem `r(x(t),t)=0` for `x(t)`, 
stepping throught the time `t`, using the `solver`.

For details on the functions that should be defined for `problem`,
see [User problem](@ref)
"""
function solve_problem!(solver, problem)
    try
        _solve_problem!(solver, problem)
    finally
        close_problem(problem)
    end
end

function _solve_problem!(solver::QuasiStaticSolver, problem)
    t = initial_time(solver.timestepper)
    step = 1
    converged = true
    xold = deepcopy(getunknowns(problem))
    while !islaststep(solver.timestepper, t, step)
        t, step = update_time(solver, t, step, converged)
        update_to_next_step!(problem, t)
        converged = solve_nonlinear!(solver.nlsolver, problem)
        if converged
            copy!(xold, getunknowns(problem))
            handle_converged!(problem)
            postprocess!(problem, step, solver)
        else
            # Reset unknowns if it didn't converge to 
            setunknowns!(problem, xold)
            # TODO: ProgressMeter
            println("the nonlinear solver didn't converge")
        end
    end
end