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

function _solve_problem!(problem, solver::QuasiStaticSolver)
    # Setup 
    nlsolver = getnlsolver(solver)
    timestepper = gettimestepper(solver)
    t = initial_time(timestepper)
    step = 1
    converged = true
    xold = deepcopy(getunknowns(problem))
    
    # Initial update of stiffness (or residual) if requested by the nlsolver
    if do_initial_update(nlsolver)
        update_problem!(problem, nothing, get_initial_update_spec(nlsolver))
    end

    # Initial postprocessing (to save initial conditions)
    postprocess!(problem, step, solver)

    # Main time-stepping loop
    while !(converged && islaststep(timestepper, t, step))
        t, step = update_time(solver, t, step, converged)
        update_to_next_step!(problem, t)
        converged = solve_nonlinear!(problem, nlsolver, converged)
        if converged
            copy!(xold, getunknowns(problem))
            postprocess!(problem, step, solver)
            handle_converged!(problem)
        else
            # Reset unknowns if it didn't converge to 
            setunknowns!(problem, xold)
        end
    end
end