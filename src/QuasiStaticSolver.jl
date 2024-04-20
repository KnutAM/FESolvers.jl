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

get_nlsolver(s::QuasiStaticSolver) = s.nlsolver
get_timestepper(s::QuasiStaticSolver) = s.timestepper

function _solve_problem!(problem, solver::QuasiStaticSolver)
    # Setup 
    nlsolver = get_nlsolver(solver)
    timestepper = get_timestepper(solver)
    xold = deepcopy(getunknowns(problem))
    
    # Initial update of stiffness (or residual) if requested by the nlsolver
    if should_do_initial_update(nlsolver)
        update_problem!(problem, nothing, get_initial_update_spec(nlsolver))
    end

    # Initial postprocessing (to save initial conditions)
    postprocess!(problem, solver)

    # Main time-stepping loop
    while !is_finished(solver)                               # FESolvers function
        step_time!(solver)                                   # FESolvers function
        update_to_next_step!(problem, get_time(timestepper)) # User function
        solve_nonlinear!(problem, nlsolver)                  # FESolvers function
        if is_converged(nlsolver)                            # FESolvers function
            copy!(xold, getunknowns(problem))
            postprocess!(problem, solver)                    # User function
            handle_converged!(problem)                       # User function
        else
            # Reset unknowns if no convergence and,
            # potentially, try a different time step
            setunknowns!(problem, xold)                      # User function
            handle_notconverged!(problem, solver)            # Optional user function
        end                                                  # (with default implementation)
    end
end