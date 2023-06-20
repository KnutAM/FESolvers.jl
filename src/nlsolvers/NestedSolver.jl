#= Adaptive solving strategies
Possible wishes for use cases
	1) Run with one solver for x iterations, then switch to the next
	2) Run with one solver until a certain tolerance is achieved, then switch to another solver.
	3) Update a solver parameter if the solver fails
	4) Switch solver if solver fails
    5) Dynamically change solver parameter depending during iterations (e.g. Depending on if/how fast the error is decreasing)

Could be condensed in the following "tasks"
* Update solver parameters depending on a criterion (and potentially retry from some point)
* Change solver depending on a criterion (and potentially retry from some point)


1) Adaptivily changes between solvers and or parameters
2) A list of solvers to run through. 
=#


struct NestedSolver{NLS}
    solvers::Vector{NLS}
    reset::Vector{Bool} # length(restart)=length(solvers)-1
end

function solve_nonlinear!(problem, nlsolvers::NestedSolver, last_converged)
    x0 = copy(getunknowns(problem))
    converged = false
    for (i, nlsolver) in enumerate(nlsolvers.solvers)
        if converged
            # Skip if problem already converged according to the current nlsolver
            check_convergence_criteria(problem, nlsolver, Δa, iter) && continue
        elseif nlsolvers.reset[i] # If not converged, potentially reset the problem
            reset_problem!(problem, nlsolver; x=x0)
        end 
        converged = solve_nonlinear!(problem, nlsolver, i>1 || last_converged)
    end
	return converged
end

function solve_nonlinear!(problem, nlsolvers::AdaptiveNLSolver, last_converged)
    x0 = copy(getunknowns(problem))
    converged = false
    while !is_converged(problem, nlsolvers, Δa, iter)
        nlsolver = update_solver(nlsolvers, converged)
        should_reset_problem(nlsolvers) && reset_problem!(problem, nlsolver; x=x0)
        converged = solve_nonlinear!(problem, nlsolver, last_converged)
        should_give_up(problem, nlsolvers, Δa, iter) && return false
    end
    return true
end


#= More generalizations
Introduce 
- struct ConvergenceStatus end
- struct SolverLog end # Could be exchanged for more details/debug information

Move responsibility to 
- timestepper
    - Keep track of current time and step number 
- nlsolvers
    - Keep track of current ConvergenceStatus 
    - Potentially allow preallocated buffer (e.g. x0 or Δx)
    
=#