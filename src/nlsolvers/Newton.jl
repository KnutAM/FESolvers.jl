"""
    NewtonSolver(;
        linsolver=BackslashSolver(), linesearch=NoLineSearch(), 
        maxiter=10, tolerance=1.e-6,
        update_jac_first=true, update_jac_each=true)

Use the standard Newton-Raphson solver to solve the nonlinear 
problem r(x) = 0 with `tolerance` within the maximum number 
of iterations `maxiter`. 

## Quasi-Newton methods
**Linesearch:** The `linsolver` argument determines the used linear solver
whereas the `linesearch` can be set currently between `NoLineSearch` or
`ArmijoGoldstein`. The latter globalizes the Newton strategy.

**Jacobian updates: **
The keyword `update_jac_first` decides if the jacobian from the previously converged
time step should be updated after calling `update_to_next_step!`, or to use the old. 
Setting `update_jac_each` implies that the jacobian will not be updated during the iterations.
If both `update_jac_each` and `update_jac_first` are false, the initial jacobian will be used 
throughout. Note that these keywords require that the problem respects the `update_jacobian`
keyword given to `update_problem!`.
For time-independent problems or time-depdent problems with 
constant time steps, `update_jac_first=false` is often a good choice. 
However, for time-dependent problems with changing time step length, 
the standard solver (default), may work better. 
"""
Base.@kwdef mutable struct NewtonSolver{LinearSolver,LineSearch,T,SS}
    const linsolver::LinearSolver = BackslashSolver()
    const linesearch::LineSearch = NoLineSearch()
    const maxiter::Int = 10
    const tolerance::T = 1e-6
    const update_jac_first::Bool=true # Should jacobian be updated first iteration
    const update_jac_each::Bool=true  # Should jacobian be updated each iteration
    const state::SS=SolverState(maxiter)
end
getsystemmatrix(problem, ::NewtonSolver) = getjacobian(problem)

get_initial_update_spec(nlsolver::NewtonSolver) = UpdateSpec(jacobian=!(nlsolver.update_jac_first), residual=false)
function get_first_update_spec(nlsolver::NewtonSolver)
    if is_converged(nlsolver)
        return UpdateSpec(jacobian=nlsolver.update_jac_first, residual=true)
    else # We must update if jacobian was updated for each
        return UpdateSpec(jacobian=nlsolver.update_jac_each, residual=true)
    end
end
function get_update_spec(nlsolver::NewtonSolver)
    return UpdateSpec(;jacobian=nlsolver.update_jac_each, residual=true)
end

get_linesearch(nlsolver::NewtonSolver) = nlsolver.linesearch
get_linear_solver(nlsolver::NewtonSolver) = nlsolver.linsolver

get_max_iter(nlsolver::NewtonSolver) = nlsolver.maxiter
get_tolerance(nlsolver::NewtonSolver) = nlsolver.tolerance
get_solver_state(nlsolver::NewtonSolver) = nlsolver.state