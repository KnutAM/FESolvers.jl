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
Base.@kwdef mutable struct NewtonSolver{LinearSolver,LineSearch,T,UT,SS}
    const linsolver::LinearSolver = BackslashSolver()
    const linesearch::LineSearch = NoLineSearch()
    const maxiter::Int = 10
    const tolerance::T = 1e-6
    const update_jac_first::Bool=true # Should jacobian be updated first iteration
    const update_jac_each::Bool=true  # Should jacobian be updated each iteration
    update_type::UT=nothing
    const state::SS=SolverState(maxiter)
end
getsystemmatrix(problem, ::NewtonSolver) = getjacobian(problem)

get_initial_update_spec(nls::NewtonSolver) = UpdateSpec(;jacobian=!(nls.update_jac_first), residual=false, type=nls.update_type)
function get_first_update_spec(nls::NewtonSolver)
    if is_converged(nls)
        return UpdateSpec(;jacobian=nls.update_jac_first, residual=true, type=nls.update_type)
    else # We must update if jacobian was updated for each
        return UpdateSpec(;jacobian=nls.update_jac_each, residual=true, type=nls.update_type)
    end
end
function get_update_spec(nls::NewtonSolver)
    return UpdateSpec(;jacobian=nls.update_jac_each, residual=true, type=nls.update_type)
end

get_linesearch(nls::NewtonSolver) = nls.linesearch
get_linear_solver(nls::NewtonSolver) = nls.linsolver

get_max_iter(nls::NewtonSolver) = nls.maxiter
get_tolerance(nls::NewtonSolver) = nls.tolerance
get_solver_state(nls::NewtonSolver) = nls.state

set_update_type!(nls::NewtonSolver, type) = (nls.update_type = type)