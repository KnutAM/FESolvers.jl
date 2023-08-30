"""
    LinearProblemSolver(;linsolver=BackslashSolver())

This is a special type of "Nonlinear solver", which actually only solves linear problems, 
but allows all other features (i.e. time stepping and postprocessing) of the `FESolvers` 
package to be used. In particular, it allows you to maintain all other parts of your problem 
exactly the same as for a nonlinear problem, but it is possible to get better performance as 
it is, in principle, not necessary to assemble twice in each time step. 

This solver is specialized for linear problems of the form
```math
\\boldsymbol{r}(\\boldsymbol{x}(t),t)=\\boldsymbol{K}(t) \\boldsymbol{x}(t) - \\boldsymbol{f}(t)
```
where ``\\boldsymbol{K}=\\partial \\boldsymbol{r}/\\partial \\boldsymbol{x}``. 
It expects that ``\\boldsymbol{x}(t)`` and ``\\boldsymbol{r}(t)`` have been updated to 
``\\boldsymbol{x}_\\mathrm{bc}`` and ``\\boldsymbol{r}_\\mathrm{bc}=\\boldsymbol{K}(t)\\boldsymbol{x}_\\mathrm{bc}-\\boldsymbol{f}(t)``,
such that 
```math
\\boldsymbol{x}(t) = \\boldsymbol{x}_\\mathrm{bc} - \\boldsymbol{K}^{-1}(t)\\boldsymbol{r}_\\mathrm{bc}
= \\boldsymbol{x}_\\mathrm{bc} - \\boldsymbol{K}^{-1}(t)\\left[\\boldsymbol{K}(t)\\boldsymbol{x}_\\mathrm{bc}-\\boldsymbol{f}(t)\\right]
= \\boldsymbol{K}^{-1}(t)\\boldsymbol{f}(t)
```
is the solution to the current time step. This normally implies when using Ferrite the same procedure as for nonlinear problems, i.e. 
that the boundary conditions are applied in `update_to_next_step!` and `update_problem!`, as well as the calculation of the residual 
according to above and that `apply_zero!(K,r,ch)` is called (on both the residual and the stiffness matrix).

If you have strange results when running the `LinearProblemSolver`, 
please ensure that the problem converges in one iteration for the [`NewtonSolver`](@ref)
"""
struct LinearProblemSolver{LinearSolver}
    linsolver::LinearSolver
end
LinearProblemSolver(;linsolver=BackslashSolver()) = LinearProblemSolver(linsolver)

get_initial_update_spec(::LinearProblemSolver) = UpdateSpec(jacobian=false, residual=false)
get_first_update_spec(::LinearProblemSolver, args...) = UpdateSpec(jacobian=true, residual=true)
get_update_spec(::LinearProblemSolver) = UpdateSpec(jacobian=false, residual=false)

getsystemmatrix(problem, ::LinearProblemSolver) = getjacobian(problem)

get_linesearch(::LinearProblemSolver) = NoLineSearch()
get_linear_solver(nlsolver::LinearProblemSolver) = nlsolver.linsolver

# Support this just for convenience, it makes it easier when printing status etc. 
get_max_iter(::LinearProblemSolver) = 1
get_tolerance(::LinearProblemSolver) = NaN

get_num_iter(::LinearProblemSolver) = 0
function get_convergence_measure(::LinearProblemSolver, args...)
    state = SolverState(1)
    state.numiter = 2
    get_convergence_measure(state, args...)
end
is_converged(::LinearProblemSolver) = true
check_convergence(::LinearProblemSolver) = nothing # Always converged

function solve_nonlinear!(problem, nlsolver::LinearProblemSolver)
    update_problem!(problem, nothing, get_first_update_spec(nlsolver))
    Δa = similar(getunknowns(problem))
    
    calculate_update!(Δa, problem, nlsolver)

    update_problem!(problem, Δa, get_update_spec(nlsolver))
    return true # Assume always converged
end