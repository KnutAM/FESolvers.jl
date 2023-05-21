"""
    solve_nonlinear!(problem, nlsolver, last_converged)

Solve the current time step in the nonlinear `problem`, (`r(x) = 0`),
by using the nonlinear solver `nlsolver`. `last_converged::Bool` 
is just for information if the last time step converged or not. 
In many cases it suffices to overload [`calculate_update!`](@ref) 
for a custom nonlinear solver. 
"""
function solve_nonlinear! end

"""
    function calculate_update!(Δx, problem, nlsolver, iter)

According to the nonlinear solver, `nlsolver`, at iteration `iter`,
calculate the update, `Δx` to the unknowns `x`.
"""
function calculate_update! end

"""
    getmaxiter(nlsolver)

Returns the maximum number of iterations allowed for the nonlinear solver
"""
function getmaxiter end

"""
    getnumiter(nlsolver)

Returns the last number of iterations used by the nonlinear solver
"""
function getnumiter end

"""
    gettolerance(nlsolver)

Returns the iteration tolerance for the solver
"""
function gettolerance end

"""
    update_state!(nlsolver, r)

A nonlinear solver may solve information about its convergence state.
`r` is the output from [`calculate_convergence_measure`](@ref) when 
this function is called by the default implementation of 
[`check_convergence_criteria`](@ref). 
`update_state!` is optional to implement
"""
update_state!(::Any, _) = nothing

"""
    reset_state!(nlsolver)

If [`update_state!`](@ref) is implemented, this function is used to 
reset its state at the beginning of each new time step. 
"""
reset_state!(::Any) = nothing

"""
    update_jac_initial(nlsolver)

Should the jacobian be updated before starting to solve the problem?
Defaults to `false` if not implemented
"""
update_jac_initial(::Any) = false

"""
    update_jac_first(nlsolver)

Should the jacobian be updated before the first iteration of each time step?
Defaults to `true` if not implemented
"""
update_jac_first(::Any) = true

"""
    update_jac_each(nlsolver)

Should the jacobian be updated after each iteration?
Defaults to `true` if not implemented
"""
update_jac_each(::Any) = true

"""
    NewtonSolver(;
        linsolver=BackslashSolver(), linesearch=NoLineSearch(), 
        maxiter=10, tolerance=1.e-6,
        update_jac_first=true, update_jac_each=true)

Use the standard NewtonRaphson solver to solve the nonlinear 
problem r(x) = 0 with `tolerance` within the maximum number 
of iterations `maxiter`. The `linsolver` argument determines the used linear solver
whereas the `linesearch` can be set currently between `NoLineSearch` or
`ArmijoGoldstein`. The latter globalizes the Newton strategy.

The keyword `update_jac_first` decides if the jacobian from the previously converged
time step should be updated after calling `update_to_next_step!`, or to use the old. 
Setting `update_jac_each` implies that the jacobian will not be updated during the iterations.
If both `update_jac_each` and `update_jac_first` are false, the initial jacobian will be used 
throughout. Note that these keywords require that the problem respects the `update_jacobian`
keyword given to `update_problem!`. For time-independent problems or time-depdent problems with 
constant time steps, `update_jac_first=false` is often a good choice. 
However, for time-dependent problems with changing time step length, 
especially without changing prescribed values, the default option should be better. 
"""
mutable struct NewtonSolver{LS,LSearch,T}
    const linsolver::LS
    const linesearch::LSearch
    const maxiter::Int
    const tolerance::T
    const update_jac_first::Bool # Should jacobian be updated first iteration
    const update_jac_each::Bool  # Should jacobian be updated each iteration
    numiter::Int                # Last step number of iterations
    const residuals::Vector{T}  # Last step residual history
end

function NewtonSolver(;
        linsolver=BackslashSolver(), linesearch=NoLineSearch(), 
        maxiter=10, tolerance=1.e-6,
        update_jac_first=true, update_jac_each=true
    )
    residuals = zeros(typeof(tolerance), maxiter+1)
    return NewtonSolver(
        linsolver, linesearch, 
        maxiter, tolerance, 
        update_jac_first, update_jac_each, 
        0, residuals
        )
end
getsystemmatrix(problem, ::NewtonSolver) = getjacobian(problem)

update_jac_initial(nlsolver::NewtonSolver) = !(nlsolver.update_jac_first) 
update_jac_first(nlsolver::NewtonSolver) = nlsolver.update_jac_first
update_jac_each(nlsolver::NewtonSolver) = nlsolver.update_jac_each

@doc raw"""
    SteepestDescent(;maxiter=10, tolerance=1.e-6)

Use a steepest descent solver to solve the nonlinear 
problem r(x) = 0, which minimizes a potential \Pi with `tolerance`
and the maximum number of iterations `maxiter`.

This method is second derivative free and is not as locally limited as a Newton-Raphson scheme.
Thus, it is especially suited for strongly nonlinear behavior with potentially vanishing tangent stiffnesses.
"""
Base.@kwdef mutable struct SteepestDescent{LineSearch,LinearSolver,T}
    const linsolver::LinearSolver = BackslashSolver()
    const linesearch::LineSearch = ArmijoGoldstein()
    const maxiter::Int = 200
    const tolerance::T = 1e-6
    numiter::Int = 0  # Last step number of iterations
    const residuals::Vector{T} = zeros(typeof(tolerance),maxiter+1)  # Last step residual history
end
getsystemmatrix(problem, ::SteepestDescent) = getdescentpreconditioner(problem)
update_jac_initial(::SteepestDescent) = false
update_jac_first(::SteepestDescent) = false
update_jac_each(::SteepestDescent) = false

"""
    getlinesearch(nlsolver)
Returns the used linesearch of the nonlinear solver.
"""
getlinesearch(nlsolver::Union{NewtonSolver,SteepestDescent}) = nlsolver.linesearch

getmaxiter(nlsolver::Union{NewtonSolver,SteepestDescent}) = nlsolver.maxiter
gettolerance(nlsolver::Union{NewtonSolver,SteepestDescent}) = nlsolver.tolerance
getnumiter(s::Union{NewtonSolver,SteepestDescent}) = s.numiter


function reset_state!(s::Union{NewtonSolver,SteepestDescent})
    s.numiter = 0
    fill!(s.residuals, 0)
end

function update_state!(s::Union{NewtonSolver,SteepestDescent}, r)
    s.numiter += 1
    s.residuals[s.numiter] = r 
end

function solve_nonlinear!(problem, nlsolver, last_converged)
    maxiter = getmaxiter(nlsolver)
    reset_state!(nlsolver)

    if last_converged || update_jac_first(nlsolver)
        update_problem!(problem, nothing; update_residual=true, update_jacobian=update_jac_first(nlsolver))
    elseif update_jac_each(nlsolver) # If last step didn't converge and we updated the jacobian for each of those iterations, 
         # we have no option than to update the jacobian to the new value
        update_problem!(problem, nothing; update_residual=true, update_jacobian=true)
    else # In this case, !update_jac_each(nlsolver) && !update_jac_first(nlsolver)
        update_problem!(problem, nothing; update_residual=true, update_jacobian=false)
    end

    Δa = zero(getunknowns(problem))
    for iter in 1:maxiter
        check_convergence_criteria(problem, nlsolver, Δa, iter) && return true
        calculate_update!(Δa, problem, nlsolver, iter)
        update_problem!(problem, Δa; update_residual=true, update_jacobian=update_jac_each(nlsolver))
    end
    check_convergence_criteria(problem, nlsolver, Δa, maxiter+1) && return true
    return false
end

function calculate_update!(Δa, problem, nlsolver::Union{SteepestDescent,NewtonSolver}, iter)
    r = getresidual(problem)
    K = getsystemmatrix(problem,nlsolver)
    solve_linear!(Δa, K, r, nlsolver.linsolver)
    linesearch!(Δa, problem, getlinesearch(nlsolver)) # Scale Δa
    return Δa
end

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

getsystemmatrix(problem, ::LinearProblemSolver) = getjacobian(problem)

# Support this just for convenience, it makes it easier when printing status etc. 
getnumiter(::LinearProblemSolver) = 0
getmaxiter(::LinearProblemSolver) = 1
gettolerance(::LinearProblemSolver) = NaN

function solve_nonlinear!(problem, nlsolver::LinearProblemSolver, last_converged)
    update_problem!(problem, nothing; update_residual=true, update_jacobian=true)
    r = getresidual(problem)
    K = getsystemmatrix(problem,nlsolver)
    Δa = similar(getunknowns(problem))
    solve_linear!(Δa, K, r, nlsolver.linsolver)
    update_problem!(problem, Δa; update_residual=false, update_jacobian=false)
    return true # Assume always converged
end