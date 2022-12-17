"""
    solve_nonlinear!(problem, nlsolver)

Solve the current time step in the nonlinear `problem`, (`r(x) = 0`),
by using the nonlinear solver: `nlsolver`. 
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
    should_update(nlsolver, iter) -> (Bool, Bool)

Should the residual and and jacobian be updated at the given state. 
The results are passed as kwargs to `update_problem!`
Defaults to both `true` if not defined. 
"""
should_update(args...) = (true, true)

# Solver state 

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
    SolverState{T}

Saves the current state of the nonlinear solver, such as the iteration number 
and the convergence history (automatically resized)
"""
mutable struct SolverState{T}
    numiter::Int
    residuals::Vector{T}
end
SolverState(::Type{T}, residual_length::Int=1) where T = SolverState(0, zeros(T, residual_length))

getnumiter(ss::SolverState) = ss.numiter 
get_convergence_history(ss::SolverState) = ss.residuals[1:getnumiter(ss)]

function reset_state!(ss::SolverState)
    ss.numiter = 0
    fill!(ss.residuals, 0)
    return nothing 
end

function update_state!(ss::SolverState, r)
    ss.numiter += 1
    if length(ss.residuals) < ss.numiter
        push!(ss.residuals, r)
    else
        ss.residuals[ss.numiter] = r
    end
    return nothing 
end

# General matrix solver

"""
    MatrixSolver

A MatrixSolver solves the problem `R(X)=0` by using the 
update direction `Δx = -K⁻¹ R(X)`, whose length may be 
adjusted by an optional line search algorithm. 
The `Type` of matrix `K` to use distinguishes different solvers, 
such as the `NewtonSolver` (uses the tangent) and 
the `SteepestDescent` (uses a constant preconditioner matrix 
that defaults to the identity, `LinearAlgebra.I`)
"""
struct MatrixSolver{Type,LinearSolver,LineSearch,State,T}
    linearsolver::LinearSolver
    linesearch::LineSearch
    maxiter::Int 
    tolerance::T 
    state::State
end

function MatrixSolver{Type}(linearsolver::LinearSolver, linesearch::LineSearch, maxiter, tolerance::T, 
        state::State=SolverState(T, maxiter+1)) where {Type,LinearSolver,LineSearch,T,State}
    return MatrixSolver{Type, LinearSolver, LineSearch, State, T}(linearsolver, linesearch, maxiter, tolerance, state)
end


"""
    getlinearsolver(nlsolver::MatrixSolver)

Returns the linear solver for `nlsolver`
"""
getlinearsolver(nlsolver::MatrixSolver) = nlsolver.linearsolver

"""
    getlinesearch(nlsolver)
Returns the used linesearch of the nonlinear solver.
"""
getlinesearch(nlsolver::MatrixSolver) = nlsolver.linesearch

getmaxiter(nlsolver::MatrixSolver) = nlsolver.maxiter
gettolerance(nlsolver::MatrixSolver) = nlsolver.tolerance
getnumiter(s::MatrixSolver) = getnumiter(s.state)
reset_state!(s::MatrixSolver) = reset_state!(s.state)
update_state!(s::MatrixSolver, args...) = update_state!(s.state, args...)

"""
    create_newton_solver(;linsolver=BackslashSolver(), linesearch=NoLineSearch(), maxiter=10, tolerance=1.e-6)

Create a `MatrixSolver` that always uses the updated jacobian matrix. 
Use the standard Newton-Raphson solver to solve the nonlinear 
problem r(x) = 0 with `tolerance` within the maximum number 
of iterations `maxiter`. The `linsolver` argument determines the used linear solver
whereas the `linesearch` can be set currently between `NoLineSearch` or
`ArmijoGoldstein`. The latter globalizes the Newton strategy.
"""
struct NewtonSolver end

function create_newton_solver(;linsolver=BackslashSolver(), linesearch=NoLineSearch(), maxiter=10, tolerance=1.e-6)
    return MatrixSolver{NewtonSolver}(linsolver, linesearch, maxiter, tolerance)
end
getsystemmatrix(problem, ::MatrixSolver{NewtonSolver}) = getjacobian(problem)
should_update(::MatrixSolver{NewtonSolver}, iter) = (true, true)


"""
    create_steepest_descent_solver(;maxiter=200, tolerance=1.e-6)

Create a `MatrixSolver` that does a steepest descents to solve the nonlinear 
problem r(x) = 0, which minimizes a potential `Π` with `tolerance`
and the maximum number of iterations `maxiter`.

This method is second derivative free and is not as locally limited as a Newton-Raphson scheme.
Thus, it is especially suited for strongly nonlinear behavior with potentially vanishing tangent stiffnesses.
"""
struct SteepestDescent end
function create_steepest_descent_solver(;linsolver=BackslashSolver(), linesearch=ArmijoGoldstein(), maxiter=200, tolerance=1e-6)
    return MatrixSolver{SteepestDescent}(linsolver, linesearch, maxiter, tolerance)
end
getsystemmatrix(problem, ::MatrixSolver{SteepestDescent}) = getdescentpreconditioner(problem)
should_update(::MatrixSolver{SteepestDescent}, iter) = (true, false)

function solve_nonlinear!(problem, nlsolver)
    maxiter = getmaxiter(nlsolver)
    reset_state!(nlsolver)
    update_residual, update_jacobian = should_update(nlsolver, 0)
    update_problem!(problem, nothing; update_residual=update_residual, update_jacobian=update_jacobian)
    Δa = zero(getunknowns(problem))
    for iter in 1:maxiter
        check_convergence_criteria(problem, nlsolver, Δa, iter) && return true
        calculate_update!(Δa, problem, nlsolver, iter)
        update_residual, update_jacobian = should_update(nlsolver, iter)
        update_problem!(problem, Δa; update_residual=update_residual, update_jacobian=update_jacobian)
    end
    check_convergence_criteria(problem, nlsolver, Δa, maxiter+1) && return true
    return false
end

function calculate_update!(Δa, problem, nlsolver::MatrixSolver, iter)
    r = getresidual(problem)
    K = getsystemmatrix(problem,nlsolver)
    solve_linear!(Δa, K, r, getlinearsolver(nlsolver))
    linesearch!(Δa, problem, getlinesearch(nlsolver)) # Scale Δa
    return Δa
end

"""
    create_linear_problem_solver(;linsolver=BackslashSolver())

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
"""
struct LinearProblem end 
function create_linear_problem_solver(;linearsolver=BackslashSolver())
    return MatrixSolver{LinearProblem}(linearsolver, NoLineSearch(), 1, NaN, nothing)
end
getsystemmatrix(problem, ::MatrixSolver{LinearProblem}) = getjacobian(problem)

check_convergence_criteria(problem, ::MatrixSolver{LinearProblem}, Δa, iter::Int) = iter>1    # Converges in one iteration 
should_update(::MatrixSolver{LinearProblem}, iter) = (iter==0, iter==0)

#=
function solve_nonlinear!(problem, nlsolver::LinearProblemSolver)
    update_problem!(problem, nothing; update_residual=true, update_jacobian=true)
    r = getresidual(problem)
    K = getsystemmatrix(problem,nlsolver)
    Δa = similar(getunknowns(problem))
    solve_linear!(Δa, K, r, nlsolver.linsolver)
    update_problem!(problem, Δa; update_residual=false, update_jacobian=false)
    return true # Assume always converged
end
=#