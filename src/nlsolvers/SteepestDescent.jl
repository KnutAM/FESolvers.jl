"""
    SteepestDescent(;maxiter=10, tolerance=1.e-6)

Use a steepest descent solver to solve the nonlinear 
problem r(x) = 0, which minimizes a potential ``\\Pi`` with `tolerance`
and the maximum number of iterations `maxiter`.

This method is second derivative free and is not as locally limited as a Newton-Raphson scheme.
Thus, it is especially suited for strongly nonlinear behavior with potentially vanishing tangent stiffnesses.
For this method, it is required to implement [`getdescentpreconditioner`](@ref) or alternatively
[`getsystemmatrix`](@ref) with `SteepestDescent`. 
"""
Base.@kwdef mutable struct SteepestDescent{LineSearch,LinearSolver,T,SS}
    const linsolver::LinearSolver = BackslashSolver()
    const linesearch::LineSearch = ArmijoGoldstein()
    const maxiter::Int = 200
    const tolerance::T = 1e-6
    const state::SS=SolverState(maxiter)
end
getsystemmatrix(problem, ::SteepestDescent) = getdescentpreconditioner(problem)

get_initial_update_spec(::SteepestDescent) = UpdateSpec(;jacobian=false, residual=false)
get_first_update_spec(::SteepestDescent) = UpdateSpec(;jacobian=false, residual=true)
get_update_spec(::SteepestDescent) = UpdateSpec(;jacobian=false, residual=true)

get_linesearch(nlsolver::SteepestDescent) = nlsolver.linesearch
get_linear_solver(nlsolver::SteepestDescent) = nlsolver.linsolver

get_max_iter(nlsolver::SteepestDescent) = nlsolver.maxiter
get_tolerance(nlsolver::SteepestDescent) = nlsolver.tolerance

get_solver_state(nlsolver::SteepestDescent) = nlsolver.state