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
Base.@kwdef mutable struct SteepestDescent{LineSearch,LinearSolver,T}
    const linsolver::LinearSolver = BackslashSolver()
    const linesearch::LineSearch = ArmijoGoldstein()
    const maxiter::Int = 200
    const tolerance::T = 1e-6
    numiter::Int = 0  # Last step number of iterations
    const residuals::Vector{T} = zeros(typeof(tolerance),maxiter+1)  # Last step residual history
end
getsystemmatrix(problem, ::SteepestDescent) = getdescentpreconditioner(problem)

get_initial_update_spec(::SteepestDescent) = UpdateSpec(;jacobian=false, residual=false)
get_first_update_spec(::SteepestDescent, _) = UpdateSpec(;jacobian=false, residual=true)
get_update_spec(::SteepestDescent) = UpdateSpec(;jacobian=false, residual=true)

get_linesearch(nlsolver::SteepestDescent) = nlsolver.linesearch
get_linear_solver(nlsolver::SteepestDescent) = nlsolver.linsolver

getmaxiter(nlsolver::SteepestDescent) = nlsolver.maxiter
gettolerance(nlsolver::SteepestDescent) = nlsolver.tolerance
getnumiter(s::SteepestDescent) = s.numiter
get_convergence_measures(s::SteepestDescent, inds=1:getnumiter(s)) = s.residuals[inds]

function reset_state!(s::SteepestDescent)
    s.numiter = 0
    fill!(s.residuals, 0)
end

function update_state!(s::SteepestDescent, _, r)
    s.numiter += 1
    s.residuals[s.numiter] = r 
end