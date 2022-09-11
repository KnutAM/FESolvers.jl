"""
    solve_nonlinear!(solver::FerriteSolver{<:NLS}, problem)

Solve one step in the nonlinear `problem`, given as `r(x) = 0`,
by using the solver of type `NLS`. 
"""
function solve_nonlinear! end

"""
    NewtonSolver(;linsolver=BackslashSolver(), linesearch=NoLineSearch(), maxiter=10, tolerance=1.e-6)

Use the standard NewtonRaphson solver to solve the nonlinear 
problem r(x) = 0 with `tolerance` within the maximum number 
of iterations `maxiter`. The `linsolver` argument determines the used linear solver
whereas the `linesearch` can be set currently between `NoLineSearch` or
`ArmijoGoldstein`. The latter globalizes the Newton strategy.
"""
struct NewtonSolver{LS,LSearch,T}
    linsolver::LS
    linesearch::LSearch
    maxiter::Int 
    tolerance::T
    numiter::Vector{Int}  # Last step number of iterations
    residuals::Vector{T}  # Last step residual history
end

function NewtonSolver(;linsolver=BackslashSolver(), linesearch=NoLineSearch(), maxiter=10, tolerance=1.e-6)
    residuals = zeros(typeof(tolerance), maxiter+1)
    return NewtonSolver(linsolver, linesearch, maxiter, tolerance, [zero(maxiter)], residuals)
end
getsystemmatrix(problem,::NewtonSolver) = getjacobian(problem)


@doc raw"""
    SteepestDescent(;maxiter=10, tolerance=1.e-6)

Use a steepest descent solver to solve the nonlinear 
problem r(x) = 0, which minimizes a potential \Pi with `tolerance`
and the maximum number of iterations `maxiter`.

This method is second derivative free and is not as locally limited as a Newton-Raphson scheme.
Thus, it is especially suited for stronlgy nonlinear behavior with potentially vanishing tangent stiffnesses.
"""
Base.@kwdef struct SteepestDescent{LineSearch,LinearSolver,T}
    linsolver::LinearSolver = BackslashSolver()
    linesearch::LineSearch = ArmijoGoldstein()
    maxiter::Int = 200
    tolerance::T = 1e-6
    numiter::Vector{Int} = [zero(maxiter)]  # Last step number of iterations
    residuals::Vector{T} = zeros(typeof(tolerance),maxiter+1)  # Last step residual history
end
getsystemmatrix(problem,::SteepestDescent) = getdescentpreconditioner(problem)


"""
    getlinesearch(nlsolver)
Returns the used linesearch of the nonlinear solver.
"""
getlinesearch(nlsolver::Union{NewtonSolver,SteepestDescent}) = nlsolver.linesearch
"""
    getmaxiter(nlsolver)
Returns the maximum number of iterations allowed for the nonlinear solver
"""
getmaxiter(nlsolver::Union{NewtonSolver,SteepestDescent}) = nlsolver.maxiter

"""
    gettolerance(nlsolver)
Returns the iteration tolerance for the solver
"""
gettolerance(nlsolver::Union{NewtonSolver,SteepestDescent}) = nlsolver.tolerance

reset_state!(args...) = nothing

function reset_state!(s::Union{NewtonSolver,SteepestDescent})
    fill!(s.numiter, 0)
    fill!(s.residuals, 0)
end

update_state!(args...) = nothing

function update_state!(s::Union{NewtonSolver,SteepestDescent}, r)
    s.numiter .+= 1
    s.residuals[s.numiter[1]] = r 
end

function solve_nonlinear!(solver::FerriteSolver, problem)
    nlsolver = solver.nlsolver
    maxiter = getmaxiter(nlsolver)
    reset_state!(nlsolver)
    for iter in 1:maxiter
        check_convergence_criteria(problem, nlsolver) && return true
        update_unknowns!(problem, nlsolver, iter)
    end
    check_convergence_criteria(problem, nlsolver) && return true
    return false
end

function update_unknowns!(problem, nlsolver::Union{SteepestDescent,NewtonSolver}, iter)
    Δa = similar(getunknowns(problem))  # TODO: Should have a solvercache for these
    r = getresidual(problem)
    K = getsystemmatrix(problem,nlsolver)
    solve_linear!(Δa, K, r, nlsolver.linsolver)
    linesearch!(Δa, problem, getlinesearch(nlsolver)) # Scale Δa
    update_problem!(problem, Δa)
end
