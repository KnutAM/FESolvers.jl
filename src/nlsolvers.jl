"""
    solve_nonlinear!(solver::FerriteSolver{<:NLS}, problem)

Solve one step in the nonlinear `problem`, given as `r(x) = 0`,
by using the solver of type `NLS`. 
"""
function solve_nonlinear! end

"""
    NewtonSolver(;maxiter=10, tolerance=1.e-6)

Use the standard NewtonRaphson solver to solve the nonlinear 
problem r(x) = 0 with `tolerance` within the maximum number 
of iterations `maxiter`
"""
struct NewtonSolver{LS,T}
    linsolver::LS
    maxiter::Int 
    tolerance::T
    numiter::Vector{Int}  # Last step number of iterations
    residuals::Vector{T}  # Last step residual history
end

function NewtonSolver(;linsolver=BackslashSolver(), maxiter=10, tolerance=1.e-6)
    residuals = zeros(typeof(tolerance), maxiter+1)
    return NewtonSolver(linsolver, maxiter, tolerance, [zero(maxiter)], residuals)
end

function reset!(s::NewtonSolver)
    fill!(s.numiter, 0)
    fill!(s.residuals, 0)
end

function Base.show(s::NewtonSolver)
    println("Newton solver has used $(s.numiter[1]) iterations")
    println("residuals")
    for i in 1:s.numiter[1]
        println("r[$i] = $(s.residuals[i])")
    end
end

function solve_nonlinear!(solver::FerriteSolver{<:NewtonSolver}, problem)
    newtonsolver = solver.nlsolver
    maxiter = newtonsolver.maxiter
    tol = newtonsolver.tolerance
    Δa = zero(getunknowns(problem))
    reset!(newtonsolver)
    for i in 1:(maxiter+1)
        newtonsolver.numiter .= i
        newtonsolver.residuals[i] = calculate_residualnorm(problem)
        if newtonsolver.residuals[i] < tol
            return true
        end
        i>maxiter && return false # Did not converge
        r = getresidual(problem)
        K = getjacobian(problem)
        update_guess!(Δa, K, r, newtonsolver.linsolver)
        update_problem!(problem, Δa)
    end
end

abstract type AbstractLineSearch end
"Singleton that does not perform a linesearch when used in a nonlinear solver"
struct NoLineSearch <: AbstractLineSearch end

"""
Backtracking line search based on the Armijo-Goldstein condition

```math
\Pi(\boldsymbol{u} + \tau \Delta\boldsymbol{u}) \leq \Pi(\boldsymbol{u}) - \mu\tau\delta\Pi(\boldsymbol{u})[\Delta \boldsymbol{u}]
```

where \$\Pi\$ is the potential, \$\tau\$ the stepsize, and \$\delta\Pi\$ the residuum.

#Fields
- `β::T = 0.9` constant factor that changes the steplength τ in each iteration
- `μ::T = 0.01` second constant factor that determines how much the potential needs to decrease additionally
- `τ0::T = 1.0` start stepsize 
- `τmin::T = 1e-4` minimal stepsize
"""
Base.@kwdef struct ArmijoGoldstein{T} <: AbstractLineSearch
    β::T = 0.9
    μ::T = 0.01
    τ0::T = 1.0
    τmin::T = 1e-4
end

"""
    SteepestDescent(;maxiter=10, tolerance=1.e-6)

Use a steepest descent solver to solve the nonlinear 
problem r(x) = 0, which minimizes a potential \Pi with `tolerance`
and the maximum number of iterations `maxiter`.

This method is second derivative free and is not as locally limited as a Newton-Raphson scheme.
Thus, it is especially suited for stronlgy nonlinear behavior with potentially vanishing tangent stiffnesses.
"""
Base.@kwdef struct SteepestDescent{LinearSolver,T,LineSearch{T}}
    linsolver::LinearSolver = BackslashSolver()
    linesearch::LineSearch = ArmijoGoldstein()
    maxiter::Int = 100
    tolerance::T = 1e-6
end

function solve_nonlinear!(solver::FerriteSolver{<:SteepestDescent}, problem)
    steepestdescent = solver.nlsolver
    ls = solver.linesearch
    maxiter = steepestdescent.maxiter
    tol = steepestdescent.tolerance
    Δa = zero(getunknowns(problem))
    reset!(steepestdescent)
    for i in 1:(maxiter+1)
        steepestdescent.numiter .= i
        _norm = calculate_residualnorm(problem)
        if _norm < tol
            return true
        end
        i>maxiter && return false # Did not converge
        r = getresidual(problem)
        K = getpreconditioning(problem)
        update_guess!(Δa, K, r, steepestdescent.linsolver)
        linesearch() 
        update_problem!(problem, Δa)
    end
end
