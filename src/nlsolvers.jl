abstract type AbstractLineSearch end
"Singleton that does not perform a linesearch when used in a nonlinear solver"
struct NoLineSearch <: AbstractLineSearch end

@doc raw"""
    Armijo-Goldstein{T}(;β=0.9,μ=0.01,τ0=1.0,τmin=1e-4)
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
    μ::T = 0.10
    τ0::T = 1.0
    τmin::T = 1e-5
end

linesearch!(searchdirection, problem, ls::NoLineSearch) = nothing

function linesearch!(searchdirection, problem, ls::ArmijoGoldstein)
    τ = ls.τ0; μ = ls.μ; β = ls.β
    𝐮 = getunknowns(problem)
    Π₀ = calculate_energy(problem,𝐮)
    δΠ₀ = getresidual(problem)
    Πₐ = calculate_energy(problem,𝐮 .+ τ .* searchdirection)
    armijo = Πₐ - Π₀ - μ * τ * δΠ₀'searchdirection
    
    while armijo > 0 && !isapprox(armijo,0.0,atol=1e-8)
        τ *= β
        Πₐ = calculate_energy(problem,𝐮 .+ τ .* searchdirection)
        armijo = Πₐ - Π₀ - μ * τ * δΠ₀'searchdirection
    end
    τ = max(ls.τmin,τ)
    searchdirection .*= τ
end

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

function Base.show(s::NewtonSolver)
    println("Newton solver has used $(s.numiter[1]) iterations")
    println("residuals")
    for i in 1:s.numiter[1]
        println("r[$i] = $(s.residuals[i])")
    end
end

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

getsystemmatrix(problem,solver::SteepestDescent) = LinearAlgebra.I

function Base.show(s::SteepestDescent)
    println("Steepest Descent")
end

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


function reset_state!(s::Union{NewtonSolver,SteepestDescent})
    fill!(s.numiter, 0)
    fill!(s.residuals, 0)
end

function update_state!(s::Union{NewtonSolver,SteepestDescent}, r)
    s.numiter .+= 1
    s.residuals[s.numiter[1]] = r 
end

function check_convergence_criteria(problem, nlsolver)
    r = calculate_convergence_measure(problem)
    update_state!(nlsolver, r)
    return r < gettolerance(nlsolver)
end

function solve_nonlinear!(solver::FerriteSolver, problem)
    nlsolver = solver.nlsolver
    maxiter = getmaxiter(nlsolver)
    reset_state!(nlsolver)
    for iter in 1:maxiter
        check_convergence_criteria(problem, nlsolver) && return true
        update_unknowns!(problem, nlsolver, iter)
    end
    check_convergence_criteria(problem, solver) && return true
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
