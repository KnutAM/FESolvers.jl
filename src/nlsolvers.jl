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
        newtonsolver.residuals[i] = calculate_convergence_criterion(problem)
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