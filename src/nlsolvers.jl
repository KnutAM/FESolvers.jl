"""
    solve_nonlinear!(solver::FerriteSolver{<:NLS}, problem)

Solve one step in the nonlinear `problem`, given as `r(x) = 0`,
by using the solver of type `NLS`. 
"""
function solve_nonlinear! end

# Default functions overloaded by specific nonlinear solvers
getoptiter(s) = Int(floor(getmaxiter(s)/2))

# Specific nonlinear solvers
"""
    NewtonSolver(;maxiter=10, tolerance=1.e-6, optiter=maxiter/2)

Use the standard NewtonRaphson solver to solve the nonlinear 
problem r(x) = 0 with `tolerance` within the maximum number 
of iterations `maxiter`. `optiter` specifies the optimal number 
of iterations, if used together with an adaptive time stepper. 
"""
struct NewtonSolver{LS,T}
    linsolver::LS
    maxiter::Int 
    optiter::Int        
    tolerance::T    
    numiter::ScalarWrapper{Int} # Last step number of iterations
    residuals::Vector{T}        # Last step residual history
end

function NewtonSolver(;linsolver=BackslashSolver(), maxiter=10, tolerance=1.e-6, optiter=nothing)
    residuals = zeros(typeof(tolerance), maxiter+1)
    optiter_ = isnothing(optiter) ? Int(round(maxiter/2)) : optiter
    return NewtonSolver(linsolver, maxiter, optiter_, tolerance, ScalarWrapper(0), residuals)
end

function reset_state!(s::NewtonSolver)
    s.numiter[] = 0
    fill!(s.residuals, 0)
end

function update_state!(s::NewtonSolver, r)
    s.numiter[] += 1
    s.residuals[s.numiter[]] = r 
end

getmaxiter(s::NewtonSolver) = s.maxiter
getnumiter(s::NewtonSolver) = s.numiter[]
getoptiter(s::NewtonSolver) = s.optiter

function Base.show(s::NewtonSolver)
    println("Newton solver has used $(s.numiter[]) iterations")
    println("residuals")
    for i in 1:s.numiter[]
        println("r[$i] = $(s.residuals[i])")
    end
end

function solve_nonlinear!(solver::FerriteSolver{<:NewtonSolver}, problem)
    newtonsolver = solver.nlsolver
    maxiter = newtonsolver.maxiter
    tol = newtonsolver.tolerance
    Δa = zero(getunknowns(problem))
    reset_state!(newtonsolver)
    while true
        r = calculate_convergence_criterion(problem)
        update_state!(newtonsolver, r)
        r < tol && return true
        getnumiter(newtonsolver) > maxiter && return false # Did not converge
        
        r = getresidual(problem)
        K = getjacobian(problem)
        update_guess!(Δa, K, r, newtonsolver.linsolver)
        update_problem!(problem, Δa)
    end
end