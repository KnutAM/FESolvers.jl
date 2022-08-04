module FerriteSolvers
include("userfunctions.jl")

export solve_ferrite_problem!

export FerriteSolver
export NewtonSolver
export FixedTimeStepper
export BackslashSolver

struct ConvergenceError <: Exception
    msg::String
end

struct BackslashSolver end

struct FerriteSolver{NLS,TS,LS}
    nlsolver::NLS 
    timestepper::TS
    linsolver::LS
end
FerriteSolver(nlsolver, timestepper) = FerriteSolver(nlsolver, timestepper, BackslashSolver())

struct NewtonSolver{T}
    maxiter::Int 
    tolerance::T
    numiter::Vector{Int}  # Last step number of iterations
    residuals::Vector{T}  # Last step residual history
end

function NewtonSolver(;maxiter=10, tolerance=1.e-6)
    residuals = zeros(typeof(tolerance), maxiter+1)
    return NewtonSolver(maxiter, tolerance, [zero(maxiter)], residuals)
end

function reset!(s::NewtonSolver)
    fill!(s.numiter, 0)
    fill!(s.residuals, 0)
end

function set_converged_status(s::NewtonSolver, numiter)
    s.numiter .= numiter
end

function print_solver(s::NewtonSolver)
    println("Newton solver has used $(s.numiter[1]) iterations")
    println("residuals")
    for i in 1:s.numiter[1]
        println("r[$i] = $(s.residuals[i])")
    end
end

struct FixedTimeStepper{T}
    t::Vector{T}
end
function FixedTimeStepper(num_steps::Int, Δt=1, t_start=zero(Δt))
    return FixedTimeStepper(t_start .+ collect(0:Δt:((num_steps)*Δt)))
end

initial_time(ts::FixedTimeStepper) = first(ts.t)
islaststep(ts::FixedTimeStepper, _, step) = step >= length(ts.t)
function update_time!(s::FerriteSolver{<:Any,<:FixedTimeStepper}, t, step, converged)
    if !converged
        msg = "nonlinear solve failed and FixedTimeStepper cannot adjust time step"
        throw(ConvergenceError(msg))
    end
    return s.timestepper.t[step+1], step+1
end

function solve_ferrite_problem!(solver::FerriteSolver{<:NewtonSolver}, problem)
    t = initial_time(solver.timestepper)
    step = 1
    converged = true
    while !islaststep(solver.timestepper, t, step)
        t, step = update_time!(solver, t, step, converged)
        update_to_next_step!(problem, t)
        # Can make improved guess here based on previous values and send to update_problem! 
        # Alternatively, this can be done inside update_loads! by the user if desired. 
        update_problem!(problem)
        converged = solve_nonlinear!(solver, problem)
        if converged
            handle_converged!(problem)
            postprocess!(problem, step)
        else
            println("the nonlinear solver didn't converge")
            print_solver(solver.nlsolver)
        end
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
        update_guess!(Δa, K, r, solver.linsolver)
        update_problem!(problem, Δa)
    end
end

update_guess!(Δa, K, r, ::BackslashSolver) = (Δa .= -K\r) 

end
