# A simple custom nonlinear solver that does linesearch in the first iteration. 
Base.@kwdef struct TestNLSolver{LineSearch,T}
    ls::LineSearch
    maxiter::Int
    tolerance::T
    status=FESolvers.SolverState(maxiter)
end
FESolvers.get_max_iter(s::TestNLSolver) = s.maxiter
FESolvers.get_tolerance(s::TestNLSolver) = s.tolerance
FESolvers.get_solver_state(s::TestNLSolver) = s.status
#=
FESolvers.get_num_iter(s::TestNLSolver) = s.numiter[1]
FESolvers.get_convergence_measure(s::TestNLSolver, i=1:s.numiter[1]) = s.residuals[i]

function FESolvers.update_state!(s::TestNLSolver, problem, r)
    s.numiter .+= 1
    s.residuals[FESolvers.get_num_iter(s)] = r
end
function FESolvers.reset_state!(s::TestNLSolver)
    s.numiter .= 0
end
=#
FESolvers.get_initial_update_spec(::TestNLSolver) = FESolvers.UpdateSpec(;jacobian=false, residual=false)
FESolvers.get_first_update_spec(::TestNLSolver) = FESolvers.UpdateSpec(;jacobian=true, residual=true)
FESolvers.get_update_spec(::TestNLSolver) = FESolvers.UpdateSpec(;jacobian=true, residual=true)

function FESolvers.calculate_update!(Δa, problem, nlsolver::TestNLSolver)
    Δa .= -FESolvers.getjacobian(problem)\FESolvers.getresidual(problem)
    if FESolvers.get_num_iter(nlsolver) == 1 
        FESolvers.linesearch!(Δa, problem, nlsolver.ls)
    end
    return Δa
end