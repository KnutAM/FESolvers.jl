# A simple custom nonlinear solver that does linesearch in the first iteration. 
Base.@kwdef struct TestNLSolver{LineSearch,T}
    ls::LineSearch
    maxiter::Int
    tolerance::T
    numiter::Vector{Int}=[0]
    residuals::Vector{T}=zeros(typeof(tolerance), maxiter)
end
FESolvers.getmaxiter(s::TestNLSolver) = s.maxiter
FESolvers.gettolerance(s::TestNLSolver) = s.tolerance
FESolvers.getnumiter(s::TestNLSolver) = s.numiter[1]
FESolvers.get_convergence_measures(s::TestNLSolver, i=1:s.numiter[1]) = s.residuals[i]
FESolvers.get_initial_update_spec(::TestNLSolver) = FESolvers.UpdateSpec(;jacobian=false, residual=false)

function FESolvers.update_state!(s::TestNLSolver, problem, r)
    s.numiter .+= 1
    s.residuals[FESolvers.getnumiter(s)] = r
end
function FESolvers.reset_state!(s::TestNLSolver)
    s.numiter .= 0
end
FESolvers.get_first_update_spec(::TestNLSolver, _) = FESolvers.UpdateSpec(;jacobian=true, residual=true)
FESolvers.get_update_spec(::TestNLSolver) = FESolvers.UpdateSpec(;jacobian=true, residual=true)

function FESolvers.calculate_update!(Δa, problem, nlsolver::TestNLSolver)
    Δa .= -FESolvers.getjacobian(problem)\FESolvers.getresidual(problem)
    if FESolvers.getnumiter(nlsolver) == 1 
        FESolvers.linesearch!(Δa, problem, nlsolver.ls)
    end
    return Δa
end