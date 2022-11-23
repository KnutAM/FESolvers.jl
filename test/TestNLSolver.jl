# A simple custom nonlinear solver that does linesearch in the first iteration. 
Base.@kwdef struct TestNLSolver{LineSearch,T}
    ls::LineSearch
    maxiter::Int
    tolerance::T
end
FESolvers.getmaxiter(s::TestNLSolver) = s.maxiter
FESolvers.gettolerance(s::TestNLSolver) = s.tolerance

function FESolvers.calculate_update!(Δa, problem, nlsolver::TestNLSolver, iter)
    Δa .= -FESolvers.getjacobian(problem)\FESolvers.getresidual(problem)
    iter == 1 && FESolvers.linesearch!(Δa, problem, nlsolver.ls)
    return Δa
end