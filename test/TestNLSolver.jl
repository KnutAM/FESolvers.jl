# A simple custom nonlinear solver that does linesearch in the first iteration. 
Base.@kwdef struct TestNLSolver{LineSearch,T}
    ls::LineSearch
    maxiter::Int
    tolerance::T
end
FerriteSolvers.getmaxiter(s::TestNLSolver) = s.maxiter
FerriteSolvers.gettolerance(s::TestNLSolver) = s.tolerance

function FerriteSolvers.calculate_update(problem, nlsolver::TestNLSolver, iter)
    Δa = -FerriteSolvers.getjacobian(problem)\FerriteSolvers.getresidual(problem)
    iter == 1 && FerriteSolvers.linesearch!(Δa, problem, nlsolver.ls)
    return Δa
end