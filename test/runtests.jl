using FerriteSolvers
using Test

include("testproblem.jl")

@testset "FerriteSolver.jl" begin
    tol = 1.e-6
    problem = TestProblem()
    solver = FerriteSolver(NewtonSolver(;tolerance=tol), FixedTimeStepper([0.0, 1.0, 2.0, 3.0]))
    solve_ferrite_problem!(solver, problem)
    @test all(isapprox.(problem.rv, 0.0; atol=tol))
    solver = FerriteSolver(SteepestDescent(;tolerance=tol), FixedTimeStepper([0.0, 1.0, 2.0, 3.0]))
    solve_ferrite_problem!(solver, problem)
    @test all(isapprox.(problem.rv, 0.0; atol=tol))
end
