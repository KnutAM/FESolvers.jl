using FerriteSolvers
using Test

include("testproblem.jl")

@testset "FerriteSolver.jl" begin
    tol = 1.e-6
    problem = TestProblem()
    solver = FerriteSolver(NewtonSolver(;tolerance=tol), FixedTimeStepper([0.0, 1.0, 2.0, 3.0]))
    solve_ferrite_problem!(solver, problem)
    @test all(isapprox.(problem.rv, 0.0; atol=tol))
    #ls = ArmijoGoldstein(μ=0.15,β=0.9,τ0=5.0)
    ls = NoLineSearch()
    solver = FerriteSolver(SteepestDescent(;maxiter=2000,tolerance=tol,linesearch=ls), FixedTimeStepper([0.0, 1.0, 2.0, 3.0]))
    problem = TestProblem()
    solve_ferrite_problem!(solver, problem)
    @test all(isapprox.(problem.rv, 0.0; atol=tol))
end
