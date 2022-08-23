using FerriteSolvers
using Test

include("testproblem.jl")

@testset "FerriteSolver.jl" begin
    tol = 1.e-6
    problem = TestProblem()
    solver = FerriteSolver(NewtonSolver(;tolerance=tol), FixedTimeStepper([0.0, 1.0, 2.0, 3.0]))
    solve_ferrite_problem!(solver, problem)
    @test all(isapprox.(problem.rv, 0.0; atol=tol))
    ls = ArmijoGoldstein(μ=0.50,β=0.9,τ0=2.0)
    solver = FerriteSolver(SteepestDescent(;maxiter=20000,tolerance=tol,linesearch=ls), FixedTimeStepper([0.0, 0.1]))
    problem = Rosenbrock()
    solve_ferrite_problem!(solver, problem)
    @test all(isapprox.(problem.rv, 0.0; atol=1e-2))
    @test all(isapprox.(problem.x, 1.0; atol=1e-5))
    @show problem.rv
    @show problem.x
end
