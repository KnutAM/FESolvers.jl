using FerriteSolvers
using Test

include("testproblem.jl")

@testset "FerriteSolver.jl" begin
    tol = 1.e-6
    problem = TestProblem()
    solver = FerriteSolver(NewtonSolver(;tolerance=tol), FixedTimeStepper([0.0, 1.0, 2.0, 3.0]))
    solve_ferrite_problem!(solver, problem)
    @test all(isapprox.(problem.rv, 0.0; atol=tol))
    ls = ArmijoGoldstein(μ=1e-4,β=0.5,τ0=1.0)
    steepestdescent= FerriteSolver(SteepestDescent(;maxiter=30000,tolerance=1e-3,linesearch=ls), FixedTimeStepper([0.0, 0.1]))
    newton_ls = FerriteSolver(NewtonSolver(;maxiter=30,tolerance=tol,linesearch=ls), FixedTimeStepper([0.0, 0.1]))
    newton = FerriteSolver(NewtonSolver(;maxiter=30,tolerance=tol), FixedTimeStepper([0.0, 0.1]))
    for solver in [steepestdescent,newton_ls,newton]
        problem = Rosenbrock() 
        solve_ferrite_problem!(solver, problem)
        @test all(isapprox.(problem.rv, 0.0; atol=1e-2))
        @test all(isapprox.(problem.x, 1.0; atol=1e-5))
    end
end
