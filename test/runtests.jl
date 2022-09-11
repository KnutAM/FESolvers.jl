using FerriteSolvers
using Test

include("testproblem.jl")
include("TestNLSolver.jl")


include("test_linearsolvers.jl")
include("test_nlsolvers.jl")
include("test_timesteppers.jl")

@testset "FerriteSolver.jl" begin
    tol = 1.e-6
    problem = TestProblem()
    timehist = [0.0, 1.0, 2.0, 3.0]
    solver = FerriteSolver(NewtonSolver(;tolerance=tol), FixedTimeStepper(timehist))
    solve_ferrite_problem!(solver, problem)
    @test problem.tv ≈ timehist[2:end]  # First time not postprocessed currently, should it?
    @test length(problem.conv) == (length(timehist)-1)  # Check handle_converged calls
    @test all(norm.(problem.rv) .<= tol)                # Check that all steps converged
    # Check that saved solutions are indeed the same solutions
    @test all(isapprox.(residual.(problem.xv, problem.fun.(problem.tv)), problem.rv))

end

