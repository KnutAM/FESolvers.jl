using FESolvers
using Test

include("testproblem.jl")
include("TestNLSolver.jl")


include("test_linearsolvers.jl")
include("test_nlsolvers.jl")
include("test_timesteppers.jl")

@testset "QuasiStaticSolver.jl" begin
    # Check order of nlsolver and timestepper to avoid unwanted API changes
    @test QuasiStaticSolver(1,2).nlsolver == 1
    
    tol = 1.e-6
    problem = TestProblem()
    timehist = [0.0, 1.0, 2.0, 3.0]
    solver = QuasiStaticSolver(nlsolver=NewtonSolver(;tolerance=tol), timestepper=FixedTimeStepper(timehist))
    solve_problem!(problem, solver)    
    
    @test problem.tv ≈ timehist
    @test length(problem.conv) == (length(timehist)-1)  # Check handle_converged calls
    @test all(norm.(problem.rv) .<= tol)                # Check that all steps converged
    # Check that saved solutions are indeed the same solutions
    @test all(isapprox.(residual.(problem.xv, problem.fun.(problem.tv)), problem.rv))

    problem = TestProblem(;throw_at_step=3)
    FESolvers.close_problem(p::TestProblem) = push!(p.steps, -1)
    @test_throws TestError solve_problem!(problem, solver)
    @test length(problem.steps) == 3    # postprocess for TestProblem doesn't write when hitting "throw_at_step",
                                        # but one step is added when close_problem is called. 
    @test last(problem.steps) == -1     # (This tests that after throwing close_problem is still called)

    # Test the fully linear case 
    k = 1.0
    p_linear = LinearTestProblem(k;dbcfun=t->0.1*t)
    s_linear = QuasiStaticSolver(nlsolver=LinearProblemSolver(), timestepper=FixedTimeStepper(timehist))
    solve_problem!(p_linear, s_linear)
    
    ubc = 0.1*timehist   # Boundary condition 
    # Two springs in series, stiffness is k/2. Displacement Δu = 2f/k
    fend = copy(timehist)
    uend = ubc + 2*fend/k 
    @test p_linear.u[1] ≈ last(ubc)
    @test last(p_linear.u) ≈ last(p_linear.uend)
    @test p_linear.uend ≈ uend
    @test length(p_linear.conv) == length(timehist)-1
end
