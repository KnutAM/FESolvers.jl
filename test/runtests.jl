using FESolvers
using Test
using LinearSolve

include("testproblem.jl")
include("TestNLSolver.jl")

include("test_linearsolvers.jl")
include("test_nlsolvers.jl")
include("test_timesteppers.jl")

@testset "QuasiStaticSolver.jl" begin
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

    x_sol = copy(FESolvers.getunknowns(problem))

    # Check using low max iterations and adaptive time stepping
    problem = TestProblem()
    ts = AdaptiveTimeStepper(timehist[end], timehist[end]; Δt_min=0.1)
    solver = QuasiStaticSolver(nlsolver=NewtonSolver(;tolerance=tol, maxiter=6), timestepper=ts)
    solve_problem!(problem, solver)
    @test length(problem.tv) > 2 # Just check that it actually didn't converge once, otherwise no point in test.
    @test problem.tv[end] == timehist[end]
    @test all(norm.(residual.(problem.xv, problem.fun.(problem.tv))) .<= tol)
    @test x_sol ≈ FESolvers.getunknowns(problem) # Check final point is the same as for fixed time stepping

    # Check using low max iterations and fixed time stepping to get no convergence
    problem = TestProblem()
    solver = QuasiStaticSolver(nlsolver=NewtonSolver(;tolerance=tol, maxiter=6), timestepper=FixedTimeStepper([0.0, 1e-3, timehist[end]]))
    @test_throws FESolvers.ConvergenceError solve_problem!(problem, solver)
    @test length(problem.tv) == 2 # Should work the first small time increase, but fail on the second. 

    # Check that we can throw another error in postprocessing and that close_problem works as expected. 
    solver = QuasiStaticSolver(nlsolver=NewtonSolver(;tolerance=tol), timestepper=FixedTimeStepper(timehist))
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

    # Smoke-test the access function for the linear nlsolver
    lps = s_linear.nlsolver
    @test FESolvers.get_num_iter(lps) == 0
    @test FESolvers.get_max_iter(lps) == 1
    @test isnan(FESolvers.get_tolerance(lps))
    @test isnan(FESolvers.get_convergence_measure(lps, 1))
    @test all(isnan, FESolvers.get_convergence_measure(lps))
    @test all(isnan, FESolvers.get_convergence_measure(lps, 1:2))
    @test_throws BoundsError FESolvers.get_convergence_measure(lps, 1:3)
end
# =#