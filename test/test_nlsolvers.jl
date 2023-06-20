@testset "nlsolvers" begin
    function test_basic_functions(nlsolver, tol, maxiter)
        @test FESolvers.get_tolerance(nlsolver) ≈ tol
        @test FESolvers.get_max_iter(nlsolver) == maxiter
    end
    tol = 1.e-6
    maxiter = 30
    ls = ArmijoGoldstein(μ=1e-3, β=0.5, τ0=1.0)
    steepestdescent_tol = 1e-4
    steepestdescent = SteepestDescent(;maxiter=maxiter, tolerance=steepestdescent_tol, linesearch=ls)
    test_basic_functions(steepestdescent, steepestdescent_tol, maxiter)
    newton_ls = NewtonSolver(;maxiter=maxiter, tolerance=tol, linesearch=ls)
    newton = NewtonSolver(;maxiter=maxiter, tolerance=tol)
    test_basic_functions(newton, tol, maxiter)
    newton_nofirst = NewtonSolver(;maxiter=30, tolerance=tol, update_jac_first=false)
    custom = TestNLSolver(;maxiter=maxiter, tolerance=tol, ls=ls)
    adaptive = FESolvers.AdaptiveNewtonSolver(;maxiter=maxiter, tolerance=tol, 
        update_types=(:steepestdescent, :true),
        linesearch=ls,
        switch_criterion=FESolvers.NumIterSwitch(3)
        )
    for nlsolver in [steepestdescent, newton_ls, newton, custom, newton_nofirst, adaptive]
        if nlsolver===steepestdescent
            problem = Rosenbrock(;a=0.1, b=0.2) # Make it easy enough for pure linesearch
        else
            problem = Rosenbrock(;a=0.2, b=1.0) # A bit harder, but still quite easy
        end
        if FESolvers.should_do_initial_update(nlsolver) 
            FESolvers.update_problem!(problem, nothing, FESolvers.get_initial_update_spec(nlsolver))
        end
        FESolvers.solve_nonlinear!(problem, nlsolver)
        @test FESolvers.is_converged(nlsolver)
        @test FESolvers.get_num_iter(nlsolver) > 3 # Test makes sense; some iterations required
        @test isapprox(norm(problem.r), 0.0; atol=FESolvers.get_tolerance(nlsolver))
        @test isapprox(problem.x[1], problem.a; atol=1e-2) # Just to ensure test is sound
        @test isapprox(problem.x[2], problem.a^2; atol=1e-2)
    end
    # Sanity check that the AdaptiveNewtonSolver needs more iterations since it starts by a few 
    # steepest descent iterations
    @test FESolvers.get_num_iter(adaptive) > FESolvers.get_num_iter(newton)

    # Test that it runs properly when it doesn't converge 
    newton = NewtonSolver(;tolerance=-1.0)
    FESolvers.solve_nonlinear!(Rosenbrock(), newton)
    @test !FESolvers.is_converged(newton)
    state = FESolvers.get_solver_state(newton)
    @test state.status.reason == :nonlinearsolver

    FESolvers.solve_nonlinear!(Rosenbrock(), newton_nofirst) # newton_nofirst is converged
    @test !FESolvers.is_converged(newton_nofirst) # Without updating problem, matrix should be all zero (for now, just throws)
    state = FESolvers.get_solver_state(newton_nofirst)
    @test state.status.reason == :linearsolver
    state.status = FESolvers.ConvergenceStatus(false, :test)
    FESolvers.solve_nonlinear!(Rosenbrock(), newton_nofirst)
    @test FESolvers.is_converged(newton_nofirst) # If did not converge in last step, should calculate jacobian again
end

@testset "AdaptiveNewtonSolver switches" begin
    makesolver(switch) = FESolvers.AdaptiveNewtonSolver(;
        update_types=[1, 2],
        switch_criterion=switch,
        maxiter=10
        )
    function test_expected!(nls; expected_reset, expected_nr)
        reset, nr = FESolvers.switch_information(nls.switch_criterion, nls)
        @test reset == expected_reset
        @test nr == expected_nr
        # Make the updates that should be made
        nls.update_type_nr = expected_nr
        nls.reset_problem = expected_reset
    end

    @testset "NumIterSwitch" begin
        nls = makesolver(FESolvers.NumIterSwitch(4))
        state = FESolvers.get_solver_state(nls)
        test_expected!(nls, expected_reset=false, expected_nr=1)
        state.numiter = 5
        test_expected!(nls, expected_reset=false, expected_nr=2)
    end
    @testset "ToleranceSwitch" begin
        nls = makesolver(FESolvers.ToleranceSwitch(2.0))
        state = FESolvers.get_solver_state(nls)
        state.numiter = 1
        state.residuals[1] = 3.0 # Should use first method
        test_expected!(nls; expected_reset=false, expected_nr=1)

        state.numiter = 2
        state.residuals[2] = 1.0 # Should use second method
        test_expected!(nls; expected_reset=false, expected_nr=2)

        state.numiter = 3
        state.residuals[3] = 3.0 # Should use first method again, and reset
        test_expected!(nls; expected_reset=true, expected_nr=1)
    end
    @testset "IncreaseSwitch" begin
        nls = makesolver(FESolvers.IncreaseSwitch(num_slow=2))
        state = FESolvers.get_solver_state(nls)
        state.numiter = 2
        state.residuals[1] = 9.0
        state.residuals[2] = 8.0 # Decreasing
        test_expected!(nls; expected_reset=false, expected_nr=1)
        
        state.numiter = 3
        state.residuals[3] = 9.0 # Increasing
        test_expected!(nls; expected_reset=true, expected_nr=2)
        
        state.numiter = 4
        state.residuals[4] = 10.0 # Still increasing 
        test_expected!(nls; expected_reset=false, expected_nr=2)
        
        state.numiter = 5
        state.residuals[5] = 9.0 # Decreasing 1 time
        test_expected!(nls; expected_reset=false, expected_nr=2)

        state.numiter = 6
        state.residuals[6] = 8.8 # Decreasing 2 times
        test_expected!(nls; expected_reset=false, expected_nr=2)

        state.numiter = 7
        state.residuals[7] = 8.4 # Decreasing 3 times > num_slow
        test_expected!(nls; expected_reset=false, expected_nr=1)

        state.numiter = 8
        state.residuals[8] = 9.0 # Increasing again
        test_expected!(nls; expected_reset=true, expected_nr=2)
    end
end