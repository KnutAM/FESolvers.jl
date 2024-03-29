function run_timestepper(solver, convergence_function=Returns(true))
    t = FESolvers.initial_time(solver.timestepper)
    t_old = t
    step = 1
    converged = true
    timehistory = [t]
    while !FESolvers.islaststep(solver.timestepper, t, step)
        t, step = FESolvers.update_time(solver, t, step, converged)
        Δt = t - t_old
        converged = convergence_function(t, Δt, step)
        if converged
            t_old = t
            push!(timehistory, t)
        end
    end
    return timehistory
end

FESolvers.getnumiter(::Nothing) = 1
FESolvers.getmaxiter(::Nothing) = 4

@testset "timesteppers" begin
    
    @testset "FixedTimeStepper" begin
        t = sort(rand(100))
        solver = QuasiStaticSolver(nothing, FixedTimeStepper(t))
        th = run_timestepper(solver, (args...) -> true)
        @test t ≈ th
        num_steps=10; Δt=0.1
        solver = QuasiStaticSolver(nothing, FixedTimeStepper(num_steps=num_steps, Δt=Δt))
        th = run_timestepper(solver, (args...) -> true)
        @test all( Δt .≈ (th[2:end]-th[1:end-1]))
        @test length(th) == (num_steps+1)

        @test_throws FESolvers.ConvergenceError run_timestepper(solver, (time, args...) -> time < sum(t)/length(t))
    end

    @testset "AdaptiveTimeStepper" begin
        # Errors 
        @test_throws ArgumentError AdaptiveTimeStepper(10.0, 0.1) # Flip Δt_init and t_end
        @test_throws ArgumentError AdaptiveTimeStepper(0.1, 10.0; t_start=20.0)
        @test_throws ArgumentError AdaptiveTimeStepper(0.1, 10.0; Δt_max=0.01)
        # Warnings
        @test_logs (:warn,) match_mode=:any AdaptiveTimeStepper(0.1, 10.0; change_factor=1.2)
        @test_logs (:warn,) match_mode=:any AdaptiveTimeStepper(0.1, 10.0; change_factor=-0.2)
        @test_logs (:warn,) match_mode=:any AdaptiveTimeStepper(0.1, 10.0; optiter_ratio=1.2)
        @test_logs (:warn,) match_mode=:any AdaptiveTimeStepper(0.1, 10.0; optiter_ratio=-0.2)
        @test_logs (:warn,) match_mode=:any AdaptiveTimeStepper(0.1, 10.0; k=-0.2)

        t_end = 2.0
        Δt = 0.1
        Δt_min = 0.05
        Δt_max = 0.2

        ts = AdaptiveTimeStepper(Δt, t_end; Δt_min=Δt_min, Δt_max=Δt_max)
        solver = QuasiStaticSolver(nothing, ts)
        @test FESolvers.initial_time(ts) ≈ 0.0
        t, step = FESolvers.update_time(solver, 0.0, 1, true)
        @test t ≈ Δt 
        @test step == 2

        cf(t, Δt, _) = t < 1.0 || Δt < (Δt_min+sqrt(eps(Δt_min)))

        th = run_timestepper(QuasiStaticSolver(nothing,ts), cf)
        
        Δth = th[2:end]-th[1:end-1]
        @test Δth[1] .≈ Δt
        @test maximum(Δth) ≈ Δt_max 
        @test minimum(Δth[1:end-2]) .≈ Δt_min
        @test minimum(Δth) > Δt_min/2

        t_end = 0.91
        Δt = 0.1
        ts = AdaptiveTimeStepper(Δt, t_end; Δt_max = Δt)
        th = run_timestepper(QuasiStaticSolver(nothing,ts), (args...)->true)
        Δth = th[2:end]-th[1:end-1]
        # Last two steps should take half each when the reminder is less then Δt_min
        @test Δth[end] ≈ Δth[end-1] ≈ (mod(t_end, Δt) + Δt)/2
        @test all(Δth[1:end-2] .≈ 0.1)

        # Test errors
        Δt = 0.1
        ts = AdaptiveTimeStepper(Δt, 3*Δt; Δt_min=Δt)
        # Not converged in first iteration (setup issue)
        @test_throws ArgumentError FESolvers.update_time(ts, nothing, 0.0, #=step=# 1, #=converged=# false)
        # Cannot reduce time step further
        @test_throws FESolvers.ConvergenceError FESolvers.update_time(ts, nothing, Δt, 2, false)
    end

end