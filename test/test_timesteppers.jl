function run_timestepper(solver, convergence_function)
    t = FerriteSolvers.initial_time(solver.timestepper)
    t_old = t
    step = 1
    converged = true
    timehistory = [t]
    while !FerriteSolvers.islaststep(solver.timestepper, t, step)
        t, step = FerriteSolvers.update_time(solver, t, step, converged)
        Δt = t - t_old
        converged = convergence_function(t, Δt, step)
        if converged
            t_old = t
            push!(timehistory, t)
        end
    end
    return timehistory
end

@testset "timesteppers" begin
    
    @testset "FixedTimeStepper" begin
        t = sort(rand(100))
        solver = FerriteSolver(nothing, FixedTimeStepper(t))
        th = run_timestepper(solver, (args...) -> true)
        @test t ≈ th
        num_steps=10; Δt=0.1
        solver = FerriteSolver(nothing, FixedTimeStepper(num_steps=num_steps, Δt=Δt))
        th = run_timestepper(solver, (args...) -> true)
        @test all( Δt .≈ (th[2:end]-th[1:end-1]))
        @test length(th) == (num_steps+1)

        @test_throws FerriteSolvers.ConvergenceError run_timestepper(solver, (time, args...) -> time < sum(t)/length(t))
    end

end