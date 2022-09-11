using ForwardDiff, LinearAlgebra
# Test problem
#= 
    | exp(t) - norm(x)  |           | -norm(x)      |
r = | cos(t) + x[1]     | = f(t) +  | x[1]          |
    | x[3]^2 - x[2]     |           | x[3]^2-x[2]   |
=#
residual(x, f) = f + [-norm(x); x[1]; x[3]^2-x[2]]

struct TestProblem{T,FT}
    x::Vector{T}
    r::Vector{T}
    drdx::Matrix{T}
    fun::FT
    f::Vector{T}
    time::Vector{T}
    # History to be saved
    rv::Vector{Vector{T}}
    xv::Vector{Vector{T}}
    tv::Vector{T}
    steps::Vector{Int}
    conv::Vector{Bool}
end
TestProblem() = TestProblem(
    zeros(3), zeros(3), zeros(3,3), # x, r, drdx
    t->[exp(t); cos(t); zero(t)],   # fun
    zeros(3), zeros(1),             # f, time
    Vector{Float64}[], Vector{Float64}[], Float64[], Int[], Bool[]
    )

FerriteSolvers.getunknowns(p::TestProblem) = p.x
FerriteSolvers.getresidual(p::TestProblem) = p.r
FerriteSolvers.getjacobian(p::TestProblem) = p.drdx
function FerriteSolvers.update_to_next_step!(p::TestProblem, time)
    p.f .= p.fun(time)
    p.time[1] = time
end

function FerriteSolvers.update_problem!(p::TestProblem, Δx=nothing)
    isnothing(Δx) || (p.x .+= Δx)
    p.r .= residual(p.x, p.f)
    p.drdx .= ForwardDiff.jacobian(x_->residual(x_, p.f), p.x)
end

FerriteSolvers.calculate_convergence_measure(p::TestProblem) = norm(p.r)
FerriteSolvers.handle_converged!(p::TestProblem) = push!(p.conv, true)
function FerriteSolvers.postprocess!(p::TestProblem, step)
    push!(p.xv, copy(p.x))
    push!(p.rv, copy(p.r))
    push!(p.tv, p.time[1])
    push!(p.steps, step)
end


# Test case for nonlinear solvers
struct Rosenbrock{T}
    x::Vector{T}
    r::Vector{T}
    drdx::Matrix{T}
end

Rosenbrock() = Rosenbrock([-1.0,1.0], zeros(2), zeros(2,2))

FerriteSolvers.getunknowns(p::Rosenbrock) = p.x
FerriteSolvers.getresidual(p::Rosenbrock) = p.r
FerriteSolvers.getsystemmatrix(p::Rosenbrock, ::NewtonSolver) = p.drdx
FerriteSolvers.getjacobian(p::Rosenbrock) = p.drdx # For TestNLSolver 
FerriteSolvers.calculate_energy(p::Rosenbrock,x) = 100*(x[2] - x[1]^2)^2 + (1-x[1])^2
function FerriteSolvers.update_problem!(p::Rosenbrock, Δx=nothing)
    isnothing(Δx) || (p.x .+= Δx)
    dfdx = ForwardDiff.gradient(x_->FerriteSolvers.calculate_energy(p,x_),p.x)
    d²fdxdx = ForwardDiff.hessian(x_->FerriteSolvers.calculate_energy(p,x_),p.x)
    p.r .= dfdx
    p.drdx .= d²fdxdx
end
FerriteSolvers.calculate_convergence_measure(p::Rosenbrock) = norm(p.r)

