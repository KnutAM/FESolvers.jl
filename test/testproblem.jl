using ForwardDiff, LinearAlgebra
# Test problem
#= 
    | exp(t) - norm(x+1) |           | -norm(x .+ 1) |
r = | cos(t) + x[1]      | = f(t) +  | x[1]          |
    | x[3]^2 - x[2]      |           | x[3]^2-x[2]   |
=#
safenorm(x) = sqrt(sum(y->y^2+eps(y),x))    # Avoid NaN in derivative at x=0
residual(x, f) = f + [-safenorm(x .+ 1); x[1]; x[3]^2-x[2]]

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

FESolvers.getunknowns(p::TestProblem) = p.x
FESolvers.getresidual(p::TestProblem) = p.r
FESolvers.getjacobian(p::TestProblem) = p.drdx
function FESolvers.update_to_next_step!(p::TestProblem, time)
    p.f .= p.fun(time)
    p.time[1] = time
end

function FESolvers.update_problem!(p::TestProblem, Δx=nothing)
    isnothing(Δx) || (p.x .+= Δx)
    p.r .= residual(p.x, p.f)
    p.drdx .= ForwardDiff.jacobian(x_->residual(x_, p.f), p.x)
end

FESolvers.calculate_convergence_measure(p::TestProblem, args...) = norm(p.r)
FESolvers.handle_converged!(p::TestProblem) = push!(p.conv, true)
function FESolvers.postprocess!(p::TestProblem, step)
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

FESolvers.getunknowns(p::Rosenbrock) = p.x
FESolvers.getresidual(p::Rosenbrock) = p.r
FESolvers.getsystemmatrix(p::Rosenbrock, ::NewtonSolver) = p.drdx
FESolvers.getjacobian(p::Rosenbrock) = p.drdx # For TestNLSolver 
FESolvers.calculate_energy(p::Rosenbrock,x) = 100*(x[2] - x[1]^2)^2 + (1-x[1])^2
function FESolvers.update_problem!(p::Rosenbrock, Δx=nothing)
    isnothing(Δx) || (p.x .+= Δx)
    dfdx = ForwardDiff.gradient(x_->FESolvers.calculate_energy(p,x_),p.x)
    d²fdxdx = ForwardDiff.hessian(x_->FESolvers.calculate_energy(p,x_),p.x)
    p.r .= dfdx
    p.drdx .= d²fdxdx
end
FESolvers.calculate_convergence_measure(p::Rosenbrock, args...) = norm(p.r)

