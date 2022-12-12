using ForwardDiff, LinearAlgebra
# Test problem
#= 
    | exp(t) - norm(x+1) |           | -norm(x .+ 1) |
r = | cos(t) + x[1]      | = f(t) +  | x[1]          |
    | x[3]^2 - x[2]      |           | x[3]^2-x[2]   |
=#
safenorm(x) = sqrt(sum(y->y^2+eps(y),x))    # Avoid NaN in derivative at x=0
residual(x, f) = f + [-safenorm(x .+ 1); x[1]; x[3]^2-x[2]]

struct TestError <: Exception end

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
    throw_at_step::Int # Test if problems throws in postprocess
end
TestProblem(;throw_at_step=-1) = TestProblem(
    zeros(3), zeros(3), zeros(3,3), # x, r, drdx
    t->[exp(t); cos(t); zero(t)],   # fun
    zeros(3), zeros(1),             # f, time
    Vector{Float64}[], Vector{Float64}[], Float64[], Int[], Bool[], throw_at_step
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
    step == p.throw_at_step && throw(TestError())
    push!(p.xv, copy(p.x))
    push!(p.rv, copy(p.r))
    push!(p.tv, p.time[1])
    push!(p.steps, step)
end

struct LinearTestProblem{T,FF<:Function,DF<:Function}
    K0::Matrix{T}
    K::Matrix{T}
    u::Vector{T}
    r::Vector{T}
    forcefun::FF        # Time varying force
    dbcfun::DF          # Time varying boundary condition
    dbval::Vector{T}    # Save time varying boundary condition
    # Some postprocessing to check results 
    uend::Vector{T}
    conv::Vector{Bool}
end
function LinearTestProblem(K0::Matrix{T}, forcefun, dbcfun) where T
    LinearTestProblem(K0, copy(K0), zeros(T,size(K0,1)), zeros(T, size(K0,1)), forcefun, dbcfun, [zero(T),], T[], Bool[])
end
function LinearTestProblem(k; forcefun=identity, dbcfun=identity)
    K0 = [k -k 0.0; -k 2k -k; 0.0 -k k]    # Two springs
    return LinearTestProblem(K0, forcefun, dbcfun)
end
FESolvers.getunknowns(p::LinearTestProblem) = p.u 
FESolvers.getresidual(p::LinearTestProblem) = p.r
FESolvers.getjacobian(p::LinearTestProblem) = p.K 
function FESolvers.update_to_next_step!(p::LinearTestProblem, time)
    fill!(p.r, 0); 
    p.r[end]=-p.forcefun(time)
    p.dbval[1] = p.dbcfun(time)
end
function FESolvers.update_problem!(p::LinearTestProblem, Δu=nothing)
    isnothing(Δu) || (p.u .+= Δu)
    p.K .= p.K0
    p.u[1] = p.dbval[1]
    p.r .+= p.K*p.u
    # "apply_zero!"
    p.K[:,1] .= 0; p.K[1,:] .= 0; p.K[1,1] = p.K0[1,1]
    p.r[1] = 0
end

FESolvers.handle_converged!(p::LinearTestProblem) = push!(p.conv, true)
FESolvers.postprocess!(p::LinearTestProblem, step) = push!(p.uend, last(p.u))


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
