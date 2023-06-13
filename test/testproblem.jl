using ForwardDiff, LinearAlgebra
# Test problem
#= 
    | (exp(t)-1) - (norm(x+1)-1) |           | √3-norm(x .+ 1) |
r = | (cos(t)-1) + x[1]          | = f(t) +  | x[1]            |
    | x[3]^2 - x[2]              |           | x[3]^2-x[2]     |
=#
safenorm(x) = sqrt(sum(y->y^2+eps(y),x))    # Avoid NaN in derivative at x=0
residual(x, f) = f + [safenorm(ones(3))-safenorm(x .+ 1); x[1]; x[3]^2-x[2]]
timefun(t) = [exp(t)-1; cos(t)-1; zero(t)]

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
    niter::Vector{Int}
    conv::Vector{Bool}
    # Custom things for testing
    throw_at_step::Int # Test if problems throws in postprocess
    stiffness_factor::T # Scaling for stiffness (to slow down convergence)
end
TestProblem(;throw_at_step=-1, stiffness_factor=1.0) = TestProblem(
    zeros(3), zeros(3), zeros(3,3), # x, r, drdx
    timefun,                        # fun
    zeros(3), zeros(1),             # f, time
    Vector{Float64}[], Vector{Float64}[], Float64[], Int[], Int[], Bool[], throw_at_step, stiffness_factor
    )

FESolvers.getunknowns(p::TestProblem) = p.x
FESolvers.getresidual(p::TestProblem) = p.r
FESolvers.getjacobian(p::TestProblem) = p.drdx
function FESolvers.update_to_next_step!(p::TestProblem, time)
    p.f .= p.fun(time)
    p.time[1] = time
end

function FESolvers.update_problem!(p::TestProblem, Δx, _)
    isnothing(Δx) || (p.x .+= Δx)
    p.r .= residual(p.x, p.f)
    p.drdx .= ForwardDiff.jacobian(x_->residual(x_, p.f), p.x)
    p.drdx .*= p.stiffness_factor
end

FESolvers.calculate_convergence_measure(p::TestProblem, args...) = norm(p.r)
FESolvers.handle_converged!(p::TestProblem) = push!(p.conv, true)
function FESolvers.postprocess!(p::TestProblem, step, solver)
    step == p.throw_at_step && throw(TestError())
    push!(p.xv, copy(p.x))
    push!(p.rv, copy(p.r))
    push!(p.tv, p.time[1])
    push!(p.steps, step)
    push!(p.niter, FESolvers.getnumiter(solver.nlsolver))
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
    p.u[1] = p.dbcfun(time)
end
function FESolvers.update_problem!(p::LinearTestProblem, Δu, update_spec)
    isnothing(Δu) || (p.u .+= Δu)
    FESolvers.should_update_residual(update_spec) && (p.r .+= p.K0*p.u; p.r[1]=0)
    if FESolvers.should_update_jacobian(update_spec)
        p.K .= p.K0
        # "apply_zero!"
        p.K[:,1] .= 0; p.K[1,:] .= 0; p.K[1,1] = p.K0[1,1]
    end
end

FESolvers.handle_converged!(p::LinearTestProblem) = push!(p.conv, true)
FESolvers.postprocess!(p::LinearTestProblem, step) = push!(p.uend, last(p.u))


# Test case for nonlinear solvers
@kwdef struct Rosenbrock{T}
    a::T=1.0
    b::T=100.0
    x::Vector{T}=[0.5*a, 0.5*a^2] # minimum at [a,a^2]
    r::Vector{T}=similar(x)
    drdx::Matrix{T}=zeros(eltype(x), length(x), length(x))
end

FESolvers.getunknowns(p::Rosenbrock) = p.x
FESolvers.getresidual(p::Rosenbrock) = p.r
FESolvers.getsystemmatrix(p::Rosenbrock, ::NewtonSolver) = p.drdx
FESolvers.getjacobian(p::Rosenbrock) = p.drdx # For TestNLSolver 
FESolvers.calculate_energy(p::Rosenbrock,x) = p.b*(x[2] - x[1]^2)^2 + (p.a-x[1])^2

function FESolvers.update_problem!(p::Rosenbrock, Δx, update_spec)
    isnothing(Δx) || (p.x .+= Δx)
    if FESolvers.should_update_residual(update_spec)
        p.r .= ForwardDiff.gradient(x_->FESolvers.calculate_energy(p,x_),p.x)
    end
    if FESolvers.should_update_jacobian(update_spec)
        if FESolvers.get_update_type(update_spec) == :steepestdescent
            copyto!(p.drdx, FESolvers.getdescentpreconditioner(p))
        else
            p.drdx .= ForwardDiff.hessian(x_->FESolvers.calculate_energy(p,x_),p.x)
        end
    end
end
FESolvers.calculate_convergence_measure(p::Rosenbrock, args...) = norm(p.r)
