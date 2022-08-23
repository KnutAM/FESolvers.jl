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
    rv::Vector{T}
end
TestProblem() = TestProblem(zeros(3), zeros(3), zeros(3,3), t->[exp(t); cos(t); zero(t)], zeros(3), zeros(0))

FerriteSolvers.getunknowns(p::TestProblem) = p.x
FerriteSolvers.getresidual(p::TestProblem) = p.r
FerriteSolvers.getjacobian(p::TestProblem) = p.drdx
function FerriteSolvers.getenergy(p::TestProblem,x)
    #first residual element
    p.f[1]*x[1] + 0.5*(x[1]*norm(x)+(x[2]^2+x[3]^2)*log(norm(x))) +
    #second residual element
    p.f[2]*x[2] + x[1]*x[2] +
    #third residual element
    p.f[3]*x[3] + (x[3]^3/3) - x[3]*x[2]
end
function FerriteSolvers.update_to_next_step!(p::TestProblem, time)
    p.f .= p.fun(time)
end

function FerriteSolvers.update_problem!(p::TestProblem, Δx=nothing)
    isnothing(Δx) || (p.x .+= Δx)
    p.r .= residual(p.x, p.f)
    p.drdx .= ForwardDiff.jacobian(x_->residual(x_, p.f), p.x)
end

FerriteSolvers.calculate_residualnorm(p::TestProblem) = norm(p.r)
FerriteSolvers.handle_converged!(::TestProblem) = nothing # Not used
FerriteSolvers.postprocess!(p::TestProblem, step) = push!(p.rv, norm(p.r))
