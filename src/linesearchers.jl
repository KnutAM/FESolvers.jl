abstract type AbstractLineSearch end
"Singleton that does not perform a linesearch when used in a nonlinear solver"
struct NoLineSearch <: AbstractLineSearch end

@doc raw"""
    Armijo-Goldstein{T}(;β=0.9,μ=0.01,τ0=1.0,τmin=1e-4)
Backtracking line search based on the Armijo-Goldstein condition

```math
\Pi(\boldsymbol{u} + \tau \Delta\boldsymbol{u}) \leq \Pi(\boldsymbol{u}) - \mu\tau\delta\Pi(\boldsymbol{u})[\Delta \boldsymbol{u}]
```

where \$\Pi\$ is the potential, \$\tau\$ the stepsize, and \$\delta\Pi\$ the residuum.

#Fields
- `β::T = 0.9` constant factor that changes the steplength τ in each iteration
- `μ::T = 0.01` second constant factor that determines how much the potential needs to decrease additionally
- `τ0::T = 1.0` start stepsize 
- `τmin::T = 1e-4` minimal stepsize
"""
Base.@kwdef struct ArmijoGoldstein{T} <: AbstractLineSearch
    β::T = 0.9
    μ::T = 0.10
    τ0::T = 1.0
    τmin::T = 1e-5
end

linesearch!(searchdirection, problem, ls::NoLineSearch) = nothing

function linesearch!(searchdirection, problem, ls::ArmijoGoldstein)
    τ = ls.τ0; μ = ls.μ; β = ls.β
    𝐮 = getunknowns(problem)
    Π₀ = calculate_energy(problem,𝐮)
    δΠ₀ = getresidual(problem)
    Πₐ = calculate_energy(problem,𝐮 .+ τ .* searchdirection)
    armijo = Πₐ - Π₀ - μ * τ * δΠ₀'searchdirection
    
    while armijo > 0 && !isapprox(armijo,0.0,atol=1e-8)
        τ *= β
        Πₐ = calculate_energy(problem,𝐮 .+ τ .* searchdirection)
        armijo = Πₐ - Π₀ - μ * τ * δΠ₀'searchdirection
    end
    τ = max(ls.τmin,τ)
    searchdirection .*= τ
end