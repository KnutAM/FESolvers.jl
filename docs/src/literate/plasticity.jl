# # Plasticity

# This example is based on 
# [Ferrite.jl's plasticity example](https://ferrite-fem.github.io/Ferrite.jl/stable/examples/plasticity/)
# and modified to show how `FESolvers` can be used to solve this nonlinear problem with time dependent loading.
# 
#md # !!! note
#md #     This example is preliminary, and doesn't necessarily represent good coding practice. 
#md #     As an example of a more general implementation, please see 
#md #     [`FerriteProblems.jl`](https://github.com/KnutAM/FerriteProblems.jl) and its 
#md #     [example](https://knutam.github.io/FerriteProblems.jl/dev/examples/plasticity/)
# 
# First we need to load all required packages
using FESolvers, Ferrite, Tensors, SparseArrays, LinearAlgebra, Plots

# We first include some basic definitions taken and modified from the original 
# [example](https://ferrite-fem.github.io/Ferrite.jl/stable/examples/plasticity/),
# specifically the material definitions: `J2Plasticity` and `J2PlasticityMaterialState`, 
# as well as the `doassemble!` function. The exact definitions are available here:
# [plasticity_definitions.jl](plasticity_definitions.jl),
include("plasticity_definitions.jl");

# ## Problem definition
# We divide the problem struct into three parts: definitions (`def`), a buffer (`buf`), and 
# postprocessing (`post`) to structure the information and make it easier to save the simulation 
# settings (enough to save `def` as the others will be created based on this one)
struct PlasticityProblem{PD,PB,PP}
    def::PD 
    buf::PB
    post::PP
end

# `PlasticityModel` is our `def` and contain all problem settings (mesh, material, loads, interpolations, etc.)
struct PlasticityModel{DH,CH,IP,M}
    dh::DH
    ch::CH
    interpolation::IP
    material::M
    traction_rate::Float64
end

function PlasticityModel()
    ## Material
    E = 200.0e9; ν = 0.3    # Young's modulus and Poisson's ratio
    σ₀ = 200e6; H = E/20    # Yield limit and hardening modulus
    material = J2Plasticity(E, ν, σ₀, H)

    ## Geometry (length, width, height)
    L = 10.0; w = 1.0; h = 1.0

    ## Loading
    traction_rate = 1.e7    # N/s

    ## Grid (beam)
    nels = (20, 2, 4)
    grid = generate_grid(Tetrahedron, nels, zero(Vec{3}), Vec((L, w, h)))

    ## Interpolation and DofHandler
    ip = Lagrange{3, RefTetrahedron, 1}()
    dh = DofHandler(grid)
    add!(dh, :u, 3, ip)
    close!(dh)
    
    ## ConstraintHandler
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfaceset(grid, "left"), Returns(zeros(3))))
    close!(ch)
    return PlasticityModel(dh, ch, ip, material, traction_rate)
end;

# `PlasticityFEBuffer` is our `buf` and contains all problem arrays and other allocated values 
struct PlasticityFEBuffer{CV,FV,KT,T,ST}
    cv::CV          # CellValues
    fv::FV          # FaceValues
    K::KT           # Stiffness matrix
    r::Vector{T}    # Residual vector
    u::Vector{T}    # Unknown vector
    states::Matrix{ST}
    states_old::Matrix{ST}
    time::Vector{T}     # Just a vector to allow mutating the time 
    old_time::Vector{T} # Same as time (not needed, but shown for completeness)
end

function build_febuffer(model::PlasticityModel)
    dh = model.dh
    n_dofs = ndofs(dh)
    qr      = QuadratureRule{3,RefTetrahedron}(2)
    face_qr = QuadratureRule{2,RefTetrahedron}(3)
    ip_geo = Ferrite.default_interpolation(Ferrite.getcelltype(dh.grid))
    cv = CellVectorValues(qr, model.interpolation, ip_geo)
    fv = FaceVectorValues(face_qr, model.interpolation, ip_geo)
    u  = zeros(n_dofs)
    r = zeros(n_dofs)
    K = create_sparsity_pattern(dh)
    nqp = getnquadpoints(cv)
    states = [J2PlasticityMaterialState() for _ in 1:nqp, _ in 1:getncells(model.dh.grid)]
    states_old = [J2PlasticityMaterialState() for _ in 1:nqp, _ in 1:getncells(model.dh.grid)]
    return PlasticityFEBuffer(cv,fv,K,r,u,states,states_old,[0.0], [0.0])
end;

# Finally, we define our `post` that contains variables that we will save during the simulation
struct PlasticityPost{T}
    umax::Vector{T}
    tmag::Vector{T}
end
PlasticityPost() = PlasticityPost(Float64[],Float64[]);

# To facilitate reuse, we define a function that gives our full problem struct 
# based on the problem definition 
build_problem(def::PlasticityModel) = PlasticityProblem(def, build_febuffer(def), PlasticityPost());

# ## Neumann boundary conditions 
# We then define a separate function for the Neumann boundary conditions 
# (note that this difference to the original example is not required, 
# but only to separate the element assembly and external boundary conditions)
# This could also be further simplified by using 
# [FerriteNeumann.jl](https://github.com/KnutAM/FerriteNeumann.jl)
function apply_neumann!(model::PlasticityModel,buf::PlasticityFEBuffer)
    t = buf.time[1]
    nu = getnbasefunctions(buf.cv)
    re = zeros(nu)
    facevalues = buf.fv
    grid = model.dh.grid
    traction = Vec((0.0, 0.0, model.traction_rate*t))

    for (i, cell) in enumerate(CellIterator(model.dh))
        fill!(re, 0)
        eldofs = celldofs(cell)
        for face in 1:nfaces(cell)
            if (cellid(cell), face) ∈ getfaceset(grid, "right")
                reinit!(facevalues, cell, face)
                for q_point in 1:getnquadpoints(facevalues)
                    dΓ = getdetJdV(facevalues, q_point)
                    for i in 1:nu
                        δu = shape_value(facevalues, q_point, i)
                        re[i] -= (δu ⋅ traction) * dΓ
                    end
                end
            end
        end
        buf.r[eldofs] .+= re
    end   
end;

# ## Specialized functions for our problem
# We first define our "get"-functions to get the key arrays for our problem. 
# Note that these functions don't calculate or update anything, that updating 
# is taken care of by `update-update_to_next_step!` and `update_problem!` below.
FESolvers.getunknowns(p::PlasticityProblem) = p.buf.u;
FESolvers.getresidual(p::PlasticityProblem) = p.buf.r;
FESolvers.getjacobian(p::PlasticityProblem) = p.buf.K;

# We then define the function to update the problem to a different time. This 
# is typically used to set time dependent boundary conditions. Here, it is also
# possible to make an improved guess for the solution to this time step if desired. 
function FESolvers.update_to_next_step!(p::PlasticityProblem, time)
    p.buf.time .= time
    update!(p.def.ch, time) # No influence in this particular example
end;

# Next, we define the updating of the problem given a new guess to the solution. 
# Note that we use `Δu::Nothing` for the case it is not given, to signal no change.
# This version is called directly after update_to_next_step! before entering 
# the nonlinear iterations. 
function FESolvers.update_problem!(p::PlasticityProblem, Δu, _)
    buf = p.buf 
    def = p.def
    if !isnothing(Δu)
        apply_zero!(Δu, p.def.ch)
        buf.u .+= Δu
    end
    doassemble!(buf.cv, buf.fv, buf.K, buf.r,
                def.dh.grid, def.dh, def.material, buf.u, buf.states, buf.states_old)
    apply_neumann!(def,buf)
    apply_zero!(buf.K, buf.r, def.ch)
end;

# In this example, we use the standard convergence criterion that the norm of the free 
# degrees of freedom is less than the iteration tolerance. 
FESolvers.calculate_convergence_measure(p::PlasticityProblem, args...) = norm(FESolvers.getresidual(p)[free_dofs(p.def.ch)]);

# As postprocessing, which is called after we detect that the solution has converged, 
# we save the maximum displacement as well as the traction magnitude.
function FESolvers.postprocess!(p::PlasticityProblem, step, solver)
    push!(p.post.umax, maximum(abs, FESolvers.getunknowns(p)))
    push!(p.post.tmag, p.def.traction_rate*p.buf.time[1])
end;

# After convergence (and postprocessing of that step),
# we also need to update the state variables and the time
function FESolvers.handle_converged!(p::PlasticityProblem)
    p.buf.states_old .= p.buf.states
    p.buf.old_time .= p.buf.time
end;

# ## Solving the problem
# First, we define a helper function to plot the results after the solution
function plot_results(problem; plt=plot(), label, markershape, markersize=4)
    plot!(plt, problem.post.umax, problem.post.tmag, 
        linewidth=0.5, title="Traction-displacement", label=label, 
        markeralpha=0.75, markershape=markershape, markersize=markersize)
    ylabel!(plt, "Traction [Pa]")
    xlabel!(plt, "Maximum deflection [m]")
    return plt
end;

# Finally, we can solve the problem with different time stepping strategies and plot the results
function example_solution()
    def = PlasticityModel()

    ## Fixed uniform time steps
    problem = build_problem(def)
    solver = QuasiStaticSolver(NewtonSolver(;tolerance=1.0), FixedTimeStepper(;num_steps=25,Δt=0.04))
    solve_problem!(problem, solver)
    plt = plot_results(problem, label="uniform", markershape=:x, markersize=5)

    ## Same time steps as Ferrite example
    problem = build_problem(def)
    solver = QuasiStaticSolver(NewtonSolver(;tolerance=1.0), FixedTimeStepper(append!([0.], collect(0.5:0.05:1.0))))
    solve_problem!(problem, solver)
    plot_results(problem, plt=plt, label="fixed", markershape=:circle)

    ## Adaptive time stepping 
    problem = build_problem(def)
    ts = AdaptiveTimeStepper(0.05, 1.0; Δt_min=0.01, Δt_max=0.2)
    solver = QuasiStaticSolver(NewtonSolver(;tolerance=1.0, maxiter=6), ts)
    solve_problem!(problem, solver)
    println(problem.buf.time)
    plot_results(problem, plt=plt, label="adaptive", markershape=:circle)
    plot!(;legend=:bottomright)
end;

example_solution()

#md # ## Plain program
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`plasticity.jl`](plasticity.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```