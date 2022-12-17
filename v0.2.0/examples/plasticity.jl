using FESolvers, Ferrite, Tensors, SparseArrays, LinearAlgebra, Plots

include("plasticity_definitions.jl");

struct PlasticityProblem{PD,PB,PP}
    def::PD
    buf::PB
    post::PP
end

struct PlasticityModel
    grid
    interpolation
    dh
    material
    traction_rate
    dbcs
end

function PlasticityModel()
    E = 200.0e9
    H = E/20
    ν = 0.3
    σ₀ = 200e6
    material = J2Plasticity(E, ν, σ₀, H)

    L = 10.0
    w = 1.0
    h = 1.0

    traction_rate = 1.e7    # N/s

    n = 2
    nels = (10n, n, 2n)
    P1 = Vec((0.0, 0.0, 0.0))
    P2 = Vec((L, w, h))
    grid = generate_grid(Tetrahedron, nels, P1, P2)
    interpolation = Lagrange{3, RefTetrahedron, 1}()
    dh = create_dofhandler(grid, interpolation)
    dbcs = create_bc(dh, grid)
    return PlasticityModel(grid,interpolation,dh,material,traction_rate,dbcs)
end;

struct PlasticityFEBuffer
    cellvalues
    facevalues
    K
    r
    u
    states
    states_old
    time::Vector # length(time)=1, could use ScalarBuffer instead
end

function build_febuffer(model::PlasticityModel)
    dh = model.dh
    n_dofs = ndofs(dh)
    cellvalues, facevalues = create_values(model.interpolation)
    u  = zeros(n_dofs)
    r = zeros(n_dofs)
    K = create_sparsity_pattern(dh)
    nqp = getnquadpoints(cellvalues)
    states = [J2PlasticityMaterialState() for _ in 1:nqp, _ in 1:getncells(model.grid)]
    states_old = [J2PlasticityMaterialState() for _ in 1:nqp, _ in 1:getncells(model.grid)]
    return PlasticityFEBuffer(cellvalues,facevalues,K,r,u,states,states_old,[0.0])
end;

struct PlasticityPost{T}
    umax::Vector{T}
    tmag::Vector{T}
end
PlasticityPost() = PlasticityPost(Float64[],Float64[]);

build_problem(def::PlasticityModel) = PlasticityProblem(def, build_febuffer(def), PlasticityPost());

function apply_neumann!(model::PlasticityModel,buffer::PlasticityFEBuffer)
    t = buffer.time[1]
    nu = getnbasefunctions(buffer.cellvalues)
    re = zeros(nu)
    facevalues = buffer.facevalues
    grid = model.grid
    traction = Vec((0.0, 0.0, model.traction_rate*t))

    for (i, cell) in enumerate(CellIterator(model.dh))
        fill!(re, 0)
        eldofs = celldofs(cell)
        for face in 1:nfaces(cell)
            if onboundary(cell, face) && (cellid(cell), face) ∈ getfaceset(grid, "right")
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
        buffer.r[eldofs] .+= re
    end
end;

FESolvers.getunknowns(p::PlasticityProblem) = p.buf.u;
FESolvers.getresidual(p::PlasticityProblem) = p.buf.r;
FESolvers.getjacobian(p::PlasticityProblem) = p.buf.K;

function FESolvers.update_to_next_step!(p::PlasticityProblem, time)
    p.buf.time .= time
end;

function FESolvers.update_problem!(p::PlasticityProblem, Δu; kwargs...)
    buf = p.buf
    def = p.def
    if !isnothing(Δu)
        apply_zero!(Δu, p.def.dbcs)
        buf.u .+= Δu
    end
    doassemble!(buf.cellvalues, buf.facevalues, buf.K, buf.r,
                def.grid, def.dh, def.material, buf.u, buf.states, buf.states_old)
    apply_neumann!(def,buf)
    apply_zero!(buf.K, buf.r, def.dbcs)
end;

FESolvers.calculate_convergence_measure(p::PlasticityProblem, args...) = norm(FESolvers.getresidual(p)[free_dofs(p.def.dbcs)]);

function FESolvers.handle_converged!(p::PlasticityProblem)
    p.buf.states_old .= p.buf.states
end;

function FESolvers.postprocess!(p::PlasticityProblem, step)
    push!(p.post.umax, maximum(abs, FESolvers.getunknowns(p)))
    push!(p.post.tmag, p.def.traction_rate*p.buf.time[1])
end;

function plot_results(problem::PlasticityProblem; plt=plot(), label=nothing, markershape=:auto, markersize=4)
    umax = vcat(0.0, problem.post.umax)
    tmag = vcat(0.0, problem.post.tmag)
    plot!(plt, umax, tmag, linewidth=0.5, title="Traction-displacement", label=label,
        markeralpha=0.75, markershape=markershape, markersize=markersize)
    ylabel!(plt, "Traction [Pa]")
    xlabel!(plt, "Maximum deflection [m]")
    return plt
end;

function example_solution()
    def = PlasticityModel()

    # Fixed uniform time steps
    problem = build_problem(def)
    solver = QuasiStaticSolver(NewtonSolver(;tolerance=1.0), FixedTimeStepper(;num_steps=25,Δt=0.04))
    solve_problem!(problem, solver)
    plt = plot_results(problem, label="uniform", markershape=:x, markersize=5)

    # Same time steps as Ferrite example
    problem = build_problem(def)
    solver = QuasiStaticSolver(NewtonSolver(;tolerance=1.0), FixedTimeStepper(append!([0.], collect(0.5:0.05:1.0))))
    solve_problem!(problem, solver)
    plot_results(problem, plt=plt, label="fixed", markershape=:circle)

    # Adaptive time stepping
    problem = build_problem(def)
    ts = AdaptiveTimeStepper(0.05, 1.0; Δt_min=0.01, Δt_max=0.2)
    solver = QuasiStaticSolver(NewtonSolver(;tolerance=1.0, maxiter=6), ts)
    solve_problem!(problem, solver)
    println(problem.buf.time)
    plot_results(problem, plt=plt, label="adaptive", markershape=:circle)
    plot!(;legend=:bottomright)
end;

example_solution()

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

