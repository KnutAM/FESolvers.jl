using FESolvers, Ferrite, Tensors, SparseArrays, LinearAlgebra, Plots

include("plasticity_definitions.jl");

struct PlasticityProblem{PD,PB,PP}
    def::PD
    buf::PB
    post::PP
end

struct PlasticityModel{DH,CH,IP,M}
    dh::DH
    ch::CH
    interpolation::IP
    material::M
    traction_rate::Float64
end

function PlasticityModel()
    # Material
    E = 200.0e9; ν = 0.3    # Young's modulus and Poisson's ratio
    σ₀ = 200e6; H = E/20    # Yield limit and hardening modulus
    material = J2Plasticity(E, ν, σ₀, H)

    # Geometry (length, width, height)
    L = 10.0; w = 1.0; h = 1.0

    # Loading
    traction_rate = 1.e7    # N/s

    # Grid (beam)
    nels = (20, 2, 4)
    grid = generate_grid(Tetrahedron, nels, zero(Vec{3}), Vec((L, w, h)))

    # Interpolation and DofHandler
    ip = Lagrange{3, RefTetrahedron, 1}()
    dh = DofHandler(grid)
    add!(dh, :u, 3, ip)
    close!(dh)

    # ConstraintHandler
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfaceset(grid, "left"), Returns(zeros(3))))
    close!(ch)
    return PlasticityModel(dh, ch, ip, material, traction_rate)
end;

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

struct PlasticityPost{T}
    umax::Vector{T}
    tmag::Vector{T}
end
PlasticityPost() = PlasticityPost(Float64[],Float64[]);

build_problem(def::PlasticityModel) = PlasticityProblem(def, build_febuffer(def), PlasticityPost());

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

FESolvers.getunknowns(p::PlasticityProblem) = p.buf.u;
FESolvers.getresidual(p::PlasticityProblem) = p.buf.r;
FESolvers.getjacobian(p::PlasticityProblem) = p.buf.K;

function FESolvers.update_to_next_step!(p::PlasticityProblem, time)
    p.buf.time .= time
    update!(p.def.ch, time) # No influence in this particular example
end;

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

FESolvers.calculate_convergence_measure(p::PlasticityProblem, args...) = norm(FESolvers.getresidual(p)[free_dofs(p.def.ch)]);

function FESolvers.postprocess!(p::PlasticityProblem, step, solver)
    push!(p.post.umax, maximum(abs, FESolvers.getunknowns(p)))
    push!(p.post.tmag, p.def.traction_rate*p.buf.time[1])
end;

function FESolvers.handle_converged!(p::PlasticityProblem)
    p.buf.states_old .= p.buf.states
    p.buf.old_time .= p.buf.time
end;

function plot_results(problem; plt=plot(), label, markershape, markersize=4)
    plot!(plt, problem.post.umax, problem.post.tmag,
        linewidth=0.5, title="Traction-displacement", label=label,
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

