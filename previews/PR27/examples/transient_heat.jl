using Ferrite, SparseArrays, FESolvers

struct TransientHeat{DEF,BUF,POST}
    def::DEF    # Problem definition
    buf::BUF    # Buffers for storing values
    post::POST  # Struct to save simulation data in each step
end

struct ProblemDefinition{DH,CH,CV}
    dh::DH
    ch::CH
    cv::CV
end

function ProblemDefinition()
    # **Grid**
    grid = generate_grid(Quadrilateral, (100, 100));

    # **Cell values**
    dim = 2
    ip = Lagrange{dim, RefCube, 1}()
    qr = QuadratureRule{dim, RefCube}(2)
    cellvalues = CellScalarValues(qr, ip);

    # **Degrees of freedom**
    # After this, we can define the `DofHandler` and distribute the DOFs of the problem.
    dh = DofHandler(grid)
    push!(dh, :u, 1)
    close!(dh);

    # **Boundary conditions**
    # In order to define the time dependent problem, we need some end time `T` and something that describes
    # the linearly increasing Dirichlet boundary condition on $\partial \Omega_2$.
    max_temp = 100
    t_rise = 100
    ch = ConstraintHandler(dh);

    # Here, we define the boundary condition related to $\partial \Omega_1$.
    ∂Ω₁ = union(getfaceset.((grid,), ["left", "right"])...)
    dbc = Dirichlet(:u, ∂Ω₁, (x, t) -> 0)
    add!(ch, dbc);
    # While the next code block corresponds to the linearly increasing temperature description on $\partial \Omega_2$
    # until `t=t_rise`, and then keep constant
    ∂Ω₂ = union(getfaceset.((grid,), ["top", "bottom"])...)
    dbc = Dirichlet(:u, ∂Ω₂, (x, t) -> max_temp * clamp(t / t_rise, 0, 1))
    add!(ch, dbc)
    close!(ch)
    return ProblemDefinition(dh, ch, cellvalues)
end;

struct ProblemBuffer{KT,T}
    K::KT
    r::Vector{T}
    u::Vector{T}
    uold::Vector{T}
    times::Vector{T}    # [t_old, t_current]
end
function ProblemBuffer(def::ProblemDefinition)
    dh = def.dh
    K = create_sparsity_pattern(dh)
    r = zeros(ndofs(dh))
    u = zeros(ndofs(dh))
    uold = zeros(ndofs(dh))
    times = zeros(2)
    return ProblemBuffer(K, r, u, uold, times)
end;

function doassemble!(K::SparseMatrixCSC, r::Vector, cellvalues::CellScalarValues, dh::DofHandler, u, uold, Δt)
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    re = zeros(n_basefuncs)
    ue = zeros(n_basefuncs)
    ue_old = zeros(n_basefuncs)
    assembler = start_assemble(K, r)
    for cell in CellIterator(dh)
        fill!(Ke, 0)
        fill!(re, 0)
        ue .= u[celldofs(cell)]
        ue_old .= uold[celldofs(cell)]
        reinit!(cellvalues, cell)
        element_routine!(Ke, re, cellvalues, ue, ue_old, Δt)
        assemble!(assembler, celldofs(cell), re, Ke)
    end
end

function element_routine!(Ke, re, cellvalues, ue, ue_old, Δt, k=1.0e-3, f=0.5)
    n_basefuncs = getnbasefunctions(cellvalues)
    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        u = function_value(cellvalues, q_point, ue)
        uold = function_value(cellvalues, q_point, ue_old)
        ∇u = function_gradient(cellvalues, q_point, ue)
        for i in 1:n_basefuncs
            δN = shape_value(cellvalues, q_point, i)
            ∇δN = shape_gradient(cellvalues, q_point, i)
            re[i] += (δN * (u - uold - Δt * f) + Δt * k * ∇δN ⋅ ∇u) * dΩ
            for j in 1:n_basefuncs
                N = shape_value(cellvalues, q_point, j)
                ∇N = shape_gradient(cellvalues, q_point, j)
                Ke[i, j] += (δN*N + Δt * k * (∇δN ⋅ ∇N)) * dΩ
            end
        end
    end
end;

FESolvers.getunknowns(p::TransientHeat) = p.buf.u
FESolvers.getresidual(p::TransientHeat) = p.buf.r
FESolvers.getjacobian(p::TransientHeat) = p.buf.K

function FESolvers.update_to_next_step!(p::TransientHeat, time)
    p.buf.times[2] = time       # Update current time
    update!(p.def.ch, time)     # Update Dirichlet boundary conditions
    apply!(FESolvers.getunknowns(p), p.def.ch)
end

function FESolvers.update_problem!(p::TransientHeat, Δu, update_spec)
    if !isnothing(Δu)
        apply_zero!(Δu, p.def.ch)
        p.buf.u .+= Δu
    end
    # Since the problem is linear, we can save some computations by only updating once per time step
    # and not after updating the temperatures to check that it has converged.
    if FESolvers.should_update_jacobian(update_spec) || FESolvers.should_update_residual(update_spec)
        Δt = p.buf.times[2]-p.buf.times[1]
        doassemble!(p.buf.K, p.buf.r, p.def.cv, p.def.dh, FESolvers.getunknowns(p), p.buf.uold, Δt)
        apply_zero!(FESolvers.getjacobian(p), FESolvers.getresidual(p), p.def.ch)
    end
    return nothing
end

function FESolvers.handle_converged!(p::TransientHeat)
    copy!(p.buf.uold, FESolvers.getunknowns(p)) # Set old temperature to current
    p.buf.times[1] = p.buf.times[2]             # Set old time to current
end;

struct PostProcessing{PVD}
    pvd::PVD
end
PostProcessing() = PostProcessing(paraview_collection("transient-heat.pvd"));

function FESolvers.postprocess!(p::TransientHeat, step, solver)
    vtk_grid("transient-heat-$step", p.def.dh) do vtk
        vtk_point_data(vtk, p.def.dh, p.buf.u)
        vtk_save(vtk)
        p.post.pvd[step] = vtk
    end
end;

function FESolvers.close_problem(p::TransientHeat)
    vtk_save(p.post.pvd)
end;

TransientHeat(def) = TransientHeat(def, ProblemBuffer(def), PostProcessing());

problem = TransientHeat(ProblemDefinition())
solver = QuasiStaticSolver(;nlsolver=LinearProblemSolver(), timestepper=FixedTimeStepper(collect(0.0:1.0:200)));

solve_problem!(problem, solver);

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

