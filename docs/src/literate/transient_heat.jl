# # Linear Time Dependent Problem
# This example is taken from 
# [`Ferrite.jl`'s transient heat flow](https://ferrite-fem.github.io/Ferrite.jl/stable/examples/transient_heat_equation/).
# We modify the material parameters to get more time-dependent behavior. 
# 
# Currently, only Quasi-static problems are supported by FESolvers. Therefore, we reformulate the linear system compared 
# to remove the mass matrices. We have the same time-discretized weak form:
# ```math
# \int_{\Omega} v\, u_{n+1}\ \mathrm{d}\Omega + \Delta t\int_{\Omega} k \nabla v \cdot \nabla u_{n+1} \ \mathrm{d}\Omega = \Delta t\int_{\Omega} v f \ \mathrm{d}\Omega + \int_{\Omega} v \, u_{n} \ \mathrm{d}\Omega.
# ```
# We then define the linear residual, ``r(u_{n+1})``, as 
# ```math
# r(u_{n+1}) = f_\mathrm{int}(u_{n+1}) - f_\mathrm{ext}(u_{n}) \\
# f_\mathrm{int}(u_{n+1}) = \int_{\Omega} v\, u_{n+1}\ \mathrm{d}\Omega + \Delta t\int_{\Omega} k \nabla v \cdot \nabla u_{n+1} \ \mathrm{d}\Omega  \\
# f_\mathrm{ext}(u_{n}) = \Delta t\int_{\Omega} v f \ \mathrm{d}\Omega + \int_{\Omega} v \, u_{n} \ \mathrm{d}\Omega. \\
# ```
# giving the discrete operators
# ```math
# r_i(\mathbf{u}_{n+1}) = K_{ij} [\mathbf{u}_{n+1}]_j - [\mathbf{f}_\mathrm{ext}(u_{n})]_i
# ```
# upon introduction of the function approximation, ``u(\mathbf{x}) \approx N_i(\mathbf{x}) u_i``, and the test approximation, 
# ``v(\mathbf{x}) \approx \delta N_i(\mathbf{x}) v_i``
# where 
# ```math
# K_{ij} = \int_{\Omega} \delta N_i(\mathbf{x})\, N_j(\mathbf{x}) \ \mathrm{d}\Omega + \Delta t\int_{\Omega} k \nabla \delta N_i(\mathbf{x}) \cdot \nabla N_j(\mathbf{x}) \ \mathrm{d}\Omega  \\
# \left[\mathbf{f}_\mathrm{ext}(u_{n})\right]_i = \Delta t\int_{\Omega} \delta N_i(\mathbf{x}) f \ \mathrm{d}\Omega + \int_{\Omega} \delta N_i(\mathbf{x}) \, u_{n}(\mathbf{x}) \ \mathrm{d}\Omega.
# ```
# and the residual expression can be simplified to 
# ```math
# r_i = 
# \int_{\Omega} \delta N_i(\mathbf{x})\, \left[u(\mathbf{x})-u_{n}(\mathbf{x})\right] \ \mathrm{d}\Omega 
# + \Delta t\int_{\Omega} k \nabla \delta N_i(\mathbf{x}) \cdot \nabla u(\mathbf{x}) \ \mathrm{d}\Omega 
# - \Delta t\int_{\Omega} \delta N_i(\mathbf{x}) f \ \mathrm{d}\Omega
# ```
# 
# ## Commented Program
#
# Now we solve the problem by using Ferrite and FESolvers. 
#md # The full program, without comments, can be found in the next [section](@ref transient_heat_equation-plain-program).
#
# First we load Ferrite, and some other packages we need.
using Ferrite, SparseArrays, FESolvers
# 
# Then, we define our problem structs. At the end, we will define a nice constructor for this. 
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
    ## **Grid**
    grid = generate_grid(Quadrilateral, (100, 100));

    ## **Cell values**
    dim = 2
    ip = Lagrange{dim, RefCube, 1}()
    qr = QuadratureRule{dim, RefCube}(2)
    cellvalues = CellScalarValues(qr, ip);

    ## **Degrees of freedom**
    ## After this, we can define the `DofHandler` and distribute the DOFs of the problem.
    dh = DofHandler(grid)
    push!(dh, :u, 1)
    close!(dh);

    ## **Boundary conditions**
    ## In order to define the time dependent problem, we need some end time `T` and something that describes
    ## the linearly increasing Dirichlet boundary condition on $\partial \Omega_2$.
    max_temp = 100
    t_rise = 100
    ch = ConstraintHandler(dh);

    ## Here, we define the boundary condition related to $\partial \Omega_1$.
    ∂Ω₁ = union(getfaceset.((grid,), ["left", "right"])...)
    dbc = Dirichlet(:u, ∂Ω₁, (x, t) -> 0)
    add!(ch, dbc);
    ## While the next code block corresponds to the linearly increasing temperature description on $\partial \Omega_2$
    ## until `t=t_rise`, and then keep constant
    ∂Ω₂ = union(getfaceset.((grid,), ["top", "bottom"])...)
    dbc = Dirichlet(:u, ∂Ω₂, (x, t) -> max_temp * clamp(t / t_rise, 0, 1))
    add!(ch, dbc)
    close!(ch)
    return ProblemDefinition(dh, ch, cellvalues)
end;

# We then define a problem buffer, that can be created based on the `ProblemDefinition`
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

# We also need functions to assemble the stiffness and residual vectors 
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

# We now define all the required methods for solving this system with using the `LinearProblemSolver`
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
    ## Since the problem is linear, we can save some computations by only updating once per time step 
    ## and not after updating the temperatures to check that it has converged. 
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

# We are now ready to solve the system, but to save some data we must define some postprocessing tasks
# In this example, we only save things to file 
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

# At the end of the simulation, we want to finish all IO operations. 
# We can then define the function `close_problem` which will be called 
# even in the case that an error is thrown during the simulation
function FESolvers.close_problem(p::TransientHeat)
    vtk_save(p.post.pvd)
end;

# We then define a nice constructor for `TransientHeat` and can solve the problem,
TransientHeat(def) = TransientHeat(def, ProblemBuffer(def), PostProcessing());

# And now we create the problem type, and define the QuasiStaticSolver with 
# the LinearProblemSolver as well as fixed time steps 
problem = TransientHeat(ProblemDefinition())
solver = QuasiStaticSolver(;nlsolver=LinearProblemSolver(), timestepper=FixedTimeStepper(collect(0.0:1.0:200)));

# Finally, we can solve the problem
solve_problem!(problem, solver);

#md # ## [Plain program](@id transient_heat_equation-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here:
#md # [`transient_heat_equation.jl`](transient_heat_equation.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
