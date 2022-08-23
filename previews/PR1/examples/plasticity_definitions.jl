struct J2Plasticity{T, S <: SymmetricTensor{4, 3, T}}
    G::T  # Shear modulus
    K::T  # Bulk modulus
    σ₀::T # Initial yield limit
    H::T  # Hardening modulus
    Dᵉ::S # Elastic stiffness tensor
end;

function J2Plasticity(E, ν, σ₀, H)
    δ(i,j) = i == j ? 1.0 : 0.0 # helper function
    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)

    Isymdev(i,j,k,l) = 0.5*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k)) - 1.0/3.0*δ(i,j)*δ(k,l)
    temp(i,j,k,l) = 2.0G *( 0.5*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k)) + ν/(1.0-2.0ν)*δ(i,j)*δ(k,l))
    Dᵉ = SymmetricTensor{4, 3}(temp)
    return J2Plasticity(G, K, σ₀, H, Dᵉ)
end;

struct J2PlasticityMaterialState{T, S <: SecondOrderTensor{3, T}}
    # Store "converged" values
    ϵᵖ::S # plastic strain
    σ::S # stress
    k::T # hardening variable
end

function J2PlasticityMaterialState()
    return J2PlasticityMaterialState(
                zero(SymmetricTensor{2, 3}),
                zero(SymmetricTensor{2, 3}),
                0.0)
end

function vonMises(σ)
    s = dev(σ)
    return sqrt(3.0/2.0 * s ⊡ s)
end;

function compute_stress_tangent(ϵ::SymmetricTensor{2, 3}, material::J2Plasticity, state::J2PlasticityMaterialState)
    # unpack some material parameters
    G = material.G
    H = material.H

    # We use (•)ᵗ to denote *trial*-values
    σᵗ = material.Dᵉ ⊡ (ϵ - state.ϵᵖ) # trial-stress
    sᵗ = dev(σᵗ)         # deviatoric part of trial-stress
    J₂ = 0.5 * sᵗ ⊡ sᵗ   # second invariant of sᵗ
    σᵗₑ = sqrt(3.0*J₂)   # effective trial-stress (von Mises stress)
    σʸ = material.σ₀ + H * state.k # Previous yield limit

    φᵗ  = σᵗₑ - σʸ # Trial-value of the yield surface

    if φᵗ < 0.0 # elastic loading
        return σᵗ, material.Dᵉ, J2PlasticityMaterialState(state.ϵᵖ, σᵗ, state.k)
    else # plastic loading
        h = H + 3G
        μ =  φᵗ / h   # plastic multiplier

        c1 = 1 - 3G * μ / σᵗₑ
        s = c1 * sᵗ           # updated deviatoric stress
        σ = s + vol(σᵗ)       # updated stress

        # Compute algorithmic tangent stiffness ``D = \frac{\Delta \sigma }{\Delta \epsilon}``
        κ = H * (state.k + μ) # drag stress
        σₑ = material.σ₀ + κ  # updated yield surface

        δ(i,j) = i == j ? 1.0 : 0.0
        Isymdev(i,j,k,l)  = 0.5*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k)) - 1.0/3.0*δ(i,j)*δ(k,l)
        Q(i,j,k,l) = Isymdev(i,j,k,l) - 3.0 / (2.0*σₑ^2) * s[i,j]*s[k,l]
        b = (3G*μ/σₑ) / (1.0 + 3G*μ/σₑ)

        Dtemp(i,j,k,l) = -2G*b * Q(i,j,k,l) - 9G^2 / (h*σₑ^2) * s[i,j]*s[k,l]
        D = material.Dᵉ + SymmetricTensor{4, 3}(Dtemp)

        # Return new state
        Δϵᵖ = 3/2 * μ / σₑ * s # plastic strain
        ϵᵖ = state.ϵᵖ + Δϵᵖ    # plastic strain
        k = state.k + μ        # hardening variable
        return σ, D, J2PlasticityMaterialState(ϵᵖ, σ, k)
    end
end

function create_values(interpolation)
    # setup quadrature rules
    qr      = QuadratureRule{3,RefTetrahedron}(2)
    face_qr = QuadratureRule{2,RefTetrahedron}(3)

    # create geometric interpolation (use the same as for u)
    interpolation_geom = Lagrange{3,RefTetrahedron,1}()

    # cell and facevalues for u
    cellvalues_u = CellVectorValues(qr, interpolation, interpolation_geom)
    facevalues_u = FaceVectorValues(face_qr, interpolation, interpolation_geom)

    return cellvalues_u, facevalues_u
end;

function create_dofhandler(grid, interpolation)
    dh = DofHandler(grid)
    dim = 3
    push!(dh, :u, dim, interpolation) # add a displacement field with 3 components
    close!(dh)
    return dh
end

function create_bc(dh, grid)
    dbcs = ConstraintHandler(dh)
    # Clamped on the left side
    dofs = [1, 2, 3]
    dbc = Dirichlet(:u, getfaceset(grid, "left"), (x,t) -> [0.0, 0.0, 0.0], dofs)
    add!(dbcs, dbc)
    close!(dbcs)
    return dbcs
end;

function doassemble!(cellvalues::CellVectorValues{dim},
                    facevalues::FaceVectorValues{dim}, K::SparseMatrixCSC, r::Vector, grid::Grid,
                    dh::DofHandler, material::J2Plasticity, u, states, states_old) where {dim}
    assembler = start_assemble(K, r)
    nu = getnbasefunctions(cellvalues)
    re = zeros(nu)     # element residual vector
    ke = zeros(nu, nu) # element tangent matrix

    for (i, cell) in enumerate(CellIterator(dh))
        fill!(ke, 0)
        fill!(re, 0)
        eldofs = celldofs(cell)
        ue = u[eldofs]
        state = @view states[:, i]
        state_old = @view states_old[:, i]
        assemble_cell!(ke, re, cell, cellvalues, facevalues, grid, material,
                       ue, state, state_old)
        assemble!(assembler, eldofs, re, ke)
    end
    return K, r
end

function assemble_cell!(Ke, re, cell, cellvalues, facevalues, grid, material,
                        ue, state, state_old)
    n_basefuncs = getnbasefunctions(cellvalues)
    reinit!(cellvalues, cell)

    for q_point in 1:getnquadpoints(cellvalues)
        # For each integration point, compute stress and material stiffness
        ϵ = function_symmetric_gradient(cellvalues, q_point, ue) # Total strain
        σ, D, state[q_point] = compute_stress_tangent(ϵ, material, state_old[q_point])

        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
            δϵ = shape_symmetric_gradient(cellvalues, q_point, i)
            re[i] += (δϵ ⊡ σ) * dΩ # add internal force to residual
            for j in 1:i # loop only over lower half
                Δϵ = shape_symmetric_gradient(cellvalues, q_point, j)
                Ke[i, j] += δϵ ⊡ D ⊡ Δϵ * dΩ
            end
        end
    end
    symmetrize_lower!(Ke)
end

function symmetrize_lower!(K)
    for i in 1:size(K,1)
        for j in i+1:size(K,1)
            K[i,j] = K[j,i]
        end
    end
end