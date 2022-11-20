using SparseArrays, LinearAlgebra
@testset "linearsolvers" begin
    using LinearSolve
    for K in (rand(10,10)+10I, sprand(10,10,0.2)+10I)
        for linsolver in (BackslashSolver(), nothing, (isa(K,Matrix) ? SimpleLUFactorization() : KLUFactorization()))
            @testset "$(typeof(K)), $(typeof(linsolver))" begin
                r =  rand(size(K,1))
                Δx = similar(r)
                FESolvers.solve_linear!(Δx, K, r, linsolver)
                @test K*Δx ≈ -r 
            end
        end
    end
end