using SparseArrays, LinearAlgebra
@testset "linearsolvers" begin
    for K in (rand(10,10)+10I, sprand(10,10,0.2)+10I)
        for linsolver in (BackslashSolver(), )
            @testset "$(typeof(K)), $(typeof(linsolver))" begin
                r =  rand(size(K,1))
                Δx = similar(r)
                FerriteSolvers.solve_linear!(Δx, K, r, linsolver)
                @test K*Δx ≈ -r 
            end
        end
    end
end