using SparseArrays, LinearAlgebra
@testset "linearsolvers" begin
    using LinearSolve
    for K in (rand(10,10)+10I, sprand(10,10,0.2)+10I)
        r0 =  rand(size(K,1))
        ls_default = LinearSolveSolver(K)
        ls_lu = LinearSolveSolver(isa(K, Matrix) ?  SimpleLUFactorization() : KLUFactorization(), K, r0)
        for linsolver in (BackslashSolver(), ls_default, ls_lu)
            #show(stdout, MIME"text/plain"(), linsolver)
            @testset "$(typeof(K)), $(sprint(show, MIME"text/plain"(), linsolver))" begin
                r = copy(r0)
                Δx = similar(r)
                FESolvers.solve_linear!(Δx, copy(K), r, linsolver)
                @test K*Δx ≈ -r 
            end
        end
    end
end