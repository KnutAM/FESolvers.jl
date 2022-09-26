# ScalarWrapper copied from Ferrite.jl
# For Julia>=1.8, this can be replaced by using const annotations 
# in combination with mutable structs. 
mutable struct ScalarWrapper{T}
    x::T
end

@inline Base.getindex(s::ScalarWrapper) = s.x
@inline Base.setindex!(s::ScalarWrapper, v) = s.x = v
Base.copy(s::ScalarWrapper{T}) where {T} = ScalarWrapper{T}(copy(s.x))