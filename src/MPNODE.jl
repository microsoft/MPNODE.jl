module MPNODE

using Random
using LinearAlgebra

using LearnBase
using MLDataPattern

using Flux
using Zygote
# using DiffEqFlux
using OrdinaryDiffEq
# using DiffEqSensitivity
using LightGraphs
using StaticArrays
using RobotDynamics: AbstractModel
import RobotDynamics: state_dim, control_dim, state, control, dynamics, discrete_dynamics

# Some standard DynamicalSystems
state(sys::AbstractModel, x::AbstractVector) = x[1:state_dim(sys)]
control(sys::AbstractModel, x::AbstractVector) = x[state_dim(sys)+1:state_dim(sys)+control_dim(sys)]
state(sys::AbstractModel, x::AbstractMatrix) = x[1:state_dim(sys), :]
control(sys::AbstractModel, x::AbstractMatrix) = x[state_dim(sys)+1:state_dim(sys)+control_dim(sys), :]

state(sys::AbstractModel, x::AbstractArray{T,3}) where T = x[1:state_dim(sys), :, :]
control(sys::AbstractModel, x::AbstractArray{T,3}) where T = x[state_dim(sys)+1:state_dim(sys)+control_dim(sys), :, :]

diffeqarray_to_array(x) = reshape(x, size(x)[1:end-2]..., prod(size(x)[end-1:end]))

function discrete_dynamics(sys::AbstractModel, x, u, dt::F, args...; kwargs...) where {F <: AbstractFloat}
    p_, re = Flux.destructure(sys)
    dzdt_(z,p,t) = dynamics(re(p), state(sys, z), control(sys, z))
    ff = ODEFunction{false}(dzdt_)
    prob = ODEProblem{false}(ff, cat(x, u; dims=1), (zero(dt), dt), p_)
    return diffeqarray_to_array(Array(solve(prob, args...; kwargs...)))
end

function rollout(sys::AbstractModel, x_0_B, us_Tm1_B, dt, alg, args...; kwargs...)
    T = size(us_Tm1_B)[end-1] # should maybe be just us_Tm1_B?
    B = size(x_0_B)[end]
    # Note: Things you have to do to avoid
    #   ERROR: Mutating arrays is not supported
    x_T_B = Zygote.Buffer(x_0_B, size(x_0_B)[1:end-1]..., T, B)
    x_T_B[axes(x_0_B)[1:end-1]..., 1, :] = x_0_B
    for i in 2:T
        x_T_B[axes(x_0_B)[1:end-1]..., i, :] = discrete_dynamics(sys, x_T_B[axes(x_0_B)[1:end-1]..., i-1, :], us_Tm1_B[:, i-1, :], dt, alg, args...; kwargs...)
    end
    return copy(x_T_B)
end

include("systems/datapath.jl")
include("systems/lorenzsystem.jl")
include("systems/pendulum.jl")
include("systems/gene_evolution.jl")
include("systems/kuramoto.jl")
include("datagen.jl")

include("datatransform.jl")

include("nde.jl")
include("gnnode.jl")
include("empde.jl")

include("learn_utils.jl")
include("learn.jl")

include("losses.jl")

end