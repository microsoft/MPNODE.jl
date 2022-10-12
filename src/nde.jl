using Flux
using DiffEqFlux
using DiffEqFlux: basic_tgrad, NeuralDELayer
using DiffEqSensitivity: ZygoteVJP, InterpolatingAdjoint
using OrdinaryDiffEq
using RobotDynamics: AbstractModel
import RobotDynamics: dynamics, discrete_dynamics



struct SimpleNeuralDELayer{M,P,RE} <: NeuralDELayer
    model::M
    p::P
    re::RE
    function SimpleNeuralDELayer(model; p=nothing)
        _p, re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        new{typeof(model),typeof(p),typeof(re)}(model, p, re)
    end
end
Flux.@functor SimpleNeuralDELayer

struct AugmentedNeuralDELayer{M,P,RE} <: NeuralDELayer
    model::M
    p::P
    re::RE
    augdim::Int
    function AugmentedNeuralDELayer(model, augdim; p=nothing)
        _p, re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        new{typeof(model),typeof(p),typeof(re)}(model, p, re, augdim)
    end
end
Flux.@functor AugmentedNeuralDELayer


function (n::SimpleNeuralDELayer)(x, p=n.p)
    n.model(x)
end

function (n::AugmentedNeuralDELayer)(x, p=n.p)
    n.model(DiffEqFlux.augment(x, n.augdim))
end


struct NeuralDynamicsModel{F1,F2,G<:NeuralDELayer,H} <: AbstractModel
    state_encoder::F1
    control_encoder::F2
    dynamics::G
    decoder::H
end

Flux.@functor NeuralDynamicsModel

dynamics(sys::NeuralDynamicsModel, x, u) = sys.dynamics.re(sys.dynamics.p)(cat(sys.state_encoder(x), sys.control_encoder(u); dims=1))

function dynamics(sys::NeuralDynamicsModel{F1,F2,G,H}, x, u) where {F1,F2,G<:AugmentedNeuralDELayer,H}
    inp = cat(sys.state_encoder(x), sys.control_encoder(u); dims=1)
    sys.dynamics.re(sys.dynamics.p)(DiffEqFlux.augment(inp, sys.dynamics.augdim))
end

function discrete_dynamics(n::NeuralDynamicsModel, x, u, dt::F, args...;sense=InterpolatingAdjoint(autojacvec=ZygoteVJP()), kwargs...) where {F <: AbstractFloat}
    se = n.state_encoder(x)
    ue = n.control_encoder(u)
    dzdt_(z,p,t) = n.dynamics.re(p)(cat(z, ue; dims=1)) # XXX Allocation
    ff = ODEFunction{false}(dzdt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff, se, (zero(dt), dt), n.dynamics.p)
    xp = solve(prob,args...;sense=sense, save_everystep=false, save_start=false, kwargs...)
    n.decoder(diffeqarray_to_array(Array(xp)))
end

function discrete_dynamics(n::NeuralDynamicsModel{F1,F2,G,H}, x, u, dt::F, args...;sense=InterpolatingAdjoint(autojacvec=ZygoteVJP()), kwargs...) where {F1,F2,G<:AugmentedNeuralDELayer,H,F <: AbstractFloat}
    se = n.state_encoder(x)
    ue = n.control_encoder(u)
    function dzdt_(z,p,t)
        inp = cat(z, ue; dims=1)
        n.dynamics.re(p)(DiffEqFlux.augment(inp, n.dynamics.augdim))
    end
    ff = ODEFunction{false}(dzdt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff, se, (zero(dt), dt), n.dynamics.p)
    xp = solve(prob,args...;sense=sense, save_everystep=false, save_start=false, kwargs...)
    n.decoder(diffeqarray_to_array(Array(xp)))
end



function discrete_dynamics(n::NeuralDynamicsModel, x, u, dt_B::AbstractVector{<:AbstractFloat}, args...;sense=InterpolatingAdjoint(autojacvec=ZygoteVJP()), kwargs...)
    se = n.state_encoder(x)
    ue = n.control_encoder(u)
    dzdt_(z,p,t) = n.dynamics.re(p)(z)
    ff = ODEFunction{false}(dzdt_,tgrad=basic_tgrad)
    # XXX Ugly!
    xp_B = reduce(hcat, (diffeqarray_to_array(Array(solve(ODEProblem{false}(ff, cat(se[:, i], ue[:, i]; dims=1), (zero(dt_B[i]), dt_B[i]), n.dynamics.p), args...;sense=sense, save_everystep=false, save_start=false, kwargs...))) for i in 1:length(dt_B)))
    n.decoder(xp_B)
end

"""
Rollout when same dt in the batch
"""
function rollout(n::NeuralDynamicsModel, x_0_B, us_Tm1_B, dt, alg, args...; kwargs...)
    T = size(us_Tm1_B)[end-1] # should maybe be just us_Tm1_B?
    B = size(x_0_B)[end]
    # Note: Things you have to do to avoid 
    #   ERROR: Mutating arrays is not supported
    x_T_B = Zygote.Buffer(x_0_B, size(x_0_B)[1:end-1]..., T, B)
    x_T_B[axes(x_0_B)[1:end-1]..., 1, :] = x_0_B
    for i in 2:T
        x_T_B[axes(x_0_B)[1:end-1]..., i, :] = discrete_dynamics(n, x_T_B[axes(x_0_B)[1:end-1]..., i-1, :], us_Tm1_B[:, i-1, :], dt, alg, args...; kwargs...)
    end
    return copy(x_T_B)
end

