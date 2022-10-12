using Mill
using Flux
import Flux
using LightGraphs
using Statistics
using DiffEqFlux: basic_tgrad, NeuralDELayer
using DiffEqSensitivity: ZygoteVJP, InterpolatingAdjoint
using OrdinaryDiffEq
using RobotDynamics: AbstractModel
import RobotDynamics: dynamics, discrete_dynamics

###
# IMPORTANT
#
Mill.bagcount!(false)

abstract type MessagePassingNN end

struct GNN{L,M, R} <: MessagePassingNN
	lift::L
	mp::M
	m::R
end

Flux.@functor GNN

function mpstep(m::GNN, xx::ArrayNode, bags, n)
	n == 0 && return(xx)
	mpstep(m, m.mp(BagNode(xx, bags)), bags, n - 1)
end

function (m::GNN)(g, x, n)
	xx = m.lift(x)
	bags = Mill.ScatteredBags(g.fadjlist)
    o = mpstep(m, xx, bags, n)
    return m.m(o.data)
end

struct GNNSC{L,M,R} <: MessagePassingNN
    lift::L
    mp::M
    m::R
end

Flux.@functor GNNSC

function mpstep(m::GNNSC, xx::ArrayNode, bags, n)
    n == 0 && return xx
    scmp = m.mp(BagNode(xx, bags)).data .+ xx.data
    mpstep(m, ArrayNode(scmp), bags, n-1)
end

function (m::GNNSC)(g, x, n)
    xx = m.lift(x)
    bags = Mill.ScatteredBags(g.fadjlist)
    o = mpstep(m, xx, bags, n)
    return m.m(o.data .+ xx.data)
end

struct NeuralMPDELayer{M,P,RE} <: NeuralDELayer
    model::M
    p::P
    re::RE
    function NeuralMPDELayer(model; p=nothing)
        _p, re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        new{typeof(model),typeof(p),typeof(re)}(model, p, re)
    end
end

# !!! TODO: this doesn't work when doing Flux.destructure
Flux.@functor NeuralMPDELayer

function (n::NeuralMPDELayer)(g, x, depth, p=n.p)
    n.model(g, ArrayNode(x), depth)
end


struct NeuralMPDynamicsModel{F1, F2, G, H} <: AbstractModel
    state_encoder::F1
    control_encoder::F2
    dynamics::G
    decoder::H
end

Flux.@functor NeuralMPDynamicsModel

dynamics(sys::NeuralMPDynamicsModel{F1,F2,G,H}, g, x, u; depth=1) where {F1, F2, G <: NeuralMPDELayer, H} = sys.dynamics.re(sys.dynamics.p)(g, ArrayNode(cat(sys.state_encoder(x), sys.control_encoder(u); dims=1)), depth)

function discrete_dynamics(n::NeuralMPDynamicsModel{F1,F2,G,H}, g, x, u, dt::F, args...; depth=1, sense = InterpolatingAdjoint(autojacvec=ZygoteVJP()), kwargs...) where {F1, F2, F <: AbstractFloat, G <: NeuralMPDELayer, H}
    se = n.state_encoder(x)
    ue = n.control_encoder(u)
    dzdt_(z,p,t) = n.dynamics.re(p)(g, ArrayNode(cat(z, ue; dims=1)), depth) # XXX Allocation
    ff = ODEFunction{false}(dzdt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff, se, (zero(dt), dt), n.dynamics.p)
    xp = solve(prob,args...;sense=sense, save_everystep=false, save_start=false, kwargs...)
    n.decoder(diffeqarray_to_array(Array(xp)))
end

function discrete_dynamics(n::NeuralMPDynamicsModel{F1,F2,G,H}, g, x, u, dt::F, args...; depth=1, sense = InterpolatingAdjoint(autojacvec=ZygoteVJP()), kwargs...) where {F1, F2, F <: AbstractFloat, G <: SimpleNeuralDELayer, H}
    se = n.state_encoder(x)
    ue = n.control_encoder(u)
    dzdt_(z, p, t) = n.dynamics.re(p)(cat(z, ue; dims=1))
    ff = ODEFunction{false}(dzdt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff, se, (zero(dt), dt), n.dynamics.p)
    xp = solve(prob,args...;sense=sense, save_everystep=false, save_start=false, kwargs...)
    n.decoder(diffeqarray_to_array(Array(xp)))    
end

function rollout(n::NeuralMPDynamicsModel, g_T, x_0_B, us_Tm1_B, dt, alg, args...; zero_msg=false, kwargs...)
    T = size(us_Tm1_B)[end-1] # should maybe be just us_Tm1_B?
    B = size(x_0_B)[end]
    # Note: Things you have to do to avoid 
    #   ERROR: Mutating arrays is not supported
    x_T_B = Zygote.Buffer(x_0_B, size(x_0_B)[1:end-1]..., T, B)
    x_T_B[axes(x_0_B)[1:end-1]..., 1, :] = x_0_B
    for i in 2:T
        x_T_B[axes(x_0_B)[1:end-1]..., i, :] = discrete_dynamics(n, g_T[i-1], x_T_B[axes(x_0_B)[1:end-1]..., i-1, :], us_Tm1_B[:, i-1, :], dt, alg, args...; kwargs...)
    end
    return copy(x_T_B)
end