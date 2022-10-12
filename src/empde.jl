using DiffEqFlux: NeuralDELayer


Mill.bagcount!(false)

struct EMPDynamicsModel{G<:NeuralDELayer} <: AbstractModel
    dynamics::G
    xdim::Int
    mdim::Int
    hdim::Int
end

Flux.@functor EMPDynamicsModel

struct EMPEDDynamicsModel{E,G<:NeuralDELayer,D} <: AbstractModel
    encoder::E
    dynamics::G
    x_decoder::D
    xdim::Int
    mdim::Int
    hdim::Int
end

Flux.@functor EMPEDDynamicsModel

struct EMPEDBypassDynamicsModel{E,G<:NeuralDELayer, D} <: AbstractModel
    encoder::E
    dynamics::G
    x_decoder::D
    xdim::Int
    mdim::Int
    hdim::Int
    ldim::Int
end

Flux.@functor EMPEDBypassDynamicsModel

struct EMPEDMsgDynamicsModel{E,G<:NeuralDELayer, D, M} <: AbstractModel
    encoder::E
    dynamics::G
    x_decoder::D
    m_decoder::M
    xdim::Int
    mdim::Int
    hdim::Int
end

Flux.@functor EMPEDMsgDynamicsModel

# TODO
# if the zero_msg ifelse turn out to be too slow, should switch to traits to implement separate functions

function discrete_dynamics(n::EMPDynamicsModel{G}, g, x, u, dt::F, args...; sense=InterpolatingAdjoint(autojacvec=ZygoteVJP()), zero_msg=false, kwargs...)  where {F <: AbstractFloat, G}
    # all part of ode
    function dzdt_(z, p, t)
        if zero_msg
            return n.dynamics.re(p)(vcat(z[1:n.xdim, :], u, zeros(eltype(z), n.mdim, size(z)[end])))
        else
            return n.dynamics.re(p)(vcat(z[1:n.xdim, :], u, SegmentedMean(n.mdim)(z[n.xdim+1:n.xdim+n.mdim, :], ScatteredBags(g.fadjlist))))
        end
    end
    ff = ODEFunction{false}(dzdt_)
    prob = ODEProblem{false}(ff, x, (zero(dt), dt), n.dynamics.p)
    xp_feat = solve(prob, args...;sense=sense, save_everystep=false, save_start=false, kwargs...)
    xp = diffeqarray_to_array(Array(xp_feat))
    return xp
end

function discrete_dynamics(n::EMPEDDynamicsModel, g, x, u, dt::F, args...; sense=InterpolatingAdjoint(autojacvec=ZygoteVJP()), zero_msg=false, kwargs...)  where {F <: AbstractFloat}
    # all to common encoder then ode then common decoder
    if zero_msg
        z = vcat(x[1:n.xdim, :], u, zeros(eltype(x), n.mdim, size(x)[end]))
    else
        z = vcat(x[1:n.xdim, :], u, SegmentedMean(n.mdim)(x[n.xdim+1:n.xdim+n.mdim, :], ScatteredBags(g.fadjlist)))
    end
    z_e = n.encoder(z)
    dzdt_(z, p, t) = n.dynamics.re(p)(z)
    ff = ODEFunction{false}(dzdt_)
    prob = ODEProblem{false}(ff, z_e, (zero(dt), dt), n.dynamics.p)
    xp_feat_ = solve(prob, args...;sense=sense, save_everystep=false, save_start=false, kwargs...)
    xp_feat = diffeqarray_to_array(Array(xp_feat_))
    x_out = n.x_decoder(xp_feat)
    return x_out
end

function discrete_dynamics(n::EMPEDBypassDynamicsModel, g, x, u, dt::F, args...; sense=InterpolatingAdjoint(autojacvec=ZygoteVJP()), zero_msg=false, kwargs...)  where {F <: AbstractFloat}
    x_e = n.encoder(vcat(x[1:n.xdim, :], u))
    if zero_msg
        x_in = vcat(x_e, zeros(eltype(x), n.mdim, size(x)[end]))
    else
        x_in = vcat(x_e, SegmentedMean(n.mdim)(x[n.xdim+1:n.xdim+n.mdim, :], ScatteredBags(g.fadjlist)))
    end
    dzdt_(z, p, t) = n.dynamics.re(p)(z)
    ff = ODEFunction{false}(dzdt_)
    prob = ODEProblem{false}(ff, x_in, (zero(dt), dt), n.dynamics.p)
    xp_feat_ = solve(prob, args...; sense=sense, save_everystep=false, save_start=false, kwargs...)
    xp_feat = diffeqarray_to_array(Array(xp_feat_))
    #ldim = size(xp_feat, 1) - n.mdim # would be n.hdim in our case
    x_d = n.x_decoder(xp_feat[1:n.ldim, :])
    x_out = vcat(x_d, xp_feat[n.ldim+1:n.ldim+n.mdim, :])
    return x_out
end

function discrete_dynamics(n::EMPEDMsgDynamicsModel, g, x, u, dt::F, args...; sense=InterpolatingAdjoint(autojacvec=ZygoteVJP()), zero_msg=false, kwargs...)  where {F <: AbstractFloat}
    # all to common encoder then ode then different decoder
    if zero_msg
        z = vcat(x[1:n.xdim, :], u, zeros(eltype(x), n.mdim, size(x)[end]))
    else
        z = vcat(x[1:n.xdim, :], u, SegmentedMean(n.mdim)(x[n.xdim+1:n.xdim+n.mdim, :], ScatteredBags(g.fadjlist)))
    end
    z_e = n.encoder(z)
    dzdt_(z, p, t) = n.dynamics.re(p)(z)
    ff = ODEFunction{false}(dzdt_)
    prob = ODEProblem{false}(ff, z_e, (zero(dt), dt), n.dynamics.p)
    xp_feat_ = solve(prob, args...;sense=sense, save_everystep=false, save_start=false, kwargs...)
    xp_feat = diffeqarray_to_array(Array(xp_feat_))
    x_d = n.x_decoder(xp_feat[1:Int(0.5*n.hdim), :])
    m_d = n.m_decoder(xp_feat[Int(0.5*n.hdim)+1:n.hdim, :])
    x_out = vcat(x_d, m_d)
    return x_out
end

const EMPModel = Union{EMPDynamicsModel, EMPEDDynamicsModel, EMPEDBypassDynamicsModel, EMPEDMsgDynamicsModel}

function rollout(n::EMPModel, g_T, x0_B, us_Tm1_B, dt, args...; zero_msg=false, kwargs...)
    #T = Zygote.@ignore size(us_Tm1_B)[end-1]
    T = Zygote.@ignore length(g_T)
    B = Zygote.@ignore size(x0_B)[end]
    msg = zeros(eltype(x0_B), n.mdim, B) # TODO check if we need num_components?
    x_T_B = Zygote.Buffer(x0_B, n.xdim+n.mdim, T, B)
    x_T_B[.., 1, :] = vcat(x0_B, msg)
    for i in 2:T
        x_T_B[.., i, :] = discrete_dynamics(n, g_T[i-1], x_T_B[.., i-1, :], us_Tm1_B[.., i-1, :], dt, args...; zero_msg=zero_msg, kwargs...)
    end
    return copy(x_T_B)[1:n.xdim, ..]
end
