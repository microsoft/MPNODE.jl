
###
# TimeSeries Norms
#
function time_series_norm(xs::AbstractArray{T, 2}, dt::F, p=2; agg=sum) where {T <: AbstractFloat, F<:AbstractFloat}
    agg(mapslices(x->time_series_norm(x, dt, p), xs; dims=2))
end

function time_series_norm(xs::AbstractArray{T, 3}, dt::F, p=2; agg=mean) where {T <: AbstractFloat, F<:AbstractFloat}
    agg(mapslices(x->time_series_norm(x, dt, p), xs; dims=3))
end

function time_series_norm(x::AbstractArray{T, 3}, x̂::AbstractArray{T, 3}, dt::F, p=2) where {T,F}
    return time_series_norm(x.-x̂, dt, p)
end
function time_series_norm(x::AbstractArray{T, 3}, x̂::AbstractArray{T, 3}, dt::AbstractVector{F}, p=2) where {T,F}
    return time_series_norm(x.-x̂, dt[1], p)
end

function time_series_norm(xs::AbstractVector{T}, dt::F, p=2) where {T <: AbstractFloat, F<:AbstractFloat}
    # TODO: write this in a form that's differentiable
    sumd = zero(eltype(xs))
    for i in 1:size(xs, 1)
        if i == 1
            prevx = zero(eltype(xs))
        else
            prevx = xs[i-1]
        end
        if i == size(xs, 1)
            nextx = zero(eltype(xs))
        else
            nextx = xs[i+1]
        end

        if xs[i] * prevx <= zero(eltype(xs))
            Am1 = dt/2 * abs(xs[i])
        else
            Am1 = dt/2 * abs2(xs[i]) / (abs(xs[i]) + abs(prevx))
        end

        if xs[i] * nextx <= zero(eltype(xs))
            Ap1 = dt/2 * abs(xs[i])
        else
            Ap1 = dt/2 * abs2(xs[i]) / (abs(xs[i]) + abs(nextx))
        end

        sumd += (Am1 + Ap1)^p
    end
    return sumd^(1/p)
end

function time_series_mse(x::AbstractArray{T, 3}, x̂::AbstractArray{T, 3}; agg=mean) where {T}
    agg(sum(abs2.(x .- x̂), dims=(1, 2)))
end

function time_series_huber(x::AbstractArray{T, 3}, x̂::AbstractArray{T, 3}; agg=mean, δ=Flux.ofeltype(x̂, 1)) where {T}
    abs_error = abs.(x̂ .- x)
    #TODO: remove dropgrad when Zygote can handle this function with CuArrays
    temp = Zygote.dropgrad(abs_error .<  δ)
    z = Flux.ofeltype(x̂, 0.5)
    agg(sum(((abs_error.^2) .* temp) .* z .+ δ*(abs_error .- z*δ) .* (1 .- temp), dims=(1, 2))) # XXX Allocation
end


### 
# Shooting Losses
#
function single_shooting_loss(model::NeuralDynamicsModel, x_D_T_B, u_D_T_B, dt, alg; lossfn=time_series_huber, kwargs...)
    # TODO
    # For full generality we will have to construct masks and PaddedViews.jl along the time axis
    # Maybe that's how the dataset should store things as well
    x_D_0_B = MPNODE.timeidx(x_D_T_B, 1)
    xpred_D_T_B = rollout(model, x_D_0_B, u_D_T_B, dt[1], alg)
    return lossfn(x_D_T_B[:, 2:end, :], xpred_D_T_B[:, 2:end, :])
    #return Flux.Losses.mse(x_D_T_B, xpred_D_T_B)
end

function graph2batch(x_D_G_T_B::AbstractArray{T,4}) where {T}
    x_D_T_G_B = permutedims(x_D_G_T_B, (1, 3, 2, 4)) # XXX Allocation
    x_D_T_GB = reshape(x_D_T_G_B, size(x_D_T_G_B)[1:end-2]..., prod(size(x_D_T_G_B)[end-1:end]))
    return x_D_T_GB
end

function graph2batch(x_D_G_B::AbstractArray{T,3}) where {T}
    x_D_GB = reshape(x_D_G_B, size(x_D_G_B)[1:end-2]..., prod(size(x_D_G_B)[end-1:end]))
    return x_D_GB
end

function batch2graph(x_D_T_GB, adj_mat)
    G = size(adj_mat, 1)
    x_D_T_G_B = reshape(x_D_T_GB, size(x_D_T_GB)[1:2]..., G, :)
    x_D_G_T_B = permutedims(x_D_T_G_B, (1, 3, 2, 4))
    return x_D_G_T_B
end

function single_shooting_loss(model::NeuralMPDynamicsModel, g_T_B, x_D_G_T_B, u_D_G_T_B, dt, alg; lossfn=time_series_huber, kwargs...)
    #x_D_0_GB = reshape(x_D_G_0_B, size(x_D_G_0_B)[1:end-2]..., prod(size(x_D_G_0_B)[end-1:end]))
    x_D_T_B = graph2batch(x_D_G_T_B)
    u_D_T_B = graph2batch(u_D_G_T_B)
    x_D_0_B = MPNODE.timeidx(x_D_T_B, 1)
    
    # Have to ignore otherwise Zygote runs into issues trying to differentiate
    g_T = Zygote.@ignore [reduce(blockdiag, g_T_B[i, :]) for i in 1:size(g_T_B, 1)]
    xpred_D_T_B = rollout(model, g_T, x_D_0_B, u_D_T_B, dt[1], alg; kwargs...)
    return lossfn(x_D_T_B[:, 2:end, :], xpred_D_T_B[:, 2:end, :])
end

function multiple_shooting_loss(model, x_D_T_B, u_D_T_B, discontinuity_weight, dt, alg; kwargs...)

end

function single_shooting_loss(model::EMPModel, g_T_B, x_D_G_T_B, u_D_G_T_B, dt, alg; lossfn=time_series_huber, kwargs...)
    #x_D_0_GB = reshape(x_D_G_0_B, size(x_D_G_0_B)[1:end-2]..., prod(size(x_D_G_0_B)[end-1:end]))
    x_D_T_B = graph2batch(x_D_G_T_B)
    u_D_T_B = graph2batch(u_D_G_T_B)
    x_D_0_B = MPNODE.timeidx(x_D_T_B, 1)
    
    # Have to ignore otherwise Zygote runs into issues trying to differentiate
    g_T = Zygote.@ignore [reduce(blockdiag, g_T_B[i, :]) for i in 1:size(g_T_B, 1)]
    xpred_D_T_B = rollout(model, g_T, x_D_0_B, u_D_T_B, dt[1], alg; kwargs...)
    return lossfn(x_D_T_B[:, 2:end, :], xpred_D_T_B[:, 2:end, :])
end

### 
# Single Step Losses
#
function onesteploss(model::NeuralDynamicsModel, x, u, xp, dt, alg; lossfn=Flux.Losses.mse, kwargs...)
    xphat = discrete_dynamics(model, x, u, dt, alg; kwargs...)
    return lossfn(xp, xphat)
end

function onesteploss(model::NeuralMPDynamicsModel, g, x, u, xp, dt, alg; lossfn=Flux.Losses.mse, kwargs...)
    x = graph2batch(x)
    u = graph2batch(u)
    xp = graph2batch(xp)
    g = Zygote.@ignore reduce(blockdiag, g)
    xphat = discrete_dynamics(model, g, x, u, dt, alg; kwargs...)
    return lossfn(xp, xphat)
end

###
# Collocation Losses
# 
function collocateloss(model::NeuralDynamicsModel, x, u, dx; lossfn=Flux.Losses.mse)
    dx̂ = dynamics(model, x, u)
    return lossfn(dx, dx̂)
end

function collocateloss(model::NeuralMPDynamicsModel, g, x, u, dx; lossfn=Flux.Losses.mse)
    x = graph2batch(x)
    u = graph2batch(u)
    dx = graph2batch(dx)
    g = Zygote.@ignore reduce(blockdiag, g)
    dx̂ = dynamics(model, g, x, u)
    return lossfn(dx, dx̂)
end

function fg2batch(x_D_G_B::AbstractArray{T,3}, num_factors) where {T}
    msg = zeros(eltype(x_D_G_B), size(x_D_G_B, 1), num_factors, size(x_D_G_B)[3:end]...)
    aug_x = hcat(x_D_G_B, msg)
    return reshape(aug_x, size(aug_x, 1), prod(size(aug_x)[2:end]))
end

function onesteploss(model::Union{FactorMPDynamicsModel,FactorIntMPDynamicsModel}, n2fg, f2ng, x, u, xp, dt, alg; lossfn=Flux.Losses.mse, kwargs...)
    num_factors = Zygote.@ignore length(unique(reduce(vcat, (f2ng[1].fadjlist))))
    x = fg2batch(x, num_factors)
    u = fg2batch(u, num_factors)
    n2fg = Zygote.@ignore reduce(blockdiag, n2fg)
    f2ng = Zygote.@ignore reduce(blockdiag, f2ng)
    xphat, _ = discrete_dynamics(model, n2fg, f2ng, x, u, dt, alg; kwargs...)
    return lossfn(reshape(xp, size(xp, 1), :), xphat)
end

function single_shooting_loss(model::Union{FactorMPDynamicsModel,FactorIntMPDynamicsModel}, n2fg_T_B, f2ng_T_B, x_D_G_T_B, u_D_G_T_B, dt, alg, args...; lossfn=MPNODE.time_series_huber, kwargs...)
    x_D_G_0_B = timeidx(x_D_G_T_B, 1)
    num_factors = Zygote.@ignore length(unique(reduce(vcat, (f2ng_T_B[1, 1].fadjlist))))
    n2fg_T = Zygote.@ignore [reduce(blockdiag, n2fg_T_B[i, :]) for i in 1:size(n2fg_T_B, 1)]
    f2ng_T = Zygote.@ignore [reduce(blockdiag, f2ng_T_B[i, :]) for i in 1:size(f2ng_T_B, 1)]
    xpred_D_G_T_B = rollout(model, n2fg_T, f2ng_T, x_D_G_0_B, u_D_G_T_B, dt[1], alg, num_factors; kwargs...)
    xpred_D_T_GB = graph2batch(xpred_D_G_T_B)
    x_D_T_GB = graph2batch(x_D_G_T_B)
    return lossfn(x_D_T_GB[:, 2:end, :], xpred_D_T_GB[:, 2:end, :])
end

