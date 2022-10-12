using StatsBase

function ztransform(data::Dataset)
    X = reshape(data.xs, size(data.xs, 1), :)
    XdTr = fit(ZScoreTransform, X, dims=2)
    Xt = StatsBase.transform(XdTr, X)
    Xt = reshape(Xt, size(data.xs)...)
    if !iszero(sum(data.us)) || (prod(size(data.us)) != 0)
        U = reshape(data.us, size(data.us, 1), :)
        UdTr = fit(ZScoreTransform, U, dims=2)
        Ut = StatsBase.transform(UdTr, U)
        Ut = reshape(Ut, size(data.us)...)
    else
        Ut = data.us
        UdTr = nothing
    end
    return Dataset(Xt, Ut, data.dt), (Xtr=XdTr, Utr=UdTr)
end

function ztransform(data::GraphDataset)
    X = reshape(data.xs, size(data.xs, 1), :)
    XdTr = fit(ZScoreTransform, X, dims=2)
    Xt = StatsBase.transform(XdTr, X)
    Xt = reshape(Xt, size(data.xs)...)
    if !iszero(sum(data.us)) || (prod(size(data.us)) != 0)
        U = reshape(data.us, size(data.us, 1), :)
        UdTr = fit(ZScoreTransform, U, dims=2)
        Ut = StatsBase.transform(UdTr, U)
        Ut = reshape(Ut, size(data.us)...)
    else
        Ut = data.us
        UdTr = nothing
    end
    return GraphDataset(data.gs, Xt, Ut, data.dt), (Xtr=XdTr, Utr=UdTr)
end

function simpletransform(data::GraphDataset)
    X = reshape(data.xs, size(data.xs, 2), :)
    mu = mean(X, dims=2)
    sigma = std(X, dims=2)
    Xt = (X .- mu) ./ sigma
    Xt = reshape(Xt, size(data.xs)...)
    Ut = data.us
    UdTr = nothing
    return GraphDataset(data.gs, Xt, Ut, data.dt), (Xtr=(mean=mu, scale=sigma),Utr=UdTr)
end

function simpletransform(dt, data::GraphDataset)
    X = reshape(data.xs, size(data.xs, 2), :)
    Xt = (X .- dt.Xtr.mean) ./ dt.Xtr.scale
    Xt = reshape(Xt, size(data.xs)...)
    Ut = data.us
    return GraphDataset(data.gs, Xt, Ut, data.dt)
end

function ztransform(data::FactorGraphDataset)
    X = reshape(data.xs, size(data.xs, 1), :)
    XdTr = fit(ZScoreTransform, X, dims=2)
    Xt = StatsBase.transform(XdTr, X)
    Xt = reshape(Xt, size(data.xs)...)
    if !iszero(sum(data.us)) || (prod(size(data.us)) != 0)
        U = reshape(data.us, size(data.us, 1), :)
        UdTr = fit(ZScoreTransform, U, dims=2)
        Ut = StatsBase.transform(UdTr, U)
        Ut = reshape(Ut, size(data.us)...)
    else
        Ut = data.us
        UdTr = nothing
    end
    return FactorGraphDataset(data.n2fgs, data.f2ngs, Xt, Ut, data.dt), (Xtr=XdTr, Utr=UdTr)
end

function ztransform(dt, data::Dataset)
    X = reshape(data.xs, dt.Xtr.len, :) # Assuming len will always be smaller
    Xt = StatsBase.transform(dt.Xtr, X)
    Xt = reshape(Xt, size(data.xs)...)
    if !iszero(sum(data.us)) || (prod(size(data.us)) != 0)
        U = reshape(data.us, dt.Utr.len, :)
        Ut = StatsBase.transform(dt.Utr, U)
        Ut = reshape(Ut, size(data.us)...)
    else
        Ut = data.us
    end
    return Dataset(Xt, Ut, data.dt)
end

function ztransform(dt, data::GraphDataset)
    X = reshape(data.xs, dt.Xtr.len, :) # Assuming len will always be smaller
    Xt = StatsBase.transform(dt.Xtr, X)
    Xt = reshape(Xt, size(data.xs)...)
    if !iszero(sum(data.us)) || (prod(size(data.us)) != 0)
        U = reshape(data.us, dt.Utr.len, :)
        Ut = StatsBase.transform(dt.Utr, U)
        Ut = reshape(Ut, size(data.us)...)
    else
        Ut = data.us
    end
    return GraphDataset(data.gs, Xt, Ut, data.dt)
end

function ztransform(dt, data::FactorGraphDataset)
    X = reshape(data.xs, dt.Xtr.len, :) # Assuming len will always be smaller
    Xt = StatsBase.transform(dt.Xtr, X)
    Xt = reshape(Xt, size(data.xs)...)
    if !iszero(sum(data.us)) || (prod(size(data.us)) != 0)
        U = reshape(data.us, dt.Utr.len, :)
        Ut = StatsBase.transform(dt.Utr, U)
        Ut = reshape(Ut, size(data.us)...)
    else
        Ut = data.us
    end
    return FactorGraphDataset(data.n2fgs, data.f2ngs, Xt, Ut, data.dt)
end

function truncate_timesteps(data::Dataset, horizon::Int)
    Dataset(data.xs[.., 1:horizon, :], data.us[.., 1:horizon, :], data.dt)
end

function truncate_timesteps(data::GraphDataset, horizon::Int)
    GraphDataset(data.gs[1:horizon, :], data.xs[.., 1:horizon, :], data.us[.., 1:horizon, :], data.dt)
end

function truncate_timesteps(data::FactorGraphDataset, horizon::Int)
    FactorGraphDataset(data.n2fgs[1:horizon, :], data.f2ngs[1:horizon, :], data.xs[.., 1:horizon, :], data.us[.., 1:horizon, :], data.dt)
end

function subset(data::Dataset, subidx)
    Dataset(data.xs[.., subidx], data.us[.., subidx], data.dt[.., subidx])
end

function subset(data::GraphDataset, subidx)
    GraphDataset(data.gs[.., subidx], data.xs[.., subidx], data.us[.., subidx], data.dt[.., subidx])
end

function tocollocation(data::Dataset)
    tocollocation(data.xs, data.us, data.dt[1])
end

function tocollocation(data::GraphDataset)
    x_oldshape = size(data.xs)
    u_oldshape = size(data.us)
    gs, xs, us, dxs = tocollocation(data.gs, reshape(data.xs, :, x_oldshape[end-1:end]...), reshape(data.us, :, x_oldshape[end-1:end]...), data.dt[1])
    return gs, reshape(xs, x_oldshape[1:end-2]..., size(xs)[end]), reshape(us, u_oldshape[1:end-2]..., size(us)[end]), reshape(dxs, x_oldshape[1:end-2]..., size(dxs)[end])
end


function toonestep(data::Dataset)
    toonestep(data.xs, data.us)
end

function toonestep(data::GraphDataset)
    x_oldshape = size(data.xs)
    u_oldshape = size(data.us)
    gs, xs, us, xsp = toonestep(data.gs, reshape(data.xs, :, x_oldshape[end-1:end]...), reshape(data.us, :, x_oldshape[end-1:end]...))
    return gs, reshape(xs, x_oldshape[1:end-2]..., size(xs)[end]), reshape(us, u_oldshape[1:end-2]..., size(us)[end]), reshape(xsp, x_oldshape[1:end-2]..., size(xsp)[end])
end

function tokstep(data::Dataset, k)
    newxs, newus = tokstep(data.xs, data.us, k)
    Dataset(newxs, newus, Fill(data.dt[1], size(newxs)[end])) # Assuming same dt
end

function tokstep(data::GraphDataset, k)
    gs, xs, us = tokstep(data.gs, data.xs, data.us, k)
    GraphDataset(gs, xs, us, Fill(data.dt[1], size(xs)[end])) # Assuming same dt
end

function tocollocation(xs, us, dt)
    dxs, xs = reduce((x, xp)->(hcat(x[1], xp[1]), hcat(x[2], xp[2])), mapslices(x->collocate_data(x, 0.0:dt:dt*(size(xs)[end-1]-1)), xs, dims=(1, 2)))
    us = reshape(us, size(us, 1), prod(size(us)[2:end]))
    @assert size(dxs)[end] == size(us)[end]
    return (xs, us, dxs)
end

function tocollocation(gs, xs, us, dt)
    return (reshape(gs[1:end, :], :), tocollocation(xs, us, dt)...)
end

function toonestep(xs, us)
    # xs, xsp = reduce((x, xp)->(hcat(x[1], xp[1]), hcat(x[2], xp[2])), slidingwindow(i->i+1, xs, 1, LearnBase.ObsDim.Constant(2)))
    # xs = reshape(xs, size(xsp)...)
    prev = MPNODE.timeidx(xs, 1:size(xs)[end-1]-1)
    next = MPNODE.timeidx(xs, 2:size(xs)[end-1])
    prev = reshape(prev, size(xs)[1:end-2]..., :)
    next = reshape(next, size(xs)[1:end-2]..., :)
    us = us[:, 1:end-1, :]
    us = reshape(us, size(us, 1), prod(size(us)[2:end]))
    return (prev, us, next)
end

function toonestep(gs, xs, us)
    return (reshape(gs[1:end-1, :], :), toonestep(xs, us)...)
end

function tokstep(xs, us, k)
    Nx = ndims(xs)
    Nu = ndims(us)
    T = size(xs)[end-1]
    intrajlen = T - k + 1
    newxs = [MPNODE.timeidx(xs, idx:idx+k-1) for idx in 1:intrajlen]
    newus = [MPNODE.timeidx(us, idx:idx+k-1) for idx in 1:intrajlen]
    newxs = reduce((x,xs)->cat(x, xs; dims=Nx), newxs)
    newus = reduce((x,xs)->cat(x, xs; dims=Nu), newus)
    return newxs, newus
end

function tokstep(gs, xs, us, k)
    intrajlen = size(gs, 1) - k + 1
    (reduce(hcat, [gs[idx:idx+k-1, :] for idx in 1:intrajlen]), tokstep(xs, us, k)...)
end

function toonestep(data::FactorGraphDataset)
    x_oldshape = size(data.xs)
    u_oldshape = size(data.us)

    n2fgs, f2ngs, xs, us, xsp = MPNODE.toonestep(data.n2fgs, data.f2ngs, reshape(data.xs, :, x_oldshape[end-1:end]...), reshape(data.us, :, x_oldshape[end-1:end]...))
    return n2fgs, f2ngs, reshape(xs, x_oldshape[1:end-2]..., size(xs)[end]), reshape(us, u_oldshape[1:end-2]..., size(us)[end]), reshape(xsp, x_oldshape[1:end-2]..., size(xsp)[end])
end

function toonestep(n2fgs, f2ngs, xs, us)
    return (reshape(n2fgs[1:end-1, :], :), reshape(f2ngs[1:end-1, :], :), MPNODE.toonestep(xs, us)...)
end

function tokstep(data::FactorGraphDataset, k)
    n2fgs, f2ngs, xs, us = MPNODE.tokstep(data.n2fgs, data.f2ngs, data.xs, data.us, k)
    FactorGraphDataset(n2fgs, f2ngs, xs, us, Fill(data.dt[1], size(xs)[end])) # Assuming same dt
end

function tokstep(n2fgs, f2ngs, xs, us, k)
    intrajlen = size(n2fgs, 1) - k + 1
    (reduce(hcat, [n2fgs[idx:idx+k-1, :] for idx in 1:intrajlen]), reduce(hcat, [f2ngs[idx:idx+k-1, :] for idx in 1:intrajlen]), MPNODE.tokstep(xs, us, k)...)
end

function tsdataloader(data::FactorGraphDataset; batchsize=1, shuffle=false)
    Flux.Data.DataLoader((data.n2fgs, data.f2ngs, data.xs, data.us, data.dt), batchsize=batchsize, shuffle=shuffle)
end
