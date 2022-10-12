function lrate(opt::Flux.Optimise.Optimiser)
    for o in opt
        if hasproperty(o, :eta)
            return o.eta
        end
    end
end

function lrate(o)
    if hasproperty(o, :eta)
        return o.eta
    end
end

function setlrate!(o, eta)
    if hasproperty(o, :eta)
        o.eta = eta
        return o.eta
    end
end

function setlrate!(opt::Flux.Optimise.Optimiser, eta)
    for o in opt
        setlrate!(o, eta)
    end
end

"""
Compute both loss as well as the gradients and return both.
"""
function lossgrads(f, args...)
    val, back = Zygote.pullback(f, args...)
    grad = back(Zygote.sensitivity(val))
    return val, grad
end

function gradinfo(gs)
    # XXX Ugly

    gnorm1 = [norm(g) for g in gs if typeof(g) <: Array]
    #gnorm2 = [norm(g[].contents.p) for g in gs if typeof(g) <:Base.RefValue]
    gnorm2 = [0.0]
    #max1 = maximum([maximum(g) for g in gs if typeof(g) <: Array])
    max1 = 0.0
    nan = sum([sum(isnan, g) for g in gs if typeof(g) <: Array])
    #g2 = [maximum(g[].contents.p) for g in gs if typeof(g) <: Base.RefValue]
    #max2 = maximum(g2)
    return (nan=nan, inf=0, max=max1, l2=norm(vcat(gnorm1, gnorm2)),)
#=     gs1 = filter(x -> isa(x, Array), gs)
    gs2 = filter(x -> isa(x, Base.RefValue), gs)
    gs2 = [g[].contents.p for g in gs2]
    return (nan = mapreduce(x -> sum(isnan, x), +, (gs1, gs2)),
     inf = mapreduce(x -> sum(isinf, x), +, (gs1, gs2)),
     max = mapreduce(x -> maximum(abs, x), max, (gs1, gs2)),
     l2 = mapreduce(norm, +, (gs1, gs2)),
     ) =#
end

function num_parameters(nn)
    return sum(length(p) for p in Flux.params(nn))
end
