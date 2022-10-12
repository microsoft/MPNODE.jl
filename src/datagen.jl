using FillArrays
using ArraysOfArrays: VectorOfArrays
using Distributions: Uniform
import StatsBase
import LazyStack: stack

abstract type AbstractDataset end

# assuming ...dim x T x B
batchidx(x::AbstractArray, idx) = x[axes(x)[1:end-1]..., idx]
batchidx(x::Vector{<:AbstractArray}, idx) = x[idx]
batchidx(x::VectorOfArrays, idx) = x[idx]

timeidx(x::AbstractArray, idx) = x[axes(x)[1:end-2]..., idx, :]
function timeidx(x::Union{VectorOfArrays, Vector{<:AbstractArray}}, idx)
    # TODO: won't work for more than 2 dim i.e only works dxT
    reduce(hcat, (x[i][axes(x[i])[1:end-1]..., idx] for i in 1:size(x, 1)))
end

StatsBase.nobs(data::AbstractDataset) = size(data.xs)[end]
LearnBase.getobs(data::AbstractDataset, idx::Int) = (batchidx(data.xs, idx), batchidx(data.us, idx))

struct Dataset{F,H,N} <: AbstractDataset
    xs::F
    us::H
    dt::N
end

Base.eltype(d::Dataset) = eltype(d.xs)

function stack(ds::Vector{<:Dataset})
    newxs = stack([d.xs for d in ds])
    newus = stack([d.us for d in ds])
    newdt = stack([d.dt for d in ds])
    newxs = reshape(newxs, size(newxs)[1:end-2]..., :)
    newus = reshape(newus, size(newus)[1:end-2]..., prod(size(newus)[end-1:end]))
    newdt = reshape(newdt, size(newdt)[1:end-2]..., :)
    return Dataset(newxs, newus, newdt)
end

function MLDataPattern.shuffleobs(data::Dataset; obsdim = LearnBase.default_obsdim(data), rng::AbstractRNG = Random.GLOBAL_RNG)
    idxs = 1:nobs(data)
    idxs = shuffle(rng, idxs)
    return Dataset(data.xs[.., idxs], data.us[.., idxs], data.dt[idxs])
end

struct GraphDataset{G,F,H, N} <: AbstractDataset
    gs::G
    xs::F
    us::H
    dt::N
end

Base.eltype(d::GraphDataset) = eltype(d.xs)

function stack(ds::Vector{<:GraphDataset})
    newgs = stack([d.gs for d in ds])
    newxs = stack([d.xs for d in ds])
    newus = stack([d.us for d in ds])
    newdt = stack([d.dt for d in ds])
    newgs = reshape(newgs, size(newgs)[1:end-2]..., :)
    newxs = reshape(newxs, size(newxs)[1:end-2]..., :)
    newus = reshape(newus, size(newus)[1:end-2]..., prod(size(newus)[end-1:end]))
    newdt = reshape(newdt, size(newdt)[1:end-2]..., :)
    return GraphDataset(newgs, newxs, newus, newdt)
end

function MLDataPattern.shuffleobs(data::GraphDataset; obsdim = LearnBase.default_obsdim(data), rng::AbstractRNG = Random.GLOBAL_RNG)
    idxs = 1:nobs(data)
    idxs = shuffle(rng, idxs)
    return GraphDataset(data.gs[.., idxs], data.xs[.., idxs], data.us[.., idxs], data.dt[idxs])
end


# ...dim x T x B
StatsBase.nobs(data::GraphDataset) = size(data.xs)[end]
LearnBase.getobs(data::GraphDataset, idx) = (data.gs[idx], batchidx(data.xs, idx), batchidx(data.us, idx))

"""
Convert vector of features to features corresponding to each node in a graph
"""
function nodefeatures(xs, adj_mat)
    d = size(adj_mat, 1)
    f = convert(Int, size(xs, 1) // d)
    return reshape(xs, f, d, size(xs)[2:end]...)
end

function GraphDataset(data::Dataset, adj_mat::AbstractMatrix)
    #gs = [Fill(SimpleGraph(adj_mat), size(batchidx(data.xs, i))[end]) for i in 1:StatsBase.nobs(data)]
    gs = Fill(SimpleGraph(adj_mat), size(data.xs)[end-1], StatsBase.nobs(data))
    return GraphDataset(gs, nodefeatures(data.xs, adj_mat), nodefeatures(data.us, adj_mat), data.dt)
end

#= """
Convert arrays of trajectories into a one step regression dataset
"""
function trajbatch2dataset(xs, us)
    A = slidingwindow(i->i+1, xs, 1, LearnBase.ObsDim.Constant(2))
    newxs = []
    newus = []
    newxsp = []
    for (i, (x, xp)) in enumerate(A)
        # time axis already present due to slidingwindow
        push!(newxs, x)
        # conserve time axis
        push!(newus, us[:, i, :][:, [CartesianIndex()], :])
        push!(newxsp, xp[:, [CartesianIndex()] ,:])
    end
    return Dataset(reduce(hcat, newxs), reduce(hcat, newus), reduce(hcat, newxsp))
end =#

function addnoise(data::Dataset, std::Float64=0.001; rng=Random.GLOBAL_RNG)
    Dataset(data.xs .+ eltype(data).(std .* randn(rng, size(data.xs))), data.us, data.dt)
end
function addnoise(data::GraphDataset, std::Float64=0.001; rng=Random.GLOBAL_RNG)
    GraphDataset(data.gs, data.xs .+ eltype(data).(std .* randn(rng, size(data.xs))), data.us, data.dt)
end

function MLDataPattern.splitobs(data::Dataset, at::AbstractFloat, obsdim=LearnBase.default_obsdim(data))
    train, test = MLDataPattern.splitobs((data.xs, data.us, data.dt), at, obsdim)
    return Dataset(train...), Dataset(test...)
end

function MLDataPattern.splitobs(data::GraphDataset, at::AbstractFloat, obsdim=LearnBase.default_obsdim(data))
    train, test = MLDataPattern.splitobs((data.gs, data.xs, data.us, data.dt), at, obsdim)
    return GraphDataset(train...), GraphDataset(test...)
end

struct FactorGraphDataset{NG,FG,H,I,T} <: AbstractDataset
    n2fgs::NG
    f2ngs::FG
    xs::H
    us::I
    dt::T
end

Base.eltype(d::FactorGraphDataset) = eltype(d.xs)

# ...dim x T x B
StatsBase.nobs(data::FactorGraphDataset) = size(data.xs)[end]
LearnBase.getobs(data::FactorGraphDataset, idx) = (data.gs[idx], batchidx(data.xs, idx), batchidx(data.us, idx))

function FactorGraphDataset(data::Dataset, n2f_adjmat::AbstractMatrix, f2n_adjmat::AbstractMatrix, num_components; node_xdim, node_udim)
    n2fgs = Fill(SimpleDiGraph(n2f_adjmat), size(data.xs)[end-1], StatsBase.nobs(data))
    f2ngs = Fill(SimpleDiGraph(f2n_adjmat), size(data.xs)[end-1], StatsBase.nobs(data))
    xs = reshape(data.xs, node_xdim, num_components, size(data.xs)[2:end]...)
    us = reshape(data.us, node_udim, num_components, size(data.xs)[2:end]...)
    return FactorGraphDataset(n2fgs, f2ngs, xs, us, data.dt)
end

function stack(ds::Vector{<:FactorGraphDataset})
    newn2fgs = stack([d.n2fgs for d in ds])
    newf2ngs = stack([d.f2ngs for d in ds])
    newxs = stack([d.xs for d in ds])
    newus = stack([d.us for d in ds])
    newdt = stack([d.dt for d in ds])
    newn2fgs = reshape(newn2fgs, size(newn2fgs)[1:end-2]..., :)
    newf2ngs = reshape(newf2ngs, size(newf2ngs)[1:end-2]..., :)
    newxs = reshape(newxs, size(newxs)[1:end-2]..., :)
    newus = reshape(newus, size(newus)[1:end-2]..., prod(size(newus)[end-1:end]))
    newdt = reshape(newdt, size(newdt)[1:end-2]..., :)
    return FactorGraphDataset(newn2fgs, newf2ngs, newxs, newus, newdt)
end

function MLDataPattern.shuffleobs(data::FactorGraphDataset; obsdim = LearnBase.default_obsdim(data), rng::AbstractRNG = Random.GLOBAL_RNG)
    idxs = 1:nobs(data)
    idxs = shuffle(rng, idxs)
    return FactorGraphDataset(data.n2fgs[.., idxs], data.f2ngs[.., idxs], data.xs[.., idxs], data.us[.., idxs], data.dt[idxs])
end


function addnoise(data::FactorGraphDataset, std::Float64=0.001; rng=Random.GLOBAL_RNG)
    FactorGraphDataset(data.n2fgs, data.f2ngs, data.xs .+ std .* randn(rng, size(data.xs)), data.us, data.dt)
end

function MLDataPattern.splitobs(data::FactorGraphDataset, at::AbstractFloat, obsdim=LearnBase.default_obsdim(data))
    train, test = MLDataPattern.splitobs((data.n2fgs, data.f2ngs, data.xs, data.us, data.dt), at, obsdim)
    return FactorGraphDataset(train...), FactorGraphDataset(test...)
end

"""
Generate batch of trajectories given initial conditions of a system
"""
function generate_trajectories(system, initial_conditions, control_inputs; trajlen=50, dt=0.05)

    p0, re = Flux.destructure(system)
    trajs = zeros(size(initial_conditions[1])..., trajlen+1, length(initial_conditions))
    for i in 1:length(initial_conditions)
        function dudt(u, p, t)
            tidx = Int(round(t/dt + 1))
            dynamics(re(p), state(system, u), control_inputs[i][:, tidx])
        end
        ff = ODEFunction{false}(dudt)
        prob = ODEProblem{false}(ff, initial_conditions[i], (0.0, dt*trajlen), p0)
        traj = solve(prob, Tsit5(), saveat=dt, reltol=1e-4)
        #push!(trajs, Array(traj))
        trajs[:, : , i] .= Array(traj)
    end
    return trajs
    #return reduce((xs,x)->cat(xs,x; dims=3), trajs) #cat(trajs..., dims=3)
end

function generate_initial_conditions(system, numtraj; sampler=:randn, rng=Random.GLOBAL_RNG)
    if length(state_range(system)) == 0 && (sampler==:grid || sampler ==:rand_constrained)
        @warn("System does not have predefined state range to select values from. Falling back to random values")
        sampler=:randn
    end
    if sampler == :grid
        initial_conditions = reshape(collect(Iterators.product([collect(x[1] : (x[2]-x[1])/numtraj : x[2]) for x in state_range(system)]...)), (:,state_dim(system)))
    elseif sampler == :randn
        initial_conditions = [randn(rng, state_dim(system)) for _ in 1:numtraj]
    elseif sampler == :rand_constrained
        initial_conditions = [[rand(rng, Uniform(r...)) for r in state_range(system)] for _ in 1:numtraj]
    elseif sampler == :custom
        initial_conditions = [custom_init_state(system; rng=rng) for _ in 1:numtraj]
    else
        error("$sampler type not defined")
    end
    return initial_conditions
end

function _sine_offsetsignal(t_points, params)
    offset, args = params
    sines = [arg.amplitude .* sin.(arg.freq .* t_points .+ arg.phase) for arg in args]
    u_T = sum(sines) .+ offset
end

function _rand_sinu(rng, sys, trajlen, dt)
    logw_ub = log(6.0)
    logw_lb = log(1)
    logw_range = logw_ub - logw_lb
    offset = rand(rng, 1, control_dim(sys)) .- 0.5
    amplitudes = [rand(rng, 1, control_dim(sys)) .* 0.75 for _ in 1:3]
    phases = [rand(rng, 1, control_dim(sys)) .* 2pi .- pi for _ in 1:3]
    freqs = [exp.(rand(rng, 1, control_dim(sys)) .* logw_range .+ logw_lb) for _ in 1:3]
    args = [(amplitude=amplitudes[i], phase=phases[i], freq=freqs[i]) for i in 1:3]
    t_points = repeat(collect(0.0:dt:(trajlen-1)*dt), inner=(1, control_dim(sys)))
    permutedims(_sine_offsetsignal(t_points, (offset, args)), (2, 1))
end

function generate_control_inputs(system, trajlen, numtraj; sampler=:randn, rng=Random.GLOBAL_RNG, randn_scale=1.0)
    if length(control_range(system)) == 0 && (sampler==:grid || sampler ==:rand_constrained || sampler ==:rand_constant_constrained)
        @warn("System does not have predefined control range to select values from. Falling back to random values")
        sampler=:randn
    end

    if sampler == :grid
        controls = reshape(collect(Iterators.product([collect(u[1] : (u[2]-u[1])/numtraj : u[2]) for u in control_range(system)]...)), (:,control_dim(system)))
    elseif sampler == :randn
        controls = [randn(rng, control_dim(system), trajlen) .* randn_scale for _ in 1:numtraj]
    elseif sampler == :rand_constrained
        controls = [reduce(vcat, [rand(rng, Uniform(r...), trajlen) for r in control_range(system)]') for _ in 1:numtraj]
    elseif sampler == :rand_constant
        controls = [ones(control_dim(system), trajlen) .* randn(rng) .* randn_scale for _ in 1:numtraj]
    elseif sampler == :rand_constant_constrained
        controls = [ones(control_dim(system), trajlen) .* (rand(rng, Uniform(r...)) for r in control_range(system)) for _ in 1:numtraj]
    elseif sampler == :rand_sinusoid
        controls = [_rand_sinu(rng, system, trajlen, 0.05) for _ in 1:numtraj]
    end
    return controls
end

function generate_system_parameters(system, numtraj; sampler=:randn, rng=Random.GLOBAL_RNG)
    num_params = sysparam_dim(system)
    param_ranges = sysparam_range(system)

    if length(param_ranges) == 0 && (sampler==:grid || sampler ==:rand_constrained)
        @warn("System does not have predefined parameter range to select values from. Falling back to random values")
        sampler=:randn
    end

    if sampler == :grid
        system_parameters = reshape(collect(Iterators.product([collect(p[1] : (p[2]-p[1])/numtraj : p[2]) for p in param_ranges]...)), (:,num_params))
    elseif sampler == :randn
        system_parameters = [randn(rng, num_params) for _ in 1:numtraj]
    elseif sampler == :rand_constrained
        system_parameters = [reduce(vcat, [rand(rng, Uniform(r...), trajlen) for r in sysparam_range(system)]') for _ in 1:numtraj]
    end
    return system_parameters
end

"""
Generate data in parallel from a given system with random initial states
"""
function generate_dataset(system, numtraj; trajlen=50, dt=0.05, rng=Random.GLOBAL_RNG, state_sampler=:randn, control_sampler=:randn, sysparam_sampler=:randn)
    initial_conditions = generate_initial_conditions(system, numtraj; sampler=state_sampler)
    control_inputs = generate_control_inputs(system, trajlen+1, numtraj; sampler=control_sampler) # TODO: why +1?
    system_parameters = generate_system_parameters(system, numtraj; sampler=sysparam_sampler)

    trajs = generate_trajectories(system, initial_conditions, control_inputs; trajlen=trajlen, dt=0.05)
    xs = Array(trajs)
    # us = cat(control_inputs..., dims=3)
    us = reduce((xs, x)-> cat(xs, x; dims=3), control_inputs)
    dt_B = Fill(dt, size(xs)[end])
    return Dataset(Float32.(state(system, xs)), Float32.(us), Float32.(dt_B))
end

function generate_graph_dataset(system, numtraj; trajlen=50, dt=0.05, rng=Random.GLOBAL_RNG, state_sampler=:randn, control_sampler=:randn, sysparam_sampler=:randn, graph="sys")
    ds = generate_dataset(system, numtraj; trajlen=trajlen, dt=dt, rng=rng, state_sampler=state_sampler, control_sampler=control_sampler, sysparam_sampler=sysparam_sampler)
    if graph == "full"
        A = ones(Int, num_components(system), num_components(system))
    elseif graph == "hollow"
        A = ones(Int, num_components(system), num_components(system)) - diagm(ones(Int, num_components(system)))
    elseif graph == "diag"
        A = diagm(ones(Int, num_components(system)))        
    elseif graph == "sys"
        A = adj_mat(system)
    end
    return GraphDataset(ds, A)
end

function generate_factorgraph_dataset(system, numtraj; numfactors=1, trajlen=50, dt=0.05, rng=Random.GLOBAL_RNG, state_sampler=:randn, control_sampler=:randn, sysparam_sampler=:randn, graph="full")
    ds = generate_dataset(system, numtraj; trajlen=trajlen, dt=dt, rng=rng, state_sampler=state_sampler, control_sampler=control_sampler, sysparam_sampler=sysparam_sampler)
    if occursin("full", graph)
        n2f_adj_mat = zeros(Int, num_components(system)+numfactors, num_components(system)+numfactors)
        n2f_adj_mat[end, 1:end-1] .= 1
        f2n_adj_mat = zeros(Int, num_components(system)+numfactors, num_components(system)+numfactors)
        f2n_adj_mat[1:end-1, end] .= 1
    end
    ins = Int(state_dim(system) // num_components(system))
    inu = Int(control_dim(system) // num_components(system))
    return FactorGraphDataset(ds, n2f_adj_mat, f2n_adj_mat, num_components(system); node_xdim=ins, node_udim=inu)
end

function generate_dataset(systems::AbstractVector{<:AbstractModel}, numtraj; trajlen=50, dt=0.05, rng=Random.GLOBAL_RNG, state_sampler=:randn, control_sampler=:randn, sysparam_sampler=:randn)
    nr_per_sys = convert(Int, ceil(numtraj//length(systems)))
    return stack([generate_dataset(sys, nr_per_sys; trajlen=trajlen, dt=dt, rng=rng, state_sampler=state_sampler, control_sampler=control_sampler, sysparam_sampler=sysparam_sampler) for sys in systems])
end

function generate_graph_dataset(systems::AbstractVector{<:AbstractModel}, numtraj; trajlen=50, dt=0.05, rng=Random.GLOBAL_RNG, state_sampler=:randn, control_sampler=:randn, sysparam_sampler=:randn, graph="sys")
    nr_per_sys = convert(Int, ceil(numtraj//length(systems)))
    return stack([generate_graph_dataset(sys, nr_per_sys; trajlen=trajlen, dt=dt, rng=rng, state_sampler=state_sampler, control_sampler=control_sampler, sysparam_sampler=sysparam_sampler, graph=graph) for sys in systems])
end

function generate_factorgraph_dataset(systems::AbstractVector{<:AbstractModel}, numtraj; numfactors=1, trajlen=50, dt=0.05, rng=Random.GLOBAL_RNG, state_sampler=:randn, control_sampler=:randn, sysparam_sampler=:randn, graph="full")
    nr_per_sys = convert(Int, ceil(numtraj//length(systems)))
    return stack([generate_factorgraph_dataset(sys, nr_per_sys; numfactors=numfactors, trajlen=trajlen, dt=dt, rng=rng, state_sampler=state_sampler, control_sampler=control_sampler, sysparam_sampler=sysparam_sampler, graph=graph) for sys in systems])
end

