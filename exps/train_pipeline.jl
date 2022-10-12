using Random
using LinearAlgebra
using Flux
using Zygote
using OrdinaryDiffEq
using DiffEqFlux
using LightGraphs
using ImageIO
using Configurations
using Statistics
using StatsBase
using Logging
using LoggingExtras
using Plots
using TensorBoardLogger
using JLD2
using LearnBase
using OrderedCollections
using MLDataPattern
using ParameterSchedulers
using EarlyStopping
using MPNODE
using MPNODE: default_lorenz_system, diagonal_lorenz_system, coupled_pendulum, state_dim, control_dim, num_components, DataPathSystem
using MPNODE: rollout, AbstractModel
using MPNODE: generate_dataset, generate_graph_dataset, generate_factorgraph_dataset, addnoise, Dataset, GraphDataset, FactorGraphDataset, splitobs
using MPNODE: ztransform, simpletransform
using MPNODE: collocateloss, onesteploss, single_shooting_loss, multiple_shooting_loss, time_series_norm

include("tasks.jl")
include("models.jl")
include("evaluate.jl")

function rec_flatten_dict(d, prefix_delim=".")
    new_d = empty(d)
    for (key, value) in pairs(d)
        if isa(value, Dict) || isa(value, OrderedCollections.OrderedDict)
            flattened_value = rec_flatten_dict(value, prefix_delim)
            for (ikey, ivalue) in pairs(flattened_value)
                new_d["$key.$ikey"] = ivalue
            end
        else
            new_d[key] = value
        end
    end
    return new_d
end

function addcontext(logger, context)
    TransformerLogger(logger) do log
        merge(log, (; message="$context/$(log.message)"))
    end
end

mutable struct NoOpSchedule
    start
end

(schedule::NoOpSchedule)(t) = schedule.start
# Base.eltype(::Type{<:NoOpSchedule{T}}) where T = T
Base.IteratorSize(::Type{<:NoOpSchedule}) = Base.IsInfinite()
Base.iterate(schedule::NoOpSchedule, t=1) = schedule(t), t + 1
Base.axes(::NoOpSchedule) = (ParameterSchedulers.OneToInf(),)


# XXX hack to allow vector of similar sys to still works

function MPNODE.num_components(sys::AbstractVector{<:AbstractModel})
    return MPNODE.num_components(sys[1])
end

function MPNODE.state_dim(sys::AbstractVector{<:AbstractModel})
    return MPNODE.state_dim(sys[1])
end

function MPNODE.control_dim(sys::AbstractVector{<:AbstractModel})
    return MPNODE.control_dim(sys[1])
end

function reconstruct_loss(datatransform, model, g_T_B, x_D_G_T_B, u_D_G_T_B, dt, alg; normalize="ztransform", lossfn=time_series_huber, kwargs...)
    x_D_T_B = MPNODE.graph2batch(x_D_G_T_B)
    u_D_T_B = MPNODE.graph2batch(u_D_G_T_B)
    x_D_0_B = MPNODE.timeidx(x_D_T_B, 1)

    # Have to ignore otherwise Zygote runs into issues trying to differentiate
    g_T = Zygote.@ignore [reduce(blockdiag, g_T_B[i, :]) for i in 1:size(g_T_B, 1)]
    xpred_D_T_B = MPNODE.rollout(model, g_T, x_D_0_B, u_D_T_B, dt[1], alg; kwargs...)

    if normalize == "ztransform"
        xpred_D_T_B_re = reshape(StatsBase.reconstruct(datatransform.Xtr, reshape(xpred_D_T_B, size(xpred_D_T_B, 1), :)), size(xpred_D_T_B)...)
        x_D_T_B_re = reshape(StatsBase.reconstruct(datatransform.Xtr, reshape(x_D_T_B, size(x_D_T_B, 1), :)), size(x_D_T_B)...)
    elseif normalize == "simple"
        G = size(x_D_G_T_B, 2)
        mu, sigma = datatransform.Xtr
        xpred_D_T_G_B = reshape(xpred_D_T_B, size(xpred_D_T_B)[1:end-1]..., G, :)
        xpred_D_G_T_B = permutedims(xpred_D_T_G_B, (1, 3, 2, 4))
        xpred_G_DTB_re = mu .+ (reshape(xpred_D_G_T_B, G, :) .* sigma)
        xpred_D_T_G_B_re = reshape(xpred_G_DTB_re, size(xpred_D_T_G_B)...)
        xpred_D_T_B_re = reshape(xpred_D_T_G_B_re, size(xpred_D_T_G_B)[1:2]..., :)

        x_D_T_G_B = reshape(x_D_T_B, size(x_D_T_B)[1:end-1]..., G, :)
        x_D_G_T_B = permutedims(x_D_T_G_B, (1, 3, 2, 4))
        x_G_DTB_re = mu .+ (reshape(x_D_G_T_B, G, :) .* sigma)
        x_D_T_G_B_re = reshape(x_G_DTB_re, size(x_D_T_G_B)...)

        x_D_T_B_re = reshape(x_D_T_G_B_re, size(x_D_T_G_B)[1:2]..., :)
    end
    return lossfn(x_D_T_B_re[:, 2:end, :], xpred_D_T_B_re[:, 2:end, :])
end

function run_experiment(taskname, trainconfig, evalconfig, resumeckptpath=nothing, expname=nothing)
    if isnothing(expname) || expname == ""
        expname = taskname
    end
    Random.seed!(trainconfig.seed)
    
    dataconfig = trainconfig.data
    # If path to the data directory is given, we don't need to generate data
    if ispath(dataconfig.path)
        sys = DataPathSystem(dataconfig.path)
    else
        sys = TASK_REGISTRY[taskname]
    end

    if typeof(sys) <: DataPathSystem
        traindata, validdata, testdata = generate_dataset(sys)
        if dataconfig.horizon < size(traindata.xs)[end-1]
            @info "Truncating data to $(dataconfig.horizon) timesteps"
            traindata = MPNODE.truncate_timesteps(traindata, dataconfig.horizon)
            validdata = MPNODE.truncate_timesteps(validdata, dataconfig.horizon)
            testdata = MPNODE.truncate_timesteps(testdata, dataconfig.horizon)
        end
        if dataconfig.datasize < size(traindata.xs)[end]
            println("--------------------------------------------")
            @info "$(dataconfig.datasize) is less than data on disk. Truncating..."
            traindata = MPNODE.subset(traindata, 1:dataconfig.datasize)
        end
        if dataconfig.datasize * (1. - dataconfig.split) < size(validdata.xs)[end]
            vsi = Int(ceil(dataconfig.datasize * (1. - dataconfig.split)))
            validdata = MPNODE.subset(validdata, 1:vsi)
        end
    elseif first(trainconfig.model.name) == 'g' || startswith(trainconfig.model.name, "emp")
        dataset = generate_graph_dataset(sys, dataconfig.datasize;
                                        rng=MersenneTwister(trainconfig.seed + 42),
                                        dt=dataconfig.dt, trajlen=dataconfig.horizon, 
                                        state_sampler=Symbol(dataconfig.statesampler), control_sampler=Symbol(dataconfig.controlsampler),
                                        graph=trainconfig.model.graph)
    elseif startswith(trainconfig.model.name, "fgmp")
        dataset = generate_factorgraph_dataset(
            sys, dataconfig.datasize;
            rng=MersenneTwister(trainconfig.seed + 42),
            numfactors=trainconfig.model.numfactors,
            dt=dataconfig.dt, trajlen=dataconfig.horizon, 
            state_sampler=Symbol(dataconfig.statesampler), control_sampler=Symbol(dataconfig.controlsampler),
            graph=trainconfig.model.graph            
        )
    else
        dataset = generate_dataset(sys, dataconfig.datasize; rng=MersenneTwister(trainconfig.seed + 42),
                                dt=dataconfig.dt, trajlen=dataconfig.horizon, 
                                state_sampler=Symbol(dataconfig.statesampler), control_sampler=Symbol(dataconfig.controlsampler))
    end

    transferconfig = evalconfig.transfer
    if transferconfig.enabled
        @info "Generating datasets for zero-shot transfer evaluation..."       
        transfer_systems = Dict(taskname => TASK_REGISTRY[taskname] for taskname in transferconfig.systems)
        if first(trainconfig.model.name) == 'g' || startswith(trainconfig.model.name, "emp")
            transfer_datasets = Dict(taskname => generate_graph_dataset(transfer_systems[taskname], transferconfig.datasize; rng=MersenneTwister(trainconfig.seed + 42),
                                                dt=transferconfig.dt, trajlen=transferconfig.horizon, state_sampler=Symbol(transferconfig.statesampler), control_sampler=Symbol(transferconfig.controlsampler), graph=trainconfig.model.graph)
                        for taskname in transferconfig.systems)
        elseif startswith(trainconfig.model.name, "fgmp")
            transfer_datasets = Dict(taskname => generate_factorgraph_dataset(transfer_systems[taskname], transferconfig.datasize; 
                                                numfactors=trainconfig.model.numfactors, rng=MersenneTwister(trainconfig.seed + 1331),
                                                dt=transferconfig.dt, trajlen=transferconfig.horizon, state_sampler=Symbol(transferconfig.statesampler), control_sampler=Symbol(transferconfig.controlsampler), graph=trainconfig.model.graph)
                        for taskname in transferconfig.systems)
        else
            transfer_datasets = Dict(taskname => generate_dataset(transfer_systems[taskname], transferconfig.datasize; 
                                                dt=transferconfig.dt, rng=MersenneTwister(trainconfig.seed + 1331), trajlen=transferconfig.horizon, state_sampler=Symbol(transferconfig.statesampler), control_sampler=Symbol(transferconfig.controlsampler))
                        for taskname in transferconfig.systems)
        end
    end

    if dataconfig.normalize == "ztransform"
        @info "Normalizing datasets"
        if typeof(sys) <: DataPathSystem
            traindata, datatransform = ztransform(traindata)
            validdata = ztransform(datatransform, validdata)
            testdata = ztransform(datatransform, testdata)
        else
            dataset, datatransform = ztransform(dataset)
        end

        if transferconfig.enabled
            transfer_datasets = Dict(name => ztransform(datatransform, ds) for (name, ds) in transfer_datasets)
        end
    elseif dataconfig.normalize == "simple"
        @info "Normalizing datasets"
        if typeof(sys) <: DataPathSystem
            traindata, datatransform = simpletransform(traindata)
            validdata = simpletransform(datatransform, validdata)
            testdata = simpletransform(datatransform, testdata)
        else
            dataset, datatransform = simpletransform(dataset)
        end

        if transferconfig.enabled
            transfer_datasets = Dict(name => simpletransform(ds)[1] for (name, ds) in transfer_datasets) # can't use the one from trained
        end
    end
    
    # Setting up logging. At a minimum we want to log to stdout as well as TensorBoard.
    loggers = [current_logger(), TBLogger(joinpath(trainconfig.save.dir, "logs"))]

    # Collect all loggers together into a single one
    explogger = TeeLogger(loggers...)

    # We split generated data. Existing data is already assumed to have been split
    if !(typeof(sys) <: DataPathSystem)
        traindata, validdata = splitobs(shuffleobs(dataset; rng=MersenneTwister(trainconfig.seed + 1)), dataconfig.split, LearnBase.ObsDim.Last())
    end

    # Add noise to the training data
    if dataconfig.noisestd > 0.0
        traindata = addnoise(traindata, dataconfig.noisestd)
    end

    # Get the model function (assuming that all are defined in the Models module)
    model_fn = getfield(Models, Symbol(trainconfig.model.name))
    if first(trainconfig.model.name) == 'n'
        # Standard Neural ODE
        model = model_fn(state_dim(sys), control_dim(sys), trainconfig.model.hiddendim)
    elseif startswith(trainconfig.model.name, "augn")
        # Augmented Neural ODE
        model = model_fn(state_dim(sys), control_dim(sys), trainconfig.model.hiddendim, trainconfig.model.augdim)
    elseif first(trainconfig.model.name) == 'g'
        # Graph Neural ODE
        ins = Int(state_dim(sys) // num_components(sys))
        inu = Int(control_dim(sys) // num_components(sys))
        model = model_fn(ins, inu, trainconfig.model.hiddendim)
        adj_mat = randn(num_components(sys), num_components(sys)) # assuming size of graph will be same 
    elseif startswith(trainconfig.model.name, "fgmp")
        # Factor Graph Neural ODE
        ins = Int(state_dim(sys) // num_components(sys))
        inu = Int(control_dim(sys) // num_components(sys))
        model = model_fn(ins, inu, trainconfig.model.hiddendim, trainconfig.model.numfactors, num_components(sys))
        adj_mat = randn(num_components(sys), num_components(sys)) # assuming size of graph will be same
    elseif startswith(trainconfig.model.name, "emp")
        # Explicit message passing Neural ODE
        ins = Int(state_dim(sys) // num_components(sys))
        inu = Int(control_dim(sys) // num_components(sys))
        model = model_fn(ins, inu, trainconfig.model.messagedim, trainconfig.model.hiddendim, trainconfig.model.embeddim)
        adj_mat = randn(num_components(sys), num_components(sys)) # assuming size of graph will be same
    else
        error("Model type `$(trainconfig.model.name)` not defined")
    end

    # load from ckpt_path
    #
    if !isnothing(resumeckptpath)
        @info "Loading checkpoint from $resumeckptpath"
        JLD2.@load resumeckptpath weights
        Flux.loadparams!(model, weights)
    end

    integratorconfig = trainconfig.integrator
    integrator = getfield(OrdinaryDiffEq, Symbol(integratorconfig.integrator))()

    # We allow training with different loss functions
    # time_series_* versions sum over the time axis rather than average over it
    if trainconfig.shooting.difffn == "time_series_huber"
        difffn = MPNODE.time_series_huber
    elseif trainconfig.shooting.difffn == "time_series_mse"
        difffn = MPNODE.time_series_mse
    elseif trainconfig.shooting.difffn == "mse"
        difffn = Flux.Losses.mse
    elseif trainconfig.shooting.difffn == "huber"
        difffn = Flux.Losses.huber_loss
    end

    numparams = MPNODE.num_parameters(model)
    @info "Training $(trainconfig.model.name) with #parameters: $(numparams)"
    # Set up plotting trajectories and other validation metrics
    if typeof(traindata) <: GraphDataset
        train_plotdata_true_gs = MPNODE.batchidx(traindata.gs, evalconfig.trajplot.trainidxs)
        train_plotdata_true_xs = MPNODE.batchidx(traindata.xs, evalconfig.trajplot.trainidxs)
        train_plotdata_true_us = MPNODE.batchidx(traindata.us, evalconfig.trajplot.trainidxs)
        train_dt = traindata.dt[1]
        train_T = size(traindata.xs)[end - 1] - 2

        valid_plotdata_true_gs = MPNODE.batchidx(validdata.gs, evalconfig.trajplot.valididxs)
        valid_plotdata_true_xs = MPNODE.batchidx(validdata.xs, evalconfig.trajplot.valididxs)
        valid_plotdata_true_us = MPNODE.batchidx(validdata.us, evalconfig.trajplot.valididxs)
        valid_dt = validdata.dt[1]
        valid_T = size(validdata.xs)[end - 1] - 2

        function plottrajsg(true_gs, true_xs, true_us, dt, T, adj_mat)
            true_trajs, pred_trajs = single_shooting_loss(model, true_gs, true_xs, true_us, dt, integrator; zero_msg=trainconfig.model.zeromsg, lossfn=(a, b) -> (MPNODE.batch2graph(a, adj_mat), MPNODE.batch2graph(b, adj_mat)))
            true_trajs = reshape(true_trajs, prod(size(true_trajs)[1:2]), size(true_trajs)[3:end]...)
            pred_trajs = reshape(pred_trajs, prod(size(pred_trajs)[1:2]), size(pred_trajs)[3:end]...)
            ps = [compare_trajectories(dt;figsize=(100 * size(true_trajs, 1), 80 * size(true_trajs, 1)), sys=MPNODE.batchidx(true_trajs, i), pred=MPNODE.batchidx(pred_trajs, i)) for i in 1:size(pred_trajs)[end]]
            return ps
        end
        plot_callbacks = Dict("train" => () -> plottrajsg(train_plotdata_true_gs, train_plotdata_true_xs, train_plotdata_true_us, train_dt, train_T, adj_mat),
                              "valid" => () -> plottrajsg(valid_plotdata_true_gs, valid_plotdata_true_xs, valid_plotdata_true_us, valid_dt, valid_T, adj_mat))
        plot_callbacks["valid/fullshooting"] = () -> single_shooting_loss(model, validdata.gs, validdata.xs, validdata.us, validdata.dt[1], integrator;
                                                                        zero_msg=trainconfig.model.zeromsg, lossfn=difffn, abstol=integratorconfig.abstol, reltol=integratorconfig.reltol)

        if dataconfig.normalize == "ztransform" || dataconfig.normalize == "simple"
            plot_callbacks["valid/reconstruct_full"] = () -> reconstruct_loss(datatransform, model, validdata.gs, validdata.xs, validdata.us, validdata.dt[1], integrator; normalize=dataconfig.normalize,
                                                                        zero_msg=trainconfig.model.zeromsg, lossfn=difffn, abstol=integratorconfig.abstol, reltol=integratorconfig.reltol)
        end
        if transferconfig.enabled
            for (name, ds) in transfer_datasets
                plot_callbacks["transfer/$name/fullshooting"] = () -> single_shooting_loss(model, ds.gs, ds.xs, ds.us, ds.dt[1], integrator; zero_msg=trainconfig.model.zeromsg, lossfn=difffn, abstol=integratorconfig.abstol, reltol=integratorconfig.reltol)
                transfer_plotdata_true_gs = MPNODE.batchidx(ds.gs, evalconfig.trajplot.valididxs)
                transfer_plotdata_true_xs = MPNODE.batchidx(ds.xs, evalconfig.trajplot.valididxs)
                transfer_plotdata_true_us = MPNODE.batchidx(ds.us, evalconfig.trajplot.valididxs)
                tr_nc = MPNODE.num_components(TASK_REGISTRY[name])
                tr_adj_mat = randn(tr_nc, tr_nc)
                plot_callbacks["transfer/$name/plot"] = () -> (@show name; plottrajsg(transfer_plotdata_true_gs, transfer_plotdata_true_xs, transfer_plotdata_true_us, ds.dt[1], size(ds.xs)[end - 1] - 2, tr_adj_mat))
            end
        end
    elseif typeof(traindata) <: FactorGraphDataset
        train_plotdata_true_n2fgs = MPNODE.batchidx(traindata.n2fgs, evalconfig.trajplot.trainidxs)
        train_plotdata_true_f2ngs = MPNODE.batchidx(traindata.f2ngs, evalconfig.trajplot.trainidxs)

        train_plotdata_true_xs = MPNODE.batchidx(traindata.xs, evalconfig.trajplot.trainidxs)
        train_plotdata_true_us = MPNODE.batchidx(traindata.us, evalconfig.trajplot.trainidxs)
        train_dt = traindata.dt[1]
        train_T = size(traindata.xs)[end - 1] - 2

        valid_plotdata_true_n2fgs = MPNODE.batchidx(validdata.n2fgs, evalconfig.trajplot.valididxs)
        valid_plotdata_true_f2ngs = MPNODE.batchidx(validdata.f2ngs, evalconfig.trajplot.valididxs)

        valid_plotdata_true_xs = MPNODE.batchidx(validdata.xs, evalconfig.trajplot.valididxs)
        valid_plotdata_true_us = MPNODE.batchidx(validdata.us, evalconfig.trajplot.valididxs)
        valid_dt = validdata.dt[1]
        valid_T = size(validdata.xs)[end - 1] - 2

        function plottrajsfg(true_n2fgs, true_f2ngs, true_xs, true_us, dt, T)
            true_trajs, pred_trajs = single_shooting_loss(model, true_n2fgs, true_f2ngs, true_xs, true_us, dt, integrator; zero_msg=trainconfig.model.zeromsg, lossfn=(a, b) -> (MPNODE.batch2graph(a, adj_mat), MPNODE.batch2graph(b, adj_mat)))
            true_trajs = reshape(true_trajs, prod(size(true_trajs)[1:2]), size(true_trajs)[3:end]...)
            pred_trajs = reshape(pred_trajs, prod(size(pred_trajs)[1:2]), size(pred_trajs)[3:end]...)
            ps = [compare_trajectories(dt;figsize=(100 * size(true_trajs, 1), 80 * size(true_trajs, 1)), sys=MPNODE.batchidx(true_trajs, i), pred=MPNODE.batchidx(pred_trajs, i)) for i in 1:size(pred_trajs)[end]]
            return ps
        end
        plot_callbacks = Dict("train" => () -> plottrajsfg(train_plotdata_true_n2fgs, train_plotdata_true_f2ngs, train_plotdata_true_xs, train_plotdata_true_us, train_dt, train_T),
                              "valid" => () -> plottrajsfg(valid_plotdata_true_n2fgs, valid_plotdata_true_f2ngs, valid_plotdata_true_xs, valid_plotdata_true_us, valid_dt, valid_T))
        plot_callbacks["valid/fullshooting"] = () -> single_shooting_loss(model, validdata.n2fgs, validdata.f2ngs, validdata.xs, validdata.us, validdata.dt, integrator; zero_msg=trainconfig.model.zeromsg, lossfn=difffn, abstol=integratorconfig.abstol, reltol=integratorconfig.reltol)
        
        if transferconfig.enabled
            for (name, ds) in transfer_datasets
                plot_callbacks["transfer/$name/fullshooting"] = () -> single_shooting_loss(model, ds.n2fgs, ds.f2ngs, ds.xs, ds.us, ds.dt, integrator; zero_msg=trainconfig.model.zeromsg, lossfn=difffn, abstol=integratorconfig.abstol, reltol=integratorconfig.reltol)
            end
        end
    elseif typeof(traindata) <: Dataset
        train_plotdata_true_xs = MPNODE.batchidx(traindata.xs, evalconfig.trajplot.trainidxs)
        train_plotdata_true_us = MPNODE.batchidx(traindata.us, evalconfig.trajplot.trainidxs)
        train_dt = traindata.dt[1]
        train_T = size(traindata.xs)[end - 1] - 2

        valid_plotdata_true_xs = MPNODE.batchidx(validdata.xs, evalconfig.trajplot.valididxs)
        valid_plotdata_true_us = MPNODE.batchidx(validdata.us, evalconfig.trajplot.valididxs)
        valid_dt = validdata.dt[1]
        valid_T = size(validdata.xs)[end - 1] - 2

        function plottrajs(true_xs, true_us, dt, T)
            true_trajs, pred_trajs = single_shooting_loss(model, true_xs, true_us, dt, integrator; lossfn=(a, b) -> (a, b))
            ps = [compare_trajectories(dt;sys=MPNODE.batchidx(true_trajs, i), pred=MPNODE.batchidx(pred_trajs, i), figsize=(100 * size(true_trajs, 1), 80 * size(true_trajs, 1))) for i in 1:size(pred_trajs)[end]]
            return ps
        end
        
        plot_callbacks = Dict("train" => () -> plottrajs(train_plotdata_true_xs, train_plotdata_true_us, train_dt, train_T),
                              "valid" => () -> plottrajs(valid_plotdata_true_xs, valid_plotdata_true_us, valid_dt, valid_T))
        
        plot_callbacks["valid/fullshooting"] = () -> single_shooting_loss(model, validdata.xs, validdata.us, validdata.dt[1], integrator; lossfn=difffn, abstol=integratorconfig.abstol, reltol=integratorconfig.reltol)
    end

    if startswith(trainconfig.model.name, "emp")
        # EMP models can only do full shooting
        # TODO: should we still allow windowed training?
        if trainconfig.collocate.nepochs > 0
            @warn "No collocation training allowed for $(trainconfig.model.name)"
        end
        if trainconfig.singlestep.nepochs > 0
            @warn "No singlestep training allowed for $(trainconfig.model.name)"
        end
        optconfig = trainconfig.shooting.opt
        opt = getfield(Flux.Optimise, Symbol(optconfig.optimizer))(optconfig.lr)
        if optconfig.weight_decay > 0
            opt = Flux.Optimise.Optimiser(opt, WeightDecay(optconfig.weight_decay))
        end
        if optconfig.clipgrad > 0
            opt = Flux.Optimise.Optimiser(ClipNorm(optconfig.clipgrad), opt)
        end

        if trainconfig.shooting.scheduler.name == "Exp"
            scheduler = Exp(λ=optconfig.lr, γ=trainconfig.shooting.scheduler.options.decay)
        elseif trainconfig.shooting.scheduler.name == "Poly"
            scheduler = Poly(λ=optconfig.lr, p=trainconfig.shooting.scheduler.options.degree, max_iter=trainconfig.shooting.scheduler.options.max_iter)
        elseif trainconfig.shooting.scheduler.name == "Cos"
            scheduler = Cos(λ0=optconfig.lr, λ1=trainconfig.shooting.scheduler.options.range1, period=trainconfig.shooting.scheduler.options.period)
        else
            scheduler = NoOpSchedule(optconfig.lr) # no decay
        end

        stopper = EarlyStopper(Patience(trainconfig.earlystop.patience), NotANumber())
        losstype = trainconfig.shooting.loss
        if losstype == "single"
            loss = (args...) -> single_shooting_loss(model, args..., integrator; zero_msg=trainconfig.model.zeromsg, lossfn=difffn, abstol=integratorconfig.abstol, reltol=integratorconfig.reltol)
        elseif losstype == "multiple"
            error("NotYETImplemented $losstype")
            # loss = (args..., dt) -> multiple_shooting_loss(model, args..., hparams.discontinuity_weight, dt, integrator; abstol=integratorconfig.abstol, reltol=integratorconfig.reltol)
        else
            error("NotImplemented $losstype")
        end

        metrics_fns = Dict(:tsnorm => (args...) -> single_shooting_loss(model, args..., integrator; lossfn=(x, x̂) -> time_series_norm(x, x̂, dataconfig.dt), abstol=integratorconfig.abstol, reltol=integratorconfig.reltol))
        MPNODE.shootingtrain!(model, loss, opt, scheduler, stopper, traindata, validdata, metrics_fns, plot_callbacks, addcontext(explogger, "shooting"), trainconfig)
    else
        # Shooting: 
        #       Single: rollout multiple timesteps (can be set from the config) form the initial state and "backprop through time"
        # For fair comparison we set horizonwindow to full the horizon of trajectories just as MPNODE for L2S as well.
        if trainconfig.shooting.nepochs > 0
            for window in trainconfig.shooting.horizonwindow   
                windowtraindata = MPNODE.tokstep(traindata, window)
                windowvaliddata = MPNODE.tokstep(validdata, window)
                optconfig = trainconfig.shooting.opt
                opt = getfield(Flux.Optimise, Symbol(optconfig.optimizer))(optconfig.lr)
                
                if optconfig.weight_decay > 0
                    opt = Flux.Optimise.Optimiser(opt, WeightDecay(optconfig.weight_decay))
                end
                
                if optconfig.clipgrad > 0
                    opt = Flux.Optimise.Optimiser(ClipNorm(optconfig.clipgrad), opt)
                end

                if trainconfig.shooting.scheduler.name == "Exp"
                    scheduler = Exp(λ=optconfig.lr, γ=trainconfig.shooting.scheduler.options.decay)
                elseif trainconfig.shooting.scheduler.name == "Poly"
                    scheduler = Poly(λ=optconfig.lr, p=trainconfig.shooting.scheduler.options.degree, max_iter=trainconfig.shooting.scheduler.options.max_iter)
                elseif trainconfig.shooting.scheduler.name == "Cos"
                    scheduler = Cos(λ0=optconfig.lr, λ1=trainconfig.shooting.scheduler.options.range1, period=trainconfig.shooting.scheduler.options.period)
                else
                    scheduler = NoOpSchedule(optconfig.lr) # no decay
                end        
                stopper = EarlyStopper(Patience(trainconfig.earlystop.patience), NotANumber())
                losstype = trainconfig.shooting.loss
                depth = trainconfig.shooting.depth
                if losstype == "single"
                    loss = (args...) -> single_shooting_loss(model, args..., integrator; depth=depth, lossfn=difffn, abstol=integratorconfig.abstol, reltol=integratorconfig.reltol)
                elseif losstype == "multiple"
                    error("NotYETImplemented $losstype")
                    # loss = (args..., dt) -> multiple_shooting_loss(model, args..., hparams.discontinuity_weight, dt, integrator; abstol=integratorconfig.abstol, reltol=integratorconfig.reltol)
                else
                    error("NotImplemented $losstype")
                end
                metrics_fns = Dict(:tsnorm => (args...) -> single_shooting_loss(model, args..., integrator; lossfn=(x, x̂) -> time_series_norm(x, x̂, dataconfig.dt), abstol=integratorconfig.abstol, reltol=integratorconfig.reltol))
                MPNODE.shootingtrain!(model, loss, opt, scheduler, stopper, windowtraindata, windowvaliddata, metrics_fns, plot_callbacks, addcontext(explogger, "shooting_$window"), trainconfig)
            end
        end
    end
    @info "Done!"
    return model
end