using Comonicon

include("config.jl")
include("train_pipeline.jl")

"""
Train DSNs
"""
@main function train(task::String, expname::String=""; conf=nothing, evalconf=nothing, savedir=nothing, datapath=nothing)
    config = Config.read(conf)
    evalconfig = TrainEvalConfig.read(evalconf)
    if !isnothing(savedir)
        mkpath(savedir)
        config.save.dir = savedir
    end
    if !isnothing(datapath)
        config.data.path = datapath
    end
    @show config
    run_experiment(task, config, evalconfig, nothing, expname)
end