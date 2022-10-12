module Config

using Configurations
using OrderedCollections

import TOML

function default_save_dir()
    haskey(ENV, "PT_OUTPUT_DIR") && return ENV["PT_OUTPUT_DIR"]
    dname = "/tmp/mpdiffeq"
    if !ispath(dname)
        mkpath(dname)
    end
    return dname
end

@option "none" mutable struct NoneConfig
    opt::Any
end

@option "exp" mutable struct ExpConfig
    decay::Float64
end

@option "poly" mutable struct PolyConfig
    degree::Int
    max_iter::Int
end

@option "cos" mutable struct CosConfig
   range1::Float64
   period::Int 
end

@option mutable struct SchedulerConfig
    name::String
    options::Union{ExpConfig,CosConfig,PolyConfig,NoneConfig}
end

@option mutable struct OptimizerConfig
    optimizer::String
    lr::Float64
    weight_decay::Float64 = 0.0
    clipgrad::Float64 = 0.0
end

@option mutable struct DataConfig
    datasize::Int
    horizon::Int
    dt::Float64
    split::Float64 = 0.7
    noisestd::Float64 = 0.0001
    statesampler::String = "rand_constrained"
    controlsampler::String = "rand_sinusoid"
    normalize::String = "ztransform"
    path::String  = ""
end

@option mutable struct IntegratorConfig
    integrator::String = "Tsit5"
    abstol::Float64 = 1e-4
    reltol::Float64 = 1e-4
end

@option mutable struct SaveConfig
    dir::String = default_save_dir()
    freq::Int = 25
end

@option mutable struct EarlyStopping
    patience::Int = 500
    waited::Int = 20000
end

@option mutable struct CollocateConfig
    nepochs::Int = 10
    batchsize::Int = 256
    opt::OptimizerConfig = OptimizerConfig(; optimizer="ADAM", lr=0.01, weight_decay=0.0)
    scheduler::SchedulerConfig = SchedulerConfig(; name="nothing", options=NoneConfig("nothing"))
    loadbestatend::Bool = true
end

@option mutable struct SingleStepConfig
    nepochs::Int = 10
    batchsize::Int = 256
    opt::OptimizerConfig = OptimizerConfig(; optimizer="ADAM", lr=0.01, weight_decay=0.0)
    scheduler::SchedulerConfig = SchedulerConfig(; name="nothing", options=NoneConfig("nothing"))
    loadbestatend::Bool = true
end

@option mutable struct ShootingConfig
    loss::String = "single"
    difffn::String = "time_series_huber"
    horizonwindow::Vector{Int} = [5, 10]
    depth::Int = 1
    nepochs::Int = 100
    batchsize::Int = 128
    opt::OptimizerConfig = OptimizerConfig(; optimizer="ADAM", lr=0.01, weight_decay=0.0)
    scheduler::SchedulerConfig = SchedulerConfig(; name="nothing", options=NoneConfig("nothing"))
    loadbestatend::Bool = true
end

@option mutable struct ModelConfig
    name::String = "nde0"
    hiddendim::Int = 64
    messagedim::Int = 0
    embeddim::Int = 0
    augdim::Int = 0
    graph::String = "full"
    numfactors::Int = 1
    zeromsg::Bool = false
end

@option mutable struct TrainConfig
    seed::Int = 42
    collocate::CollocateConfig = CollocateConfig()
    singlestep::SingleStepConfig = SingleStepConfig()
    shooting::ShootingConfig = ShootingConfig()
    data::DataConfig = DataConfig(; datasize=1000, horizon=40, dt=0.05)
    model::ModelConfig = ModelConfig()
    integrator::IntegratorConfig = IntegratorConfig()
    save::SaveConfig = SaveConfig()
    earlystop::EarlyStopping = EarlyStopping()
end


function read(conf::Union{String,Nothing})::TrainConfig
    if isnothing(conf)
        return TrainConfig()
    end
    if ispath(conf)
        return Configurations.from_toml(TrainConfig, conf)
    end
end

function full_dict(x)
    Configurations.is_option(x) || return x
    d = OrderedDict{String,Any}()
    T = typeof(x)
    for name in fieldnames(T)
        type = fieldtype(T, name)
        value = getfield(x, name)
        field_dict = full_dict(value)
        if Configurations.is_option(value) && type isa Union
            d[string(name)] = OrderedDict{String,Any}(alias(typeof(value)) => field_dict)
        else
            d[string(name)] = field_dict
        end
    end
    return d
end

function dump(config::TrainConfig)
    TOML.print(full_dict(config))
end

function dump(io::IO, config::TrainConfig)
    TOML.print(io, full_dict(config))
end


end # Options

include("evalconfig.jl")