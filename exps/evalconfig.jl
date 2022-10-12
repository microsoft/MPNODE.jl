module TrainEvalConfig

using Configurations
using OrderedCollections
import TOML

@option mutable struct TransferConfig
    enabled::Bool = false
    systems::Vector{String} = [""]
    datasize::Int = 0
    horizon::Int = 0
    dt::Float64 = 0.05
    statesampler::String = "rand_constrained"
    controlsampler::String = "rand_sinusoid"
end

@option mutable struct TrajPlotConfig
    trainidxs::Vector{Int} = [1, 2, 3, 4]
    valididxs::Vector{Int} = [1, 2, 3, 4]
end

@option mutable struct EvalConfig
    transfer::TransferConfig = TransferConfig()
    trajplot::TrajPlotConfig = TrajPlotConfig()
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

function read(conf::Union{String,Nothing})
    if isnothing(conf)
        return EvalConfig()
    end
    if ispath(conf)
        return from_toml(EvalConfig, conf)
    end
end

function Base.dump(config::EvalConfig)
    TOML.print(full_dict(config))
end

function Base.dump(io::IO, config::EvalConfig)
    TOML.print(io, full_dict(config))
end

end