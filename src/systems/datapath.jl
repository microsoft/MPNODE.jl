using FileIO
using NPZ
using JSON

struct DataPathSystem <: AbstractModel
    traindata
    validdata
    testdata
    metadata
end

function _loaddata(path, name)
    if isfile(joinpath(path, "$name.jld2"))
        data = FileIO.load(joinpath(path, "$name.jld2"))
    elseif isfile(joinpath(path, "$name.npz"))
        data = npzread(joinpath(path, "$name.npz"))
    end
    return data
end

"""
function DataPathSystem(path::AbstractString)

"""
function DataPathSystem(path::AbstractString)
    traindata = _loaddata(path, "train")
    validdata = _loaddata(path, "valid")
    testdata = _loaddata(path, "test")
    md = open(joinpath(path, "metadata.json")) do io
        JSON.parse(io)
    end
    return DataPathSystem(traindata, validdata, testdata, md)
end

state_dim(sys::DataPathSystem) = sys.metadata["state_dim"]
control_dim(sys::DataPathSystem) = sys.metadata["control_dim"]
sysparam_dim(sys::DataPathSystem) = get(sys.metadata, "sysparam_dim", 0)
state_range(sys::DataPathSystem) = map(Tuple{Float32,Float32}, get(sys.metadata, "state_range", []))
control_range(sys::DataPathSystem) = map(Tuple{Float32,Float32}, get(sys.metadata, "control_range", []))
sysparam_range(sys::DataPathSystem) = map(Tuple{Float32,Float32}, get(sys.metadata, "sysparam_range", []))
num_components(sys::DataPathSystem) = get(sys.metadata, "num_components", 1)

function dynamics(f::DataPathSystem, x, u)
    error("Not defined for systems that point to datasets")
end

function generate_dataset(system::DataPathSystem)
    if haskey(system.traindata, "gs")
        traindata = GraphDataset(system.traindata["gs"], system.traindata["xs"], system.traindata["us"], system.traindata["dt"])
        testdata = GraphDataset(system.testdata["gs"], system.testdata["xs"], system.testdata["us"], system.testdata["dt"])       
        validdata = GraphDataset(system.validdata["gs"], system.validdata["xs"], system.validdata["us"], system.validdata["dt"])
    elseif haskey(system.traindata, "n2fgs")
        traindata = FactorGraphDataset(system.traindata["n2fgs"], system.traindata["f2ngs"], system.traindata["xs"], system.traindata["us"], system.traindata["dt"])
        testdata = FactorGraphDataset(system.testdata["n2fgs"], system.testdata["f2ngs"], system.testdata["xs"], system.testdata["us"], system.testdata["dt"])       
        validdata = FactorGraphDataset(system.validdata["n2fgs"], system.validndata["f2ngs"], system.validdata["xs"], system.validdata["us"], system.validdata["dt"])        
    else
        traindata = Dataset(system.traindata["xs"], system.traindata["us"], system.traindata["dt"])
        validdata = Dataset(system.validdata["xs"], system.validdata["us"], system.validdata["dt"])
        testdata = Dataset(system.testdata["xs"], system.testdata["us"], system.testdata["dt"])
    end
    return traindata, validdata, testdata
end