module Models

using LinearAlgebra
using Flux
using Zygote
using Mill
using MPNODE: NeuralDynamicsModel, NeuralMPDynamicsModel, SimpleNeuralDELayer, AugmentedNeuralDELayer, NeuralMPDELayer
using MPNODE: GNN, GNNSC
using MPNODE: EMPDynamicsModel, EMPEDDynamicsModel, EMPEDMsgDynamicsModel, EMPEDBypassDynamicsModel

Mill.bagcount!(false)

function DenseLN(in, out, σ=identity; kwargs...)
    Chain(Dense(in, out, σ; kwargs...), LayerNorm(out))
end

function nde4(statedim, controldim, hiddendim)
    state_encoder = identity
    control_encoder = identity
    dyn = SimpleNeuralDELayer(Chain(Dense(statedim+controldim, hiddendim, tanh), Dense(hiddendim, hiddendim, tanh), Dense(hiddendim, statedim)))
    decoder = identity
    return NeuralDynamicsModel(state_encoder, control_encoder, dyn, decoder)
end

function ndeln4(statedim, controldim, hiddendim)
    state_encoder = identity
    control_encoder = identity
    dyn = SimpleNeuralDELayer(Chain(DenseLN(statedim+controldim, hiddendim, tanh), DenseLN(hiddendim, hiddendim, tanh), Dense(hiddendim, statedim)))
    decoder = identity
    return NeuralDynamicsModel(state_encoder, control_encoder, dyn, decoder)
end

function gnn_de0(statefeaturedim, controlfeaturedim, hiddendim)
    agg(d) = Mill.sum_aggregation(d)
    state_encoder = identity
    control_encoder = identity
    gnn = GNN(
        ArrayModel(Dense(statefeaturedim + controlfeaturedim, hiddendim, tanh)),
        BagModel(Dense(hiddendim, hiddendim, tanh), agg(hiddendim), Dense(hiddendim, hiddendim, tanh)),
        Dense(hiddendim, statefeaturedim)
    )
    dyn = NeuralMPDELayer(gnn)
    decoder = identity
    model = NeuralMPDynamicsModel(state_encoder, control_encoder, dyn, decoder)
    return model
end

function gnnln_de0(statefeaturedim, controlfeaturedim, hiddendim)
    agg(d) = Mill.sum_aggregation(d)
    state_encoder = identity
    control_encoder = identity
    gnn = GNN(
        ArrayModel(DenseLN(statefeaturedim + controlfeaturedim, hiddendim, tanh)),
        BagModel(DenseLN(hiddendim, hiddendim, tanh), agg(hiddendim), DenseLN(hiddendim, hiddendim, tanh)),
        Dense(hiddendim, statefeaturedim))
    dyn = NeuralMPDELayer(gnn)
    decoder = identity
    model = NeuralMPDynamicsModel(state_encoder, control_encoder, dyn, decoder)
    return model
end

function gnnsc_de0(statefeaturedim, controlfeaturedim, hiddendim)
    agg(d) = Mill.sum_aggregation(d)
    state_encoder = identity
    control_encoder = identity
    gnn = GNNSC(
        ArrayModel(Dense(statefeaturedim + controlfeaturedim, hiddendim, tanh)),
        BagModel(Dense(hiddendim, hiddendim, tanh), agg(hiddendim), Dense(hiddendim, hiddendim, tanh)),
        Dense(hiddendim, statefeaturedim)        
    )
    dyn = NeuralMPDELayer(gnn)
    decoder = identity
    model = NeuralMPDynamicsModel(state_encoder, control_encoder, dyn, decoder)
    return model
end

function gnnscln_de0(statefeaturedim, controlfeaturedim, hiddendim)
    agg(d) = Mill.sum_aggregation(d)
    state_encoder = identity
    control_encoder = identity
    gnn = GNNSC(
        ArrayModel(DenseLN(statefeaturedim + controlfeaturedim, hiddendim, tanh)),
        BagModel(DenseLN(hiddendim, hiddendim, tanh), agg(hiddendim), DenseLN(hiddendim, hiddendim, tanh)),
        Dense(hiddendim, statefeaturedim))
    dyn = NeuralMPDELayer(gnn)
    decoder = identity
    model = NeuralMPDynamicsModel(state_encoder, control_encoder, dyn, decoder)
    return model
end


function empode(xdim, udim, mdim, hdim, ldim)
    dyn = Chain(Dense(xdim + udim + mdim, hdim, tanh), Dense(hdim, hdim, tanh), Dense(hdim, xdim + mdim))
    return EMPDynamicsModel(SimpleNeuralDELayer(dyn), xdim, mdim, hdim)
end

end