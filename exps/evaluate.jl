using Plots

function compare_trajectories(dt;figsize=(800, 600), trajs...)
    @assert all(ndims(traj) == 2 for traj in values(trajs))
    xdim = size(iterate(values(trajs))[1], 1)
    t_size = size(values(trajs)[1])[end]
    #ts = 0.0:dt:dt*t_size
    ts = collect(1:t_size) .* dt
    plot([plot(ts, hcat((traj[i, :] for traj in values(trajs))...), label=map(string, reshape(collect(keys(trajs)), 1, length(trajs)))) for i in 1:xdim]..., size=figsize)
end

#= sys = TASK_REGISTRY["default_lorenz2"] =#
#= 
dim = 6;
indim = dim
hiddendim = 64;
outdim = dim;
state_encoder = Dense(indim, hiddendim, tanh)
control_encoder = x -> x
dyn = SimpleNeuralDELayer(Dense(hiddendim, hiddendim, tanh))
decoder = Dense(hiddendim, outdim)
model = NeuralDynamicsModel(state_encoder, control_encoder, dyn, decoder)
JLD2.@load "pt/compare_nde_vs_gde/nde/ckpt_epoch_best.jld2" weights
=#

#= Flux.loadparams!(model, weights)

x0 = randn(6)
us = [randn(0) for _ in 1:50]
true_traj = rollout(sys, x0, us; save_everystep=false, save_start=false)
pred_traj = rollout(model, x0, us; save_everystep=false, save_start=false)
p = compare_trajectories(0:0.05:0.05*50, sys=true_traj[:, :, 1], pred=pred_traj[:, :, 1]) =#