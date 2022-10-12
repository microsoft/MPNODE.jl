import Zygote
import JLD2
using EarlyStopping

using BangBang
using ProgressLogging
using Logging

function _train_epoch!(model, opt, data, loss, metric_fns::Dict{Symbol,F}) where {F}
    # Ugly XXX
    epoch_train_losses = 0.0f0
    metrics = Dict{Symbol,Float64}()
    for (name, metric_fn) in metric_fns
        metrics[name] = 0.0
    end
    epoch_grad_norms = 0.0
    epoch_max_grad = -Inf
    count = 0

    train_t = @elapsed begin
        ps = Zygote.Params(Flux.params(model))
        for d in data
            try
                tl, gs = lossgrads(ps) do
                    loss(d...)
                end

                # XXX
                Zygote.ignore() do
                    epoch_train_losses += tl
                    count += 1
                    for (name, metric_fn) in metric_fns
                        metrics[name] += metric_fn(d...)
                    end

                    if all(isnothing, values(gs.grads))
                        @warn "-> All gradients are `nothing`!!!"
                        return
                    end
                    ginfo = gradinfo(collect(values(gs.grads)))
                    if ginfo.nan > 0
                        @warn "NaN in grad"
                    end
                    if ginfo.inf > 0
                        @warn "Inf in grad"
                    end
                    epoch_grad_norms += ginfo.l2
                    epoch_max_grad = max(epoch_max_grad, ginfo.max)
                end
                Flux.Optimise.update!(opt, ps, gs)
            catch ex
                if ex isa Flux.Optimise.StopException
                    break
                else
                    rethrow(ex)
                end
            end

        end
    end

    epoch_train_losses = epoch_train_losses / count
    epoch_grad_norms = epoch_grad_norms / count
    for name in keys(metrics)
        metrics[name] /= count
    end

    return merge((train_t=train_t, train_loss=epoch_train_losses, gnorm=epoch_grad_norms, gmax=epoch_max_grad), metrics)
end


"""
Transform TimeSeries Dataset to iterator for minibatch training
"""
function tsdataloader(data::Dataset; batchsize=1, shuffle=false)
    Flux.Data.DataLoader((data.xs, data.us, data.dt), batchsize=batchsize, shuffle=shuffle)
end
function tsdataloader(data::GraphDataset; batchsize=1, shuffle=false)
    Flux.Data.DataLoader((data.gs, data.xs, data.us, data.dt), batchsize=batchsize, shuffle=shuffle)
end

function shootingtrain!(model, loss, opt, scheduler, stopper, traindata, validdata, metrics_fns, other_callbacks, logger, config)

    traindata = tsdataloader(traindata; batchsize=config.shooting.batchsize, shuffle=true)
    validdata = tsdataloader(validdata; batchsize=config.shooting.batchsize)

    best_valid_error = Inf
    last_improvement = 0

    with_logger(logger) do
        @progress for (η, epoch_idx) in zip(scheduler, 1:config.shooting.nepochs)
            setlrate!(opt, η)
            metrics = _train_epoch!(model, opt, traindata, loss, metrics_fns)
            @info "train" epoch=epoch_idx metrics... lr=lrate(opt)


            test_t = @elapsed begin
                valid_error = mean(loss(td...) for td in validdata)
                valid_metrics = Dict([Symbol("valid_"*String(name)) => mean(metrics_fn(td...) for td in validdata) for (name, metrics_fn) in metrics_fns])
            end
            @info "test" val_loss=valid_error valid_metrics... test_time=test_t log_step_increment=0

            #=
            for (cname, callback) in other_callbacks
                cbv = callback()
                @info cname cbv log_step_increment = 0
            end
            =#
            # Save params with best validation error
            if valid_error <= best_valid_error
                best_valid_error = valid_error
                last_improvement = epoch_idx
                weights = Flux.params(model)
                JLD2.@save joinpath(config.save.dir, "ckpt_epoch_best.jld2") weights
            end

            # If we haven't seen improvement in patience epochs, drop our learning rate:
            if epoch_idx - last_improvement >= config.earlystop.patience && lrate(opt) > 1e-6
                setlrate!(opt, lrate(opt)/ 2.0)
                # XXX Hack to allow reducing lr on plateau at least with NoOpSchedule
                try
                    scheduler = setproperties!!(scheduler; start=scheduler.start / 2.0)
                catch
                end
                @warn(" -> Haven't improved in a while, dropping learning rate to $(lrate(opt))!")

                # After dropping learning rate, give it a few epochs to improve
                last_improvement = epoch_idx
            end

            if epoch_idx % config.save.freq == 0
                @info "Saving nn..." log_step_increment=0
                weights = Flux.params(model)
                JLD2.@save joinpath(config.save.dir, "ckpt_epoch_$epoch_idx.jld2") weights

                for (cname, callback) in other_callbacks
                    cbvs = callback()
                    @info cname cbvs log_step_increment = 0
                end
            end

            if done!(stopper, valid_error)
                @warn message(stopper)
                break
            end
        end
    end

    # Load back the best model?
    if config.shooting.loadbestatend
        JLD2.@load joinpath(config.save.dir, "ckpt_epoch_best.jld2") weights
        Flux.loadparams!(model, weights)
    end
    model
end
