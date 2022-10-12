# Learning Modular Simulations for Homogenous Networks

First install Julia from https://julialang.org/downloads/ and ensure that it's available on your `$PATH`. Currently only tested on Ubuntu 18.04 and Julia v1.6.

## Usage

After cloning the repository, run:

```bash
julia --project=exps 'using Pkg; Pkg.instantiate()'
```

`train.jl` allows training a particular model defined in the config on a particular system dataset.

For example, to train on the system `default_lorenz3` system on the L2S model:
```bash
julia --project=exps exps/train.jl default_lorenz3 --conf configs/gnnsc_de0_sys_32.toml --evalconf configs/evaltemplate.toml --savedir /tmp/mpdifflogs/l2strial
```
or on the MP-NODE model:
```bash
julia --project=exps exps/train.jl default_lorenz3 --conf configs/empode_sys_32_3.toml --evalconf configs/evaltemplate.toml --savedir /tmp/mpdifflogs/mpnodetrial
```

## Code overview

- `src/systems`: contains the systems we evaluate the MP-NODE and other baselines on.
- `src/datagen.jl`: contains utilities to generate data from the systems.
- `src/empde.jl`: contains the pieces required to setup MP-NODE models.
- `src/gnnode.jl`: contains the pieces required to setup L2S models.
- `src/nde.jl`: contains the pieces required to setup NODE models.
- `src/learn.jl` and `src/learn_utils.jl`: contains the model training utilities.
- `exps/models.jl`: contains the specific models used in the paper.
- `exps/train_pipeline.jl`: contains orchestration for full training pipeline.
- `exps/tasks.jl`: contains the specific systems used in the paper.  


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
