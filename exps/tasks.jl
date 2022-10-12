using MPNODE
using MPNODE: default_lorenz_system, diagonal_lorenz_system, coupled_pendulum, simple_kuramoto_barabasi, simple_controlled_kuramoto_barabasi, simple_kuramoto_erdos, simple_kuramoto_wattz, default_lorenz_system_medium, default_lorenz_system_high
using Random

const LORENZ_TASKS = merge(Dict("diagonal_lorenz$i" => diagonal_lorenz_system(i) for i in 1:10), 
                        Dict("default_lorenz$i" => default_lorenz_system(i) for i in 1:10),
                        Dict("default_lorenz_med$i" => default_lorenz_system_medium(i) for i in 1:10),
                        Dict("default_lorenz_high$i" => default_lorenz_system_high(i) for i in 1:10),                        
                        Dict("different_seeds_default_lorenz$i" => [default_lorenz_system(i; seed=k) for k in abs.(rand(MersenneTwister(69), Int, 5))] for i in 2:10),
                        Dict("default+diagonal_lorenz$i" => [default_lorenz_system(i), diagonal_lorenz_system(i)] for i in 2:10),
                        )
const KURAMOTO_TASKS = merge(
    Dict("simple_barabasi_kuramoto$(n)_$k" => simple_kuramoto_barabasi(n, k, 3) for n in 1:20, k in 1:10 if n > k),
    Dict("simple_erdos_kuramoto$(n)_$k" => simple_kuramoto_erdos(n, k, 3) for n in 1:20, k in 1:10 if n > k),
    #Dict("simple_barabasi_wattz$(n)_$k" => simple_kuramoto_wattz(n, k, 3) for n in 1:20, k in 1:10 if n > k),
    Dict("simple_wattz_kuramoto10_5" => simple_kuramoto_wattz(10, 5, 3)),
    Dict("simple_kuramoto_barabasi_multi10_5" => [simple_kuramoto_barabasi(10, 5, 3; seed=i) for i in 1:5],
        "simple_kuramoto_erdos_multi10_5" => [simple_kuramoto_erdos(10, 5, 3; seed=i) for i in 1:5],
        "simple_kuramoto_wattz_multi10_5" => [simple_kuramoto_wattz(10, 5, 3; seed=i) for i in 1:5],
        "simple_kuramoto_two_multi10_5" => vcat([simple_kuramoto_barabasi(10, 5, 3; seed=i) for i in 1:5], [simple_kuramoto_erdos(10, 5, 3; seed=i) for i in 1:5]),
    ),

    Dict("simple_kuramoto_barabasi_two10_5" => [simple_kuramoto_barabasi(10, 5, 3; seed=i) for i in 1:2]),
    #Dict("simple_controlled_barabasi_kuramoto$(n)_$k" => simple_controlled_kuramoto_barabasi(n, k, 3) for n in 1:20, k in 3:10 if n > k),
    Dict("different_seeds_simple_barabasi_kuramoto$(n)_$k" => [simple_kuramoto_barabasi(n, k, 3; seed=s) for s in abs.(rand(MersenneTwister(69), Int, 5))] for n in 1:20, k in 1:10 if n>k)
)
#=
const KURAMOTO_TASKS = merge(
    Dict("simple_barabasi_kuramoto$(n)_$k" => simple_kuramoto_barabasi(n, k, 3) for n in 1:20, k in 1:10 if n > k),
    Dict("simple_barabasi_erdos$(n)_$k" => simple_kuramoto_erdos(n, k, 3) for n in 1:20, k in 1:10 if n > k),
    #Dict("simple_barabasi_wattz$(n)_$k" => simple_kuramoto_wattz(n, k, 3) for n in 1:20, k in 1:10 if n > k),
    Dict("simple_kuramoto_wattz10_5" => simple_kuramoto_wattz(10, 5, 3)),
    Dict("simple_kuramoto_barabasi_multi10_5" => [simple_kuramoto_barabasi(10, 5, 3; seed=i) for i in 1:5],
        "simple_kuramoto_erdos_multi10_5" => [simple_kuramoto_erdos(10, 5, 3; seed=i) for i in 1:5],
        "simple_kuramoto_wattz_multi10_5" => [simple_kuramoto_wattz(10, 5, 3; seed=i) for i in 1:5],
        "simple_kuramoto_two_multi10_5" => vcat([simple_kuramoto_barabasi(10, 5, 3; seed=i) for i in 1:5], [simple_kuramoto_erdos(10, 5, 3; seed=i) for i in 1:5]),

    ),
    #Dict("simple_controlled_barabasi_kuramoto$(n)_$k" => simple_controlled_kuramoto_barabasi(n, k, 3) for n in 1:20, k in 3:10 if n > k),
    Dict("different_seeds_simple_barabasi_kuramoto$(n)_$k" => [simple_kuramoto_barabasi(n, k, 3; seed=s) for s in abs.(rand(MersenneTwister(69), Int, 5))] for n in 1:20, k in 1:10 if n>k)
)
=#
const HEAT_TASKS = Dict(

    "heat_er_small" => MPNODE.heat_diffusion_random(16),
    "heat_ba_small" => MPNODE.heat_diffusion_powerlaw(16),
    "heat_ws_small" => MPNODE.heat_diffusion_smallworld(16),
    "heat_er_large" => MPNODE.heat_diffusion_random(64),
    "heat_ba_large" => MPNODE.heat_diffusion_powerlaw(64),
    "heat_ws_large" => MPNODE.heat_diffusion_smallworld(64),

    "heat_multiple_small" => [MPNODE.heat_diffusion_powerlaw(16), MPNODE.heat_diffusion_smallworld(16)],
    "heat_er_small_multi" => [MPNODE.heat_diffusion_random(16,seed=i) for i in 1:5],
    "heat_ba_small_multi" => [MPNODE.heat_diffusion_powerlaw(16,seed=i) for i in 1:5],
    "heat_ws_small_multi" => [MPNODE.heat_diffusion_smallworld(16,seed=i) for i in 1:5],
    "heat_er_large_multi" => [MPNODE.heat_diffusion_random(64,seed=i) for i in 1:5],
    "heat_ws_large_multi" => [MPNODE.heat_diffusion_smallworld(64,seed=i) for i in 1:5],
    "heat_ba_large_multi" => [MPNODE.heat_diffusion_powerlaw(64,seed=i) for i in 1:5],
    "heat_multi_two_small" => vcat([MPNODE.heat_diffusion_random(16,seed=i) for i in 1:5], [MPNODE.heat_diffusion_smallworld(16,seed=j) for j in 1:5]),
    "heat_multi_two_large" => vcat([MPNODE.heat_diffusion_random(64,seed=i) for i in 1:5], [MPNODE.heat_diffusion_smallworld(64,seed=j) for j in 1:5]),

)

const GENE_TASKS = Dict(
    "gene_er_small" => MPNODE.gene_evolution_random(16),
    "gene_ba_small" => MPNODE.gene_evolution_powerlaw(16),
    "gene_ws_small" => MPNODE.gene_evolution_smallworld(16),
    "gene_er_large" => MPNODE.gene_evolution_random(64),
    "gene_ba_large" => MPNODE.gene_evolution_powerlaw(64),
    "gene_ws_large" => MPNODE.gene_evolution_smallworld(64),
    "gene_er_small_multi" => [MPNODE.gene_evolution_random(16,seed=i) for i in 1:5],
    "gene_ba_small_multi" => [MPNODE.gene_evolution_powerlaw(16,seed=i) for i in 1:5],
    "gene_ws_small_multi" => [MPNODE.gene_evolution_smallworld(16,seed=i) for i in 1:5],
    "gene_ws_large_multi" => [MPNODE.gene_evolution_random(64,seed=i) for i in 1:5],
    "gene_ws_large_multi" => [MPNODE.gene_evolution_powerlaw(64,seed=i) for i in 1:5],
    "gene_ws_large_multi" => [MPNODE.gene_evolution_smallworld(64,seed=i) for i in 1:5],
    "gene_multiple_small" => [MPNODE.gene_evolution_powerlaw(16),MPNODE.gene_evolution_smallworld(16)],
    "gene_er_small_multi" => [MPNODE.gene_evolution_random(16,seed=i) for i in 1:5],
    "gene_multi_two_small" => vcat([MPNODE.gene_evolution_random(16,seed=i) for i in 1:5], [MPNODE.gene_evolution_smallworld(16,seed=j) for j in 1:5]),    
    "gene_multi_two_small" => vcat([MPNODE.gene_evolution_random(16,seed=i) for i in 1:5], [MPNODE.gene_evolution_smallworld(16,seed=j) for j in 1:5]),
    "gene_multi_two_large" => vcat([MPNODE.gene_evolution_random(64,seed=i) for i in 1:5], [MPNODE.gene_evolution_smallworld(64,seed=j) for j in 1:5]),    
)

const TASK_REGISTRY = merge(LORENZ_TASKS, KURAMOTO_TASKS,
    # ndcn networks
    HEAT_TASKS,
    GENE_TASKS,
    Dict(
    "single_pendulum" => MPNODE.single_pendulum(),        
    "coupled_pendulum" => coupled_pendulum(),
    "cartpole" => MPNODE.default_cartpole(),
    "default_linear4" => MPNODE.default_linear_system(4),
    "default_linear6" => MPNODE.default_linear_system(6),
    "diagonal_linear4" => MPNODE.diagonal_linear_system(4),
    "coupled_linear4" => MPNODE.default_coupledlinear_system(4),
    # "stickyrice_elastic" => StickyRice("/scratch/jagupt/stickyrice/data_elastic/"),
    # "stickyrice_plastic" => StickyRice("/scratch/jagupt/stickyrice/data_plastic/"),

))

function get_lorenz_transfer()
    #vcat(["default_lorenz$i" for i in 2:10], ["diagonal_lorenz$i" for i in 2:10])
    ["default_lorenz10", "default_lorenz_med10", "default_lorenz_high10"]
end

function get_heatdiffusion_transfer()
    vcat([""])
end

function get_simple_kuramoto_transfer()
    vcat(["simple_barabasi_kuramoto$(n)_$k" for n in 1:10 for k in 1:8 if n > k], [])
end

function get_simple_controlled_kuramoto_transfer()
    vcat(["simple_controlled_barabasi_kuramoto$(n)_$k" for n in 1:10 for k in 1:8 if n > k], [])
end

const TASK2TRANSFER = merge(
    Dict("quad3" => [], "quad6" => []),
    Dict("default_lorenz$i" => get_lorenz_transfer() for i in 2:10),
    Dict("default_lorenz_med$i" => get_lorenz_transfer() for i in 2:10),
    Dict("heat_diffusion$i" => get_heatdiffusion_transfer() for i in 2:2:20),
    Dict("heat_er_small" => vcat(["heat_ba_small", "heat_ws_small", "heat_er_small_multi",  "heat_er_large", "heat_ws_large", "heat_ba_large"])),
    Dict("gene_er_small" => vcat(["gene_ba_small", "gene_ws_small", "gene_er_small_multi", "gene_er_large", "gene_ws_large", "gene_ba_large"])),
    Dict("gene_ba_small_multi" => vcat(["gene_er_small", "gene_ws_small", "gene_ba_small", "gene_er_large", "gene_ws_large", "gene_ba_large"])),
    Dict("gene_multi_two_small" => vcat(["gene_er_small", "gene_ws_small", "gene_ba_small", "gene_er_large", "gene_ws_large", "gene_ba_large"])),
    Dict("heat_ba_small_multi" => vcat(["heat_er_small", "heat_ws_small", "heat_ba_small", "heat_er_large", "heat_ws_large", "heat_ba_large"])),
    Dict("heat_multi_two_small" => vcat(["heat_er_small", "heat_ws_small", "heat_ba_small", "heat_er_large", "heat_ws_large", "heat_ba_large"])),

    Dict("simple_barabasi_kuramoto$(n)_$(k)" => ["simple_barabasi_kuramoto6_3", "simple_barabasi_kuramoto10_3", "simple_barabasi_kuramoto7_6", "simple_barabasi_kuramoto12_10", "simple_barabasi_kuramoto20_9", "simple_barabasi_kuramoto3_2"] for n in 1:20, k in 3:10 if n > k),
    Dict("simple_kuramoto_two_multi10_5" => ["simple_barabasi_kuramoto10_5", "simple_erdos_kuramoto10_5", "simple_wattz_kuramoto10_5"]),
    Dict("simple_kuramoto_barabasi_multi10_5"=> ["simple_barabasi_kuramoto6_3", "simple_barabasi_kuramoto10_3", "simple_barabasi_kuramoto7_6", "simple_barabasi_kuramoto12_10", "simple_barabasi_kuramoto20_9", "simple_barabasi_kuramoto3_2"]),
    Dict("simple_controlled_barabasi_kuramoto$(n)_$(k)" => get_simple_controlled_kuramoto_transfer() for n in 1:20, k in 3:10 if n > k)
)