using LightGraphs
using Plots
# Follows the ndcn implementation
# https://github.com/calvin-zcx/ndcn/blob/master/gene_dynamics.py#L186

struct GeneEvolution <: AbstractModel
    A
    b
    g
    h
end

state_dim(sys::GeneEvolution) = size(sys.A, 1)
control_dim(sys::GeneEvolution) = 0
sysparam_dim(sys::GeneEvolution) = 0
num_components(sys::GeneEvolution) = size(sys.A, 1)
adj_mat(sys::GeneEvolution) = sys.A

state_range(sys::GeneEvolution) = repeat([(0, 50)], state_dim(sys)) # arbitrary, need to create a "spatial layout"
control_range(sys::GeneEvolution) = []
sysparam_range(sys::GeneEvolution) = []

dynamics(f::GeneEvolution, x, u) = dynamics(f, x)

function dynamics(f::GeneEvolution, x)
    return -f.b * (x.^f.g) + (f.A * (x.^f.h ./ (1 .+ x.^f.h)))
end

function gene_evolution_random(n; seed=42)
    G = erdos_renyi(n, 0.5, seed=seed)
    A = Matrix(adjacency_matrix(G))
    system = GeneEvolution(A, 1, 1, 2)
    return system
end

function gene_evolution_powerlaw(n; seed=42)
    G = barabasi_albert(n, Int(n/2), seed=seed)
    A = Matrix(adjacency_matrix(G))

    system = GeneEvolution(A, 1, 1, 2)
    return system
end

function gene_evolution_smallworld(n; seed=42)
    G = watts_strogatz(n, Int(n/2), 0.5, seed=seed)
    A = Matrix(adjacency_matrix(G))

    system = GeneEvolution(A, 1, 1, 2)
    return system
end

function custom_init_state(sys::GeneEvolution; rng=Random.GLOBAL_RNG)
    N = convert(Int, sqrt(state_dim(sys)))
    x0 = zeros(N, N)

    for hs = 1:3
        hs_start_x = rand(rng, Uniform(0.1, 0.9))
        hs_start_y = rand(rng, Uniform(0.1, 0.9))

        hs_end_x = min(hs_start_x + rand(rng, Uniform(0.1, 0.2)), 1.0)
        hs_end_y = min(hs_start_y + rand(rng, Uniform(0.1, 0.2)), 1.0)

        x0[Int(ceil(hs_start_x * N)):Int(ceil(hs_end_x * N)), Int(ceil(hs_start_y * N)):Int(ceil(hs_end_y * N))] .= rand(Random.GLOBAL_RNG, Uniform(30, 50))
    end

    return reshape(x0, (state_dim(sys),))
end

using Plots.PlotMeasures

function plot_state(sys::GeneEvolution, x, xgt)
    # return a handle to some nice system-specific plot over the grid
    N = Int(sqrt(state_dim(sys)))
    z = reshape(x, (N, N))
    zgt = reshape(xgt, (N, N))
    x = 1:N
    y = 1:N
    p1 = plot(x,y,zgt, st=:surface, xlabel="Node X", ylabel="Node Y", zlabel="Value", title="Ground Truth", rightmargin=15mm, opacity=0.9)
    p2 = plot(x,y,z, st=:surface, xlabel="Node X", ylabel="Node Y", zlabel="Value", title="Prediction", rightmargin=15mm, opacity=0.9)
    return plot(p1, p2, layout=(2, 1))
end

# Sample initial condition from ndcn implementation
#=
function custom_init_state(sys::GeneEvolution; rng=Random.GLOBAL_RNG)
    N = convert(Int, sqrt(state_dim(sys)))
    x0 = zeros(N, N)

    x0[Int(ceil(0.05 * N)):Int(ceil(0.25 * N)), Int(ceil(0.05 * N)):Int(ceil(0.25 * N))] .= 25
    x0[Int(ceil(0.45 * N)):Int(ceil(0.75 * N)), Int(ceil(0.35 * N)):Int(ceil(0.65 * N))] .= 17

    return reshape(x0, (state_dim(sys),))
end
=#
