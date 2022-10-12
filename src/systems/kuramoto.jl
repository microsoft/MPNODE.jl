using NetworkDynamics
using LightGraphs

struct SimpleKuramoto <: AbstractModel
    node_dim
    edge_dim
    A
    vertex_params
    edge_params
    dynamics
end

Flux.@functor SimpleKuramoto

state_dim(sys::SimpleKuramoto) = sys.node_dim * size(sys.A, 1)
control_dim(sys::SimpleKuramoto) = 0
sysparam_dim(sys::SimpleKuramoto) = 0
num_components(sys::SimpleKuramoto) = size(sys.A, 1)
adj_mat(sys::SimpleKuramoto) = sys.A
state_range(sys::SimpleKuramoto) = repeat([(-1.0, 1.0)], state_dim(sys)) # arbitrary, need to create a "spatial layout"
control_range(sys::SimpleKuramoto) = []
sysparam_range(sys::SimpleKuramoto) = []


function simple_kuramoto_vertex!(dv, v, edges, p, t)
    dv .= p
    sum_coupling!(dv, edges)
    nothing
end

function simple_kuramoto_edge!(e, v_s, v_d, p, t)
    e .= p .* sin.(v_s .- v_d)
    nothing
end

function dynamics(f::SimpleKuramoto, x, u)
    dx = similar(x)
    f.dynamics(dx, x, (f.vertex_params, f.edge_params), 0.0)
    return dx
end

function simple_kuramoto_barabasi(n, k, dim; seed=42)
    rng = MersenneTwister(seed)
    G = barabasi_albert(n, k; seed=seed)
    A = Matrix(adjacency_matrix(G))
    odevertex = ODEVertex(f! = simple_kuramoto_vertex!, dim=dim)
    staticedge = StaticEdge(f! = simple_kuramoto_edge!, dim=dim)

    v_par = randn(rng, dim)
    v_pars = [v_par for v in vertices(G)] # homogenous
    e_pars = [1. / 3. .* ones(dim) for e in edges(G)]
    kuramoto_dynamics! = network_dynamics(odevertex, staticedge, G)
    return SimpleKuramoto(dim, dim, A, v_pars, e_pars, kuramoto_dynamics!)
end

function simple_kuramoto_erdos(n, k, dim; seed=42)
    rng = MersenneTwister(seed)
    G = erdos_renyi(n, k; seed=seed)
    A = Matrix(adjacency_matrix(G))
    odevertex = ODEVertex(f! = simple_kuramoto_vertex!, dim=dim)
    staticedge = StaticEdge(f! = simple_kuramoto_edge!, dim=dim)

    v_par = randn(rng, dim)
    v_pars = [v_par for v in vertices(G)] # homogenous
    e_pars = [1. / 3. .* ones(dim) for e in edges(G)]
    kuramoto_dynamics! = network_dynamics(odevertex, staticedge, G)
    return SimpleKuramoto(dim, dim, A, v_pars, e_pars, kuramoto_dynamics!)    
end

function simple_kuramoto_wattz(n, k, dim; seed=42)
    rng = MersenneTwister(seed)
    G = watts_strogatz(n, k, 0.5; seed=seed)
    A = Matrix(adjacency_matrix(G))
    odevertex = ODEVertex(f! = simple_kuramoto_vertex!, dim=dim)
    staticedge = StaticEdge(f! = simple_kuramoto_edge!, dim=dim)

    v_par = randn(rng, dim)
    v_pars = [v_par for v in vertices(G)] # homogenous
    e_pars = [1. / 3. .* ones(dim) for e in edges(G)]
    kuramoto_dynamics! = network_dynamics(odevertex, staticedge, G)
    return SimpleKuramoto(dim, dim, A, v_pars, e_pars, kuramoto_dynamics!)        
end

struct SimpleControlledKuramoto <: AbstractModel
    node_dim
    edge_dim
    A
    vertex_params
    edge_params
    dynamics
end

Flux.@functor SimpleControlledKuramoto

state_dim(sys::SimpleControlledKuramoto) = sys.node_dim * size(sys.A, 1)
control_dim(sys::SimpleControlledKuramoto) = size(sys.A, 1)
sysparam_dim(sys::SimpleControlledKuramoto) = 0
num_components(sys::SimpleControlledKuramoto) = size(sys.A, 1)
adj_mat(sys::SimpleControlledKuramoto) = sys.A
state_range(sys::SimpleControlledKuramoto) = repeat([(-1., 1)], state_dim(sys)) # arbitrary, need to create a "spatial layout"
control_range(sys::SimpleControlledKuramoto) = []
sysparam_range(sys::SimpleControlledKuramoto) = []


function dynamics(f::SimpleControlledKuramoto, x, u)
    dx = similar(x)
    f.dynamics(dx, x, (map(.+, u, f.vertex_params), f.edge_params), 0.0)
end

function simple_controlled_kuramoto_barabasi(n, k, dim; seed=42)
    rng = MersenneTwister(seed)
    G = barabasi_albert(n, k)
    A = Matrix(adjacency_matrix(G))
    odevertex = ODEVertex(f! = simple_kuramoto_vertex!, dim=dim)
    staticedge = StaticEdge(f! = simple_kuramoto_edge!, dim=dim)

    v_par = randn(rng, dim)
    v_pars = [v_par for v in vertices(G)] # homogenous
    e_pars = [1. / 3. .* ones(dim) for e in edges(G)]
    kuramoto_dynamics! = network_dynamics(odevertex, staticedge, G)
    return SimpleControlledKuramoto(dim, dim, A, v_pars, e_pars, kuramoto_dynamics!)    
end