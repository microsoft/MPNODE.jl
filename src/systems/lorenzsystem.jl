struct LorenzSystem{F <: AbstractVector,M <: AbstractMatrix, G <:AbstractMatrix} <: AbstractModel
    sigmas::F
    rhos::F
    betas::F
    H::M
    A::G
end
state_dim(sys::LorenzSystem) = size(sys.H, 1)
control_dim(sys::LorenzSystem) = 0
sysparam_dim(sys::LorenzSystem) = 0
num_components(sys::LorenzSystem) = convert(Int, size(sys.H, 1) // 3)
state_range(sys::LorenzSystem) = [(-1., 1.) for i in 1:state_dim(sys)]
control_range(sys::LorenzSystem) = []
sysparam_range(sys::LorenzSystem) = []
adj_mat(sys::LorenzSystem) = sys.A

Flux.@functor LorenzSystem

function dynamics(f::LorenzSystem, x)
    n = convert(Int, size(f.H, 1) // 3)
    x_ = x[1:3:end]
    y_ = x[2:3:end]
    z_ = x[3:3:end]
    if size(x_, 1) != n
        x_ = reshape(x_, n, size(x)[2:end]...)
        y_ = reshape(y_, n, size(x)[2:end]...)
        z_ = reshape(z_, n, size(x)[2:end]...)
    end
    dxdt = f.sigmas .* (y_ - x_)
    dydt = x_ .* (f.rhos .- z_) - y_
    dzdt = x_ .* y_ - f.betas .* z_

    # WTF: this doesn't work???
    #dinpdt = Zygote.Buffer(x, size(x)...)
    dinpdt = zeros(size(x)...)
    dinpdt[1:3:3*n] .= dxdt
    dinpdt[2:3:3*n] .= dydt
    dinpdt[3:3:3*n] .= dzdt
    #dinpdt = copy(dinpdt)

    # coupling
    dinpdt = dinpdt + (f.H * x)
    return dinpdt
end
# LorenzSystem has no control
dynamics(f::LorenzSystem, x, u) = dynamics(f, x)

function init_state(sys::LorenzSystem; rng=Random.GLOBAL_RNG)
    return randn(rng, state_dim(sys))
end


struct StaticSystem{T,I <: Int} <: AbstractModel
    sys::T
    n::I
end

function (f::StaticSystem)(x)
    # Converting to StaticArrays for compatibility with DynamicalSystems.jl
    # May not be great for larger problems
    ret = SVector{f.n}(f.sys(x))
    return ret
end


function default_lorenz_system(k; seed=42)
    n = 3 * k # 3 spatial directions
    rng = MersenneTwister(seed)
    H = rand(rng, n, n)

    for i in 1:3:n
        H[i:i+2, i:i+2] .= 0.0
    end

    if k > 1
        H = 0.01 .* H ./ norm(H)
    end

    sigmas = [10.0 for _ in 1:k]
    rhos = [28.0 for _ in 1:k]
    betas = [8 / 3 for _ in 1:k]

    A = ones(Int, k, k) - Diagonal(ones(Int, k))
    system = LorenzSystem(sigmas, rhos, betas, H, A)
    return system
end

function default_lorenz_system_medium(k; seed=42)
    n = 3 * k # 3 spatial directions
    rng = MersenneTwister(seed)
    H = rand(rng, n, n)

    for i in 1:3:n
        H[i:i+2, i:i+2] .= 0.0
    end

    if k > 1
        H = 1 .* H ./ norm(H)
    end

    sigmas = [10.0 for _ in 1:k]
    rhos = [28.0 for _ in 1:k]
    betas = [8 / 3 for _ in 1:k]

    A = ones(Int, k, k) - Diagonal(ones(Int, k))
    system = LorenzSystem(sigmas, rhos, betas, H, A)
    return system
end

function default_lorenz_system_high(k; seed=42)
    n = 3 * k # 3 spatial directions
    rng = MersenneTwister(seed)
    H = rand(rng, n, n)

    for i in 1:3:n
        H[i:i+2, i:i+2] .= 0.0
    end

    if k > 1
        H = 5 .* H ./ norm(H)
    end

    sigmas = [10.0 for _ in 1:k]
    rhos = [28.0 for _ in 1:k]
    betas = [8 / 3 for _ in 1:k]

    A = ones(Int, k, k) - Diagonal(ones(Int, k))
    system = LorenzSystem(sigmas, rhos, betas, H, A)
    return system
end

function diagonal_lorenz_system(k; seed=42)
    n = 3 * k
    rng = MersenneTwister(seed)
    H = Diagonal(randn(rng, n))
    H = 5.0 .* H ./ norm(H)

    H = Diagonal(ones(n))
    sigmas = [10.0 for _ in 1:k]
    rhos = [28.0 for _ in 1:k]
    betas = [8 / 3 for _ in 1:k]

    A = Diagonal(ones(Int, k))
    system = LorenzSystem(sigmas, rhos, betas, H, A)
    return system
end

#=
sys = default_lorenz_system(2)
x0 = randn(6)
p0, re = Flux.destructure(sys)
dudt(u, p, t) = re(p)(u)
ff = ODEFunction{false}(dudt)
prob = ODEProblem{false}(ff, x0, (0.0, 10.0), p0)
sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
solve(prob, Tsit5(), saveat=0.1, reltol=1e-4, sense=sense) =#

#=
using DynamicalSystems
x0 = SVector{6}(randn(6))
sys = StaticSystem(default_lorenz_system(2), 6)
p0, re = Flux.destructure(sys)
dudt(u, p, t) = re(p)(u)
ff = ODEFunction{false}(dudt)
prob = ODEProblem{false}(ff, x0, (0.0, 10.0), p0)
cds = ContinuousDynamicalSystem(prob) =#
