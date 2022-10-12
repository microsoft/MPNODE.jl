# Define the model struct with parameters
struct Pendulum <: AbstractModel
    m      # mass in kg
    l      # length in m
    mu       # damping constant
end
state_dim(sys::Pendulum) = 2
control_dim(sys::Pendulum) = 0
sysparam_dim(sys::Pendulum) = 0
state_range(sys::Pendulum) = [(-pi/2, pi/2), (-1.0, 1.0)]
control_range(sys::Pendulum) = []
sysparam_range(sys::Pendulum) = []
num_components(sys::Pendulum) = 1

Flux.@functor Pendulum

struct CoupledPendulum <: AbstractModel
    p1::Pendulum
    p2::Pendulum
    k    # Spring constant
    H
end
state_dim(sys::CoupledPendulum) = 4
control_dim(sys::CoupledPendulum) = 0
sysparam_dim(sys::CoupledPendulum) = 0
state_range(sys::CoupledPendulum) = [(-pi/4, pi/4), (-0.5, 0.5), (-pi/4, pi/4), (-0.5, 0.5)]
control_range(sys::CoupledPendulum) = []
sysparam_range(sys::CoupledPendulum) = []
num_components(sys::CoupledPendulum) = 2
adj_mat(sys::CoupledPendulum) = ones(Int, 2, 2) - Diagonal(ones(Int, 2))

Flux.@functor CoupledPendulum

function dynamics(f::Pendulum, x)
    g = 9.81
    θ_ddot = ((-f.mu) * x[2]) + (-g / f.l) * sin(x[1])
    θ_dot = x[2]

    return [θ_dot, θ_ddot]
end
dynamics(f::Pendulum, x, u) = dynamics(f, x)

function dynamics(f::CoupledPendulum, x)
    g = 9.81
    θ1_ddot = (sin(x[1]) * (f.p1.m * (f.p1.l * (x[2] * x[2]) - g) - (f.k * f.p1.l)) + (f.k * f.p2.l * sin(x[3]))) / (f.p1.m * f.p1.l * cos(x[1]))
    θ2_ddot = (sin(x[3]) * (f.p2.m * (f.p2.l * (x[4] * x[4]) - g) - (f.k * f.p2.l)) + (f.k * f.p1.l * sin(x[1]))) / (f.p2.m * f.p2.l * cos(x[3]))

    θ1_dot = x[2]
    θ2_dot = x[4]

    return [θ1_dot, θ1_ddot, θ2_dot, θ2_ddot]
end
dynamics(f::CoupledPendulum, x, u) = dynamics(f, x)

function single_pendulum()
    # SysName(x_dim, u_dim, n, ...)
    system = Pendulum(1.0, 1.5, 0)
    return system
end

function init_state(sys::CoupledPendulum; rng=Random.GLOBAL_RNG)
    return randn(rng, state_dim(sys))
end

function coupled_pendulum()
    p1 = Pendulum(1.0, 1.5, 0)
    p2 = Pendulum(1.0, 1.5, 0)

    coupled_pend = CoupledPendulum(p1, p2, 2.0, [0 1 ; 1 0])

    return coupled_pend
end
