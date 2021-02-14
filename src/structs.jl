using Distributions, Parameters
import Base.Iterators: product

abstract type AbstractTask end

mutable struct Model{F<:Function,P}
    prior::P
    loglike::F
end

function Model(args...; prior, loglike, kwargs...)
    return Model(prior, x->loglike(x..., args...; kwargs...))
end

"""
*Optimizer*

`Optimizer` constructs a model object for adaptive design optimization

- `task`: utility curvature for gains
- `model`: utility curvature for losses
- `grid_design`: probability weighting parameter for gains 
- `grid_parms`: probability weighting parameter for losses
- `grid_response`: loss aversion parameter

Constructor
````julia
Optimizer(;task, model, grid_design, grid_parms, grid_response)
````
"""
mutable struct Optimizer{T,M<:Model,T1,T2,T3,T4}
    task::T
    model::M
    design_grid::T1
    parm_grid::T2
    data_grid::T3
    likelihoods::T4
end

function Optimizer(;task=nothing, model, design_grid, parm_grid, data_grid)
    dims = [length.(values(data_grid))...; 
        length.(values(parm_grid))...;  
        length.(values(design_grid))...]
    likelihoods = fill(0.0, dims...)
    return Optimizer(task, model, design_grid, parm_grid, data_grid, likelihoods)
end

function compute_likelihoods!(optimizer)
    @unpack model, design_grid, parm_grid, data_grid = optimizer
    grid = product(data_grid..., parm_grid..., design_grid...)
    optimizer.likelihoods .= map(x->model.loglike(x) |> exp, grid)
end

function compute_marg_entropy()

end

function compute_cond_entropy()

end




discount(t, κ) = 1/(1 + κ*t)

function loglike(data, κ, τ, t_ss, t_ll, r_ss, r_ll)
    u_ll = r_ll * discount(t_ll, κ)
    u_ss = r_ss * discount(t_ss, κ)
    p = 1/(1 + exp(-τ * (u_ll - u_ss)))
    p = max(p, 1e-10)
    LL = data ? log(p) : log(1 - p)
    return LL
end

prior = [Exponential(.1), Uniform(0, 5)]

model = Model(;prior, loglike)

parm_grid = (κ = range(-5, 0, length=50) .|> x->10^x, 
   τ = range(0, 5, length=11)[2:end])

   design_grid = (
    t_ss = [0.0], 
    t_ll = [0.43, 0.714, 1, 2, 3,
        4.3, 6.44, 8.6, 10.8, 12.9,
        17.2, 21.5, 26, 52, 104,
        156, 260, 520], 
    r_ss = 12.5:12.5:800,
    r_ll = [800.0])

data_grid = (choice=[true, false],)

optimizer = Optimizer(;design_grid, parm_grid, data_grid, model)
