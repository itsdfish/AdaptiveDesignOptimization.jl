using Distributions, Parameters, StatsFuns
import Base.Iterators: product
mutable struct Model{F<:Function,P}
    prior::P
    loglike::F
end

function Model(args...; prior=nothing, loglike, kwargs...)
    return Model(prior, (x...)->loglike(x..., args...; kwargs...))
end

"""
*Optimizer*

`Optimizer` constructs a model object for adaptive design optimization

- `model`: utility curvature for losses
- `grid_design`: probability weighting parameter for gains 
- `grid_parms`: probability weighting parameter for losses
- `grid_response`: loss aversion parameter

Constructor
````julia
Optimizer(;task, model, grid_design, grid_parms, grid_response)
````
"""
mutable struct Optimizer{M<:Model,T1,T2,T3,T4,T5,T6,T7}
    model::M
    design_grid::T1
    parm_grid::T2
    data_grid::T3
    log_like::Array{Float64,3}
    marg_log_like::T4
    priors::T5
    log_post::Vector{Float64}
    entropy::T6
    marg_entropy::Vector{Float64}
    cond_entropy::Vector{Float64}
    mutual_info::Vector{Float64}
    best_design::T7
end

function Optimizer(;model, design_grid, parm_grid, data_grid)
    parm_grid = product(parm_grid...) |> collect
    design_grid = product(design_grid...) |> collect
    data_grid = product(data_grid...) |> collect
    log_like = loglikelihood(model, design_grid, parm_grid, data_grid)
    priors = prior_probs(model, parm_grid)
    post = priors[:]
    log_post = log.(post)
    entropy = compute_entropy(log_like)
    marg_log_like = marginal_log_like(log_post, log_like)
    marg_entropy = marginal_entropy(marg_log_like)
    cond_entropy = conditional_entropy(entropy, post)
    mutual_info = mutual_information(marg_entropy, cond_entropy)
    best_design = get_best_design(mutual_info, design_grid)
    return Optimizer(model, design_grid, parm_grid, data_grid, log_like,
        marg_log_like, priors, log_post, entropy, marg_entropy, cond_entropy, 
        mutual_info, best_design)
end


mutable struct Randomizer{M<:Model,T1,T2,T3,T4,T5}
    model::M
    design_grid::T1
    parm_grid::T2
    data_grid::T3
    log_like::Array{Float64,3}
    priors::T4
    log_post::Vector{Float64}
    best_design::T5
end

function Randomizer(;model, design_grid, parm_grid, data_grid)
    parm_grid = product(parm_grid...) |> collect
    design_grid = product(design_grid...) |> collect
    data_grid = product(data_grid...) |> collect
    log_like = loglikelihood(model, design_grid, parm_grid, data_grid)
    priors = prior_probs(model, parm_grid)
    post = priors[:]
    log_post = log.(post)
    best_design = rand(design_grid)
    return Randomizer(model, design_grid, parm_grid, data_grid, log_like,
        priors, log_post, best_design)
end