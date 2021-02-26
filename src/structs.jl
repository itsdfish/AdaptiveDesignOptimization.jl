import Base.Iterators: product

"""
**Model**

Creates a model object containing a log likelihood function and prior distributions

- `prior`: a vector of distribution objects for each parameter. A uniform prior is used if no
prior is passed.
- `loglike`: a function that computes the log likelihood 

Constructor 

````julia 
Model(args...; prior=nothing, loglike, kwargs...)
````
"""
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

- `model`: a model object
- `grid_design`:a grid of design parameters 
- `grid_parms`: a grid of model parameters
- `grid_data`: a grid of data
- `log_like`: a three dimensional array of precomputed log likelihoods
- `marg_log_like`: a two dimensional array containing marginal log likelihoods for design and data
- `priors`: a multidimensional array of prior probabilities for parameters
- `log_post`: a one dimensional array of log posterior probabilities for parameters
- `entropy`: a two dimensional array of entropy values for parameter and design combinations
- 

Constructor

````julia
Optimizer(;task, model, grid_design, grid_parms, grid_response)
````
"""
mutable struct Optimizer{M<:Model,T1,T2,T3,T4,T5,T6,T7,T8,T9}
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
    parm_names::T8
    design_names::T9
end

function Optimizer(;model, design_list, parm_list, data_list)
    design_names,design_grid = to_grid(design_list)
    parm_names,parm_grid = to_grid(parm_list)
    _,data_grid = to_grid(data_list)
    log_like = loglikelihood(model, design_grid, parm_grid, data_grid)
    priors = prior_probs(model, parm_grid)
    post = priors[:]
    log_post = log.(post)
    entropy = compute_entropy(log_like)
    marg_log_like = marginal_log_like(log_post, log_like)
    marg_entropy = marginal_entropy(marg_log_like)
    cond_entropy = conditional_entropy(entropy, post)
    mutual_info = mutual_information(marg_entropy, cond_entropy)
    best_design = find_best_design(mutual_info, design_grid, design_names)
    return Optimizer(model, design_grid, parm_grid, data_grid, log_like,
        marg_log_like, priors, log_post, entropy, marg_entropy, cond_entropy, 
        mutual_info, best_design, parm_names, design_names)
end


mutable struct Randomizer{M<:Model,T1,T2,T3,T4,T5,T6,T7}
    model::M
    design_grid::T1
    parm_grid::T2
    data_grid::T3
    log_like::Array{Float64,3}
    priors::T4
    log_post::Vector{Float64}
    best_design::T5
    parm_names::T6
    design_names::T7
end

function Randomizer(;model, design_list, parm_list, data_list)
    design_names,design_grid = to_grid(design_list)
    parm_names,parm_grid = to_grid(parm_list)
    _,data_grid = to_grid(data_list)
    log_like = loglikelihood(model, design_grid, parm_grid, data_grid)
    priors = prior_probs(model, parm_grid)
    post = priors[:]
    log_post = log.(post)
    best_design = rand(design_grid)
    return Randomizer(model, design_grid, parm_grid, data_grid, log_like,
        priors, log_post, best_design, parm_names, design_names)
end