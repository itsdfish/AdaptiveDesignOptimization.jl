import Base.Iterators: product

abstract type DesignType end

struct Opt <: DesignType
end
struct Rand <: DesignType
end

const Optimize = Opt()
const Randomize = Rand()

abstract type ModelType end

struct Stat <: DesignType
end
struct Dyn <: DesignType
end

const Static = Stat()
const Dynamic = Dyn()

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
*Examples*

```julia
# Default uninform prior
loglike(θ, design, data) = ...

model = Model(;loglike)
```

```julia
# Custom priors
prior = [Beta(5,5),...]
loglike(θ, design, data) = ...

model = Model(;loglike, prior)
```
"""
mutable struct Model{F <: Function, P}
    prior::P
    loglike::F
end

function Model(args...; prior = nothing, loglike, kwargs...)
    return Model(prior, (x...) -> loglike(x..., args...; kwargs...))
end

"""
    Optimizer(;task, model, grid_design, grid_parms, grid_response)

`Optimizer` constructs a model object for adaptive design optimization

# Fields 

- `model`: a model object
- `grid_design`:a grid of design parameters 
- `grid_parms`: a grid of model parameters
- `grid_data`: a grid of data
- `log_like`: a three dimensional array of precomputed log likelihoods
- `marg_log_like`: a two dimensional array containing marginal log likelihoods for design and data
- `priors`: a multidimensional array of prior probabilities for parameters
- `log_post`: a one dimensional array of log posterior probabilities for parameters
- `entropy`: a two dimensional array of entropy values for parameter and design combinations
- `marg_entropy`:
- `cond_entropy`:
- `mutual_info`:
- `best_design`:
- `parm_names`:
- `model_state`:
- `update_state!`:
"""
mutable struct Optimizer{A, MT, M <: Model, T1, T2, T3, T4, T5, T6, T7, T8, T9,
    T10, T11}
    design_type::A
    model_type::MT
    model::M
    design_grid::T1
    parm_grid::T2
    data_grid::T3
    log_like::Array{Float64, 3}
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
    model_state::T10
    update_state!::T11
end

function Optimizer(args...; design_type = Optimize, model, design_list, parm_list,
    data_list,
    state_type = nothing, model_type = Static, update_state! = nothing, kwargs...)
    design_names, design_grid = to_grid(design_list)
    parm_names, parm_grid = to_grid(parm_list)
    _, data_grid = to_grid(data_list)
    dims = map(length, (parm_grid, design_grid, data_grid))
    model_state = create_state(model_type, state_type, dims, args...; kwargs...)
    log_like =
        loglikelihood(model, design_grid, parm_grid, data_grid, model_type, model_state)
    priors = prior_probs(model, parm_grid)
    post = priors[:]
    log_post = log.(post)
    entropy = compute_entropy(log_like)
    marg_log_like = marginal_log_like(log_post, log_like)
    marg_entropy = marginal_entropy(marg_log_like)
    cond_entropy = conditional_entropy(entropy, post)
    mutual_info = mutual_information(marg_entropy, cond_entropy)
    best_design = find_best_design(mutual_info, design_grid, design_names)
    return Optimizer(design_type, model_type, model, design_grid, parm_grid,
        data_grid, log_like, marg_log_like, priors, log_post, entropy, marg_entropy,
        cond_entropy, mutual_info, best_design, parm_names, design_names,
        model_state, update_state!)
end
