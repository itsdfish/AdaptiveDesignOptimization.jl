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
mutable struct Optimizer{M<:Model,T1,T2,T3,T4,T5,T6}
    model::M
    design_grid::T1
    parm_grid::T2
    data_grid::T3
    log_like::Array{Float64,3}
    marg_log_like::T4
    priors::T5
    log_post::Vector{Float64}
    entropy::T6
    marginal_entropy::Vector{Float64}
    conditional_entropy::Vector{Float64}
end

function Optimizer(;model, design_grid, parm_grid, data_grid)
    log_like = compute_log_like(model, design_grid, parm_grid, data_grid)
    priors = compute_prior(model, parm_grid)
    log_post = log.(priors[:])
    entropy = compute_entropy(log_like)
    marginal_entropy = nothing
    conditional_entropy = nothing
    return Optimizer(model, design_grid, parm_grid, data_grid, log_like,
        priors, entropy, marginal_entropy, conditional_entropy)
end

function compute_prior(model::Model, parm_grid)
    return compute_prior(model.prior, parm_grid)
end

function compute_prior(prior, parm_grid)
    dens = [mapreduce((θ,d)->pdf(d, θ), *, g, prior) for g in parm_grid]
    return dens/sum(dens)
end

function compute_prior(prior::Nothing, parm_grid)
    return fill(1/length(parm_grid), size(parm_grid))
end

function compute_log_like(model::Model, design_grid, parm_grid, data_grid)
    return compute_log_like(model.loglike, design_grid, parm_grid, data_grid)
end

function compute_log_like(loglike, design_grid, parm_grid, data_grid)
    LLs = zeros(length(parm_grid), length(design_grid), length(data_grid))
    for (d, datum) in enumerate(data_grid)
        for (k,design) in enumerate(design_grid)
            for (p,parm) in enumerate(parm_grid)
                LLs[p,k,d] = loglike(parm..., design..., datum...) 
            end
        end
    end
    return LLs
end

function compute_marg_log_like(optimizer)
    @unpack log_like, log_post = optimizer
    return compute_marg_log_like(log_post, log_like)
end

function compute_marg_log_like(log_post, log_like)
    return logsumexp(log_post .+ log_like, dims=1)
end

function compute_marg_log_like!(optimizer)
    optimizer.marg_like .= compute_marg_log_like(optimizer) 
    return nothing
end

function compute_marg_post(optimizer)
    @unpack posteriors = optimizer
    return map(d->sum(posterior, dims=d), ndims(posterior):-1:1)
end

function compute_cond_entropy(entropy, post)
    return entropy'*post
end

function compute_entropy(log_like)
    return -1*sum(exp.(log_like) .* log_like, dims=3)[:,:]
end

function compute_marg_entropy!(optimizer::Optimizer)
    @unpack marg_log_like = optimizer
    return compute_marg_entropy!(marg_log_like)
end

function compute_marg_entropy!(marg_log_like)
    return -sum(exp.(marg_log_like).*marg_log_like, dims=3)[:]
end

function update!()
    #update posterior
    #

end

function update_posterior!(optimizer, data)
    @unpack log_post = optimizer
    da = findfirst(x->x == data, data_grid)
    dn = find
    log_post .+= log_like[:,dn,da]
    log_post .-= logsumexp(log_post)
    return nothing 
end

function find_index(grid, val)
    i = 1
    for g in grid 
        if g == val 
            return i
        end
        i += 1
    end 
    return i
end
