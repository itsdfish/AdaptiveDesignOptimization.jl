function prior_probs(model::Model, parm_grid)
    return prior_probs(model.prior, parm_grid)
end

function prior_probs(prior, parm_grid)
    dens = [mapreduce((θ,d)->pdf(d, θ), *, g, prior) for g in parm_grid]
    return dens/sum(dens)
end

function prior_probs(prior::Nothing, parm_grid)
    return fill(1/length(parm_grid), size(parm_grid))
end

function loglikelihood(model::Model, design_grid, parm_grid, data_grid, model_type, model_state)
    return loglikelihood(model.loglike, design_grid, parm_grid, data_grid)
end

function loglikelihood(model::Model, design_grid, parm_grid, data_grid, model_type::Dyn, model_state)
    return loglikelihood(model.loglike, design_grid, parm_grid, data_grid, model_state)
end

function loglikelihood(loglike, design_grid, parm_grid, data_grid)
    LLs = zeros(length(parm_grid), length(design_grid), length(data_grid))
    for (d, data) in enumerate(data_grid)
        for (k,design) in enumerate(design_grid)
            for (p,parms) in enumerate(parm_grid)
                LLs[p,k,d] = loglike(parms..., design..., data...) 
            end
        end
    end
    return LLs
end

function loglikelihood(loglike, design_grid, parm_grid, data_grid, model_state)
    LLs = zeros(length(parm_grid), length(design_grid), length(data_grid))
    i = 0
    for (d, data) in enumerate(data_grid)
        for (k,design) in enumerate(design_grid)
            for (p,parms) in enumerate(parm_grid)
                i += 1
                LLs[p,k,d] = loglike(parms..., design..., data..., model_state[i]) 
            end
        end
    end
    return LLs
end

function loglikelihood!(optimizer) 
    @unpack model, log_like, design_grid, parm_grid, data_grid, model_state = optimizer
    return loglikelihood!(model.loglike, log_like, design_grid, parm_grid, data_grid, model_state)
end

function loglikelihood!(loglike, log_like, design_grid, parm_grid, data_grid, model_state)
    i = 0
    for (d, data) in enumerate(data_grid)
        for (k,design) in enumerate(design_grid)
            for (p,parms) in enumerate(parm_grid)
                i += 1
                log_like[p,k,d] = loglike(parms..., design..., data..., model_state[i]) 
            end
        end
    end
    return nothing
end

function marginal_log_like!(optimizer)
    @unpack marg_log_like,log_like,log_post = optimizer
    marg_log_like .= marginal_log_like(log_post, log_like)
end

function marginal_log_like(log_post, log_like)
    return logsumexp(log_post .+ log_like, dims=1)
end

function marginal_posterior(optimizer)
    @unpack posteriors = optimizer
    return map(d->sum(posterior, dims=d), ndims(posterior):-1:1)
end

function conditional_entropy(entropy, post)
    return entropy'*post
end

function conditional_entropy!(optimizer)
    @unpack cond_entropy, entropy,log_post = optimizer
    post = exp.(log_post)
    cond_entropy .= conditional_entropy(entropy, post)
end

function compute_entropy(log_like)
    return -1*sum(exp.(log_like) .* log_like, dims=3)[:,:]
end

function marginal_entropy!(optimizer::Optimizer)
    @unpack marg_entropy,marg_log_like = optimizer
    marg_entropy .= marginal_entropy(marg_log_like)
end

function marginal_entropy(marg_log_like)
    return -sum(exp.(marg_log_like).*marg_log_like, dims=3)[:]
end

function mutual_information(marg_entropy, cond_entropy)
    return marg_entropy .- cond_entropy
end

function mutual_information!(optimizer)
    @unpack mutual_info,marg_entropy,cond_entropy = optimizer
    mutual_info .= mutual_information(marg_entropy, cond_entropy)
    return nothing
end

function get_best_design(optimizer)
    @unpack best_design,design_names = optimizer
    return NamedTuple{design_names}(best_design)
end

function find_best_design!(optimizer)
    @unpack design_names,mutual_info,design_grid = optimizer
    best_design = find_best_design(mutual_info, design_grid, design_names)
    optimizer.best_design = best_design
    return best_design
end

function find_best_design(mutual_info, design_grid, design_names)
    _,best = findmax(mutual_info)
    best_design = design_grid[best]
    return best_design
end

function find_best_design!(optimizer::Optimizer{A}) where {A<:Rand}
    @unpack design_grid,design_names = optimizer
    best_design = rand(design_grid)
    optimizer.best_design = best_design
    return best_design
end

function update_posterior!(optimizer, data)
    @unpack log_post,design_grid,data_grid,log_like,best_design = optimizer
    dn = find_index(design_grid, best_design)
    for datum in data
        da = find_index(data_grid, (datum,))
        log_post .+= log_like[:,dn,da]
    end
    log_post .-= logsumexp(log_post)
    return nothing 
end

function update!(optimizer, data)
    update_posterior!(optimizer, data)
    marginal_log_like!(optimizer)
    marginal_entropy!(optimizer)
    conditional_entropy!(optimizer)
    mutual_information!(optimizer)
    best_design = find_best_design!(optimizer)
    return best_design
end

function update!(optimizer::Optimizer{A,MT}, data, args...; kwargs...) where {A,MT<:Dyn}
    update_posterior!(optimizer, data)
    update_states!(optimizer, data, args...; kwargs...)
    loglikelihood!(optimizer)
    marginal_log_like!(optimizer)
    marginal_entropy!(optimizer)
    conditional_entropy!(optimizer)
    mutual_information!(optimizer)
    best_design = find_best_design!(optimizer)
    return best_design
end

function update!(optimizer::Optimizer{A}, data) where {A<:Rand}
    update_posterior!(optimizer, data)
    best_design = find_best_design!(optimizer)
    return best_design
end

function update_states!(optimizer, obs_data, args...; kwargs...)
    @unpack model_state, data_grid, design_grid, parm_grid, update_state! = optimizer
    for (d, data) in enumerate(data_grid)
        for (k,design) in enumerate(design_grid)
            for (p,parms) in enumerate(parm_grid)
                update_state!(model_state, parms, design, data, obs_data, args...; kwargs...)
            end
        end
    end
    return nothing
end

function to_grid(vals::NamedTuple)
    k = keys(vals)
    v = product(vals...) |> collect
    return k,v
end

function to_grid(vals)
    k = [Symbol(string("v",i)) for i in 1:length(vals[1])]
    return (k...,),vals
end

to_grid(vals::Tuple) = vals

function find_index(grid, val)
    i = 0
    for g in grid 
        i += 1
        if g == val 
            return i
        end
    end 
    return i
end

function mean_post(optimizer)
    @unpack log_post,parm_grid = optimizer
    post = exp.(log_post)
    return mean_post(post, parm_grid)
end

function mean_post(post, parm_grid)
    return mapreduce((p,v)->p.*v, .+, post, parm_grid)
end

function std_post(post, parm_grid)
    mu = mean_post(post, parm_grid)
    return mapreduce((p,v)->p.*(v .- mu).^2, .+, post, parm_grid) .|> sqrt
end

function std_post(optimizer)
    @unpack log_post,parm_grid = optimizer
    post = exp.(log_post)
    return std_post(post, parm_grid)
end

function create_state(model_type::Dyn, T, dims, args...; kwargs...)
    state = fill(T(args...; kwargs...), dims)
    return state .= deepcopy.(state)
end

function create_state(model_type::Stat, T, dims, args...; kwargs...)
    return nothing
end


