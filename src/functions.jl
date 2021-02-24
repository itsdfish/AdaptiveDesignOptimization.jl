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

function loglikelihood(model::Model, design_grid, parm_grid, data_grid)
    return loglikelihood(model.loglike, design_grid, parm_grid, data_grid)
end

function loglikelihood(loglike, design_grid, parm_grid, data_grid)
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

function get_best_design!(optimizer)
    @unpack mutual_info,design_grid = optimizer
    best_design = get_best_design(mutual_info, design_grid)
    optimizer.best_design = best_design
    return best_design
end

function get_best_design(mutual_info, design_grid)
    _,best = findmax(mutual_info)
    best_design = design_grid[best]
    return best_design
end

function get_best_design!(randomizer::Randomizer)
    @unpack design_grid = randomizer
    best_design = rand(design_grid)
    randomizer.best_design = best_design
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
    best_design = get_best_design!(optimizer)
    return best_design
end

function update!(randomizer::Randomizer, data)
    update_posterior!(randomizer, data)
    best_design = get_best_design!(randomizer)
    return best_design
end

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