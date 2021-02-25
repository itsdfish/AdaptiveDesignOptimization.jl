function loglike(δ, β, γ, θ, pa, va, pb, vb, data)
    eua,eub = expected_utilities(δ, β, γ, θ, pa, va, pb, vb)
    p = choice_prob(eua, eub, θ)
    p = p == 1 ? 1 - eps() : p
    p = p == 0 ? eps() : p
    LL = data ? log(p) : log(1 - p)
    return LL
end

function choice_prob(eua, eub, θ)
    return 1/(1 + exp(-θ*(eua - eub)))
end

function simulate(δ, β, γ, θ, pa, va, pb, vb)
    eua,eub = expected_utilities(δ, β, γ, θ, pa, va, pb, vb)
    p = choice_prob(eua, eub, θ)
    return rand() ≤ p ? true : false
end

function expected_utilities(δ, β, γ, θ, pa, va, pb, vb)
    model = TAX(;γ, δ, β)
    gambleA = Gamble(;p=pa, v=va)
    gambleB = Gamble(;p=pb, v=vb)
    eua = mean(model, gambleA)
    eub = mean(model, gambleB)
    return eua,eub
end

function random_design(outcome_dist, n_vals, n_choices)
    design = Vector{Vector{Float64}}()
    for _ in 1:n_choices
        p = rand(Dirichlet(n_vals, 1))
        push!(design, p)
        v = rand(outcome_dist, n_vals)
        push!(design, v)
    end
    return design
end

function abs_zscore(x)
    model = ExpectedUtility(1)
    gambleA = Gamble(;p=x[1], v=x[2])
    gambleB = Gamble(;p=x[3], v=x[4])
    μa = mean(model, gambleA)
    μb = mean(model, gambleB)
    σa = var(model, gambleA)
    σb = var(model, gambleB)
    return abs((μa - μb)/sqrt(σa + σb))
end

    
# using Revise, Turing, Random, MCMCChains

# # Set the true probability of heads in a coin.
# p_true = 0.5

# # Iterate from having seen 0 observations to 100 observations.
# Ns = 0:100

# # Draw data from a Bernoulli distribution, i.e. draw heads or tails.
# Random.seed!(12)
# data = rand(Bernoulli(p_true), last(Ns))

# # Declare our Turing model.
# @model function coinflip(y)
#     # Our prior belief about the probability of heads in a coin.
#     p ~ Beta(1, 1)

#     # The number of observations.
#     N = length(y)
#     for n in 1:N
#         # Heads or tails of a coin are drawn from a Bernoulli distribution.
#         y[n] ~ Bernoulli(p)
#     end
# end

# # Settings of the Hamiltonian Monte Carlo (HMC) sampler.
# iterations = 1000
# ϵ = 0.05
# τ = 10

# # Start sampling.
# chain = sample(coinflip(data), HMC(ϵ, τ), iterations)


# lpfun = function f(chain::Chains, dist, chains) # function to compute the logpdf values
#     niter, nparams, nchains = size(chain)
#     lp = zeros(niter + nchains) # resulting logpdf values
#     for i = 1:nparams
#         lp += map(p ->logpdf(dist(p...), data), Array(chain[:,i,:]))
#     end
#     return lp
# end

# lpfun = function f(chain::Chains) # function to compute the logpdf values
#     niter, _, nchains = size(chain)
#     parms = (Symbol.(chain.name_map.parameters)...,)
#     lp = zeros(niter*nchains, 1) # resulting logpdf values
#     for parm in parms
#         lp .+= map(p -> logpdf(Bernoulli(p...), data), Array(chain[parm]))
#     end
#     return lp
# end

# DIC, pD = dic(chain, lpfun)
# ​