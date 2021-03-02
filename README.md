# AdaptiveDesignOptimization

This package is a grid-based approach for performing Bayesian adaptive design optimization. After each observation, the optimizer chooses an experimental design that maximizes mutual information between model parameters and design parameters. In so doing, the optimizer selects designs that minimize the variance in the posterior distribution of model parameters.
# Example

In this example, we will optimize a decision making experiment for the model called Transfer of Attention Exchange (TAX; Birnbaum, 2008). Additional examples can be found in the folder titled Examples. sds

```julia 
using AdaptiveDesignOptimization, Random, UtilityModels, Distributions
include("TAX_Model.jl")
Random.seed!(25974)
```

## Define Model

The model object contains a log likelihood function and optional prior distributions. Unless an array of distribution objects is passed as `prior`, uniform distributions will be used by default. Arguments in the log likelihood function must be ordered as follows:

- loglike(model_parameters..., design_parameters..., data..., args...; kwargs...)

`args...` and `kwargs...` are optional arguments that may be preloaded through the `Model` constructor. Enter 

```julia 
? Model
``` 
for additional details. The likelihood function for the TAX model (see `TAX_Model.jl`) is defined as:

```julia
function loglike(δ, β, γ, θ, pa, va, pb, vb, data)
    eua,eub = expected_utilities(δ, β, γ, θ, pa, va, pb, vb)
    p = choice_prob(eua, eub, θ)
    p = max(p, eps())
    return logpdf(Bernoulli(p), data)
end
```

The model object is contructed with default uniform prior distributions. 

```julia 
# model with default uniform prior
model = Model(;loglike)
```
## Define Parameters

Define a `NamedTuple` of parameter value ranges. Note that the parameters listed in the same order that they appear in `loglike`.

```julia
parm_list = (
    δ = range(-2, 2, length=10),
    β = range(.5, 1.5, length=10),
    γ = range(.5, 1.2, length=10),
    θ = range(.5, 3, length=10)
)
```

The experiment will consist of two gambles with three outcomes each. The number of dimensions in the design space is large (2X2X3X3 = 36). In this case, we will sample random gambles and select a subset of 100 with high distributional overlap. In this case, the `design_list` will be a `Tuple` of design names and design values. 

```julia
# outcome distribution
dist = Normal(0,10)
n_vals = 3
n_choices = 2
design_vals = map(x->random_design(dist, n_vals, n_choices), 1:1000)
# select gambles with overlapping distributions
filter!(x->abs_zscore(x) ≤ .4, design_vals)
design_names = (:p1,:v1,:p2,:v2)
design_list = (design_names,design_vals[1:100])
```

Lastly, we will define a list for the data. The value true indicates gamble A was choosen and false indicates gamble B was chosen.

```julia 
data_list = (choice=[true, false],)
```
## Optimize Exeriment


# References

* Birnbaum, M. H., & Chavez, A. (1997). Tests of theories of decision making: Violations of branch independence and distribution   independence. Organizational Behavior and human decision Processes, 71(2), 161-194. Birnbaum, M. H. (2008). New paradoxes of risky decision making. Psychological review, 115(2), 463.

* Myung, J. I., Cavagnaro, D. R., and Pitt, M. A. (2013). A tutorial on adaptive design optimization. Journal of Mathematical Psychology, 57, 53–67.

* Yang, J., Pitt, M. A., Ahn, W., & Myung, J. I. (2020). ADOpy: A Python Package for Adaptive Design Optimization. Behavior Research Methods, 1--24. https://doi.org/10.3758/s13428-020-01386-4

