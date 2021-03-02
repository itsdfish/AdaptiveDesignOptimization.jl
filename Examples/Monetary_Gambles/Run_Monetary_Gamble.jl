#######################################################################################
#                                 Load Packages
#######################################################################################
# set the working directory to the directory in which this file is contained
cd(@__DIR__)
# load the package manager
using Pkg
# activate the project environment
Pkg.activate("../../")
using AdaptiveDesignOptimization, Random, UtilityModels, Distributions
include("TAX_Model.jl")
Random.seed!(25974)
#######################################################################################
#                                  Define Parameters
#######################################################################################

# model with default uniform prior
model = Model(;loglike)

parm_list = (
    δ = range(-2, 2, length=10),
    β = range(.5, 1.5, length=10),
    γ = range(.5, 1.2, length=10),
    θ = range(.5, 3, length=10)
)

dist = Normal(0,10)
n_vals = 3
n_choices = 2
design_vals = map(x->random_design(dist, n_vals, n_choices), 1:1000)
filter!(x->abs_zscore(x) ≤ .4, design_vals)
design_names = (:p1,:v1,:p2,:v2)
design_list = (design_names,design_vals[1:100])

data_list = (choice=[true, false],)
#######################################################################################
#                              Optimize Experiment
#######################################################################################
using DataFrames
true_parms = (δ=-1.0, β=1.0, γ=.7, θ=1.5)
n_trials = 100
optimizer = Optimizer(;design_list, parm_list, data_list, model)
design = optimizer.best_design
df = DataFrame(design=Symbol[], trial=Int[], mean_δ=Float64[], mean_β=Float64[],
    mean_γ=Float64[], mean_θ=Float64[], std_δ=Float64[], std_β=Float64[],
    std_γ=Float64[], std_θ=Float64[])
new_data = [:optimal, 0, mean_post(optimizer)..., std_post(optimizer)...]
push!(df, new_data)

for trial in 1:n_trials
    data = simulate(true_parms..., design...)
    design = update!(optimizer, data)
    new_data = [:optimal, trial, mean_post(optimizer)..., std_post(optimizer)...]
    push!(df, new_data)
end
#######################################################################################
#                              Random Experiment
#######################################################################################
randomizer = Optimizer(;design_list, parm_list, data_list, model, design_type=Randomize);
design = randomizer.best_design
new_data = [:random, 0, mean_post(randomizer)..., std_post(randomizer)...]
push!(df, new_data)

for trial in 1:n_trials
    data = simulate(true_parms..., design...)
    design = update!(randomizer, data)
    new_data = [:random, trial, mean_post(randomizer)..., std_post(randomizer)...]
    push!(df, new_data)
end
#######################################################################################
#                                 Plot Results
#######################################################################################
using StatsPlots
@df df plot(:trial, :mean_δ, xlabel="trial", ylabel="mean δ", group=:design, grid=false)
hline!([true_parms.δ], label="true")

@df df plot(:trial, :mean_β, xlabel="trial", ylabel="mean β", group=:design, grid=false)
hline!([true_parms.β], label="true")

@df df plot(:trial, :mean_γ, xlabel="trial", ylabel="mean γ", group=:design, grid=false)
hline!([true_parms.γ], label="true")

@df df plot(:trial, :mean_θ, xlabel="trial", ylabel="mean θ", group=:design, grid=false)
hline!([true_parms.θ], label="true")

@df df plot(:trial, :std_δ, xlabel="trial", ylabel="σ of δ", grid=false, group=:design, ylims=(0,1.5))

@df df plot(:trial, :std_β, xlabel="trial", ylabel="σ of β", grid=false, group=:design, ylims=(0,.5))

@df df plot(:trial, :std_γ, xlabel="trial", ylabel="σ of γ", grid=false, group=:design, ylims=(0,.5))

@df df plot(:trial, :std_θ, xlabel="trial", ylabel="σ of θ", grid=false, group=:design, ylims=(0,.8))