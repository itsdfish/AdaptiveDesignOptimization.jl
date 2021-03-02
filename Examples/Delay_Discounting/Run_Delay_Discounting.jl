#######################################################################################
#                                 Load Packages
#######################################################################################
# set the working directory to the directory in which this file is contained
cd(@__DIR__)
# load the package manager
using Pkg
# activate the project environment
Pkg.activate("../../")
using Revise, AdaptiveDesignOptimization, Random, UtilityModels, Distributions
include("Delay_Discounting.jl")
#######################################################################################
#                                  Define Model
#######################################################################################
Random.seed!(12034)
prior = [Uniform(-5, 5), Uniform(-5, 50)]

model = Model(;prior, loglike)

parm_list = (κ = range(-5, 0, length=50) .|> x->10^x, 
   τ = range(0, 5, length=11)[2:end])

# parm_list = (κ = [.1], 
#    τ = [.2,.5])

design_list = (
    t_ss = [0.0], 
    t_ll = [0.43, 0.714, 1, 2, 3,
        4.3, 6.44, 8.6, 10.8, 12.9,
        17.2, 21.5, 26, 52, 104,
        156, 260, 520], 
    r_ss = 12.5:12.5:787.5,
    r_ll = [800.0]
)

# design_list = (
#     t_ss = [0.0], 
#     t_ll = [5.0, 10.0], 
#     r_ss = [12.0, 20.0],
#     r_ll = [80.0]
# )

data_list = (choice=[true, false],)
#######################################################################################
#                              Simulate Experiment
#######################################################################################
using DataFrames
true_parms = (κ=.12, τ=1.5)
n_trials = 100
optimizer = Optimizer(;design_list, parm_list, data_list, model);
design = optimizer.best_design
df = DataFrame(design=Symbol[], trial=Int[], mean_κ=Float64[], mean_τ=Float64[],
    std_κ=Float64[], std_τ=Float64[])
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
@df df plot(:trial, :mean_κ, xlabel="trial", ylabel="mean κ", group=:design, grid=false)
hline!([true_parms.κ], label="true")

@df df plot(:trial, :mean_τ, xlabel="trial", ylabel="mean τ", group=:design, grid=false)
hline!([true_parms.τ], label="true")

@df df plot(:trial, :std_κ, xlabel="trial", ylabel="σ of κ", grid=false, group=:design, ylims=(0,.3))

@df df plot(:trial, :std_τ, xlabel="trial", ylabel="σ of τ", grid=false, group=:design, ylims=(0,2))
