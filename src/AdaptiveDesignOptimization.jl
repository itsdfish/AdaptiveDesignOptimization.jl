module AdaptiveDesignOptimization
    using Distributions, Parameters, StatsFuns
    export Optimizer, Model, Optimize, Randomize
    export Dynamic, Static
    export update!, mean_post, std_post, get_best_design

    include("structs.jl")
    include("functions.jl")
end
