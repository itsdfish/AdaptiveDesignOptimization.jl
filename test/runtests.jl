using SafeTestsets

@safetestset "Updating Prior" begin
    using Test, Distributions, AdaptiveDesignOptimization

    α = 5
    β = 5
    prior = [Beta(α, β)]
    loglike(θ, d, data) = logpdf(Bernoulli(θ), data)
    model = Model(;prior, loglike)

    parm_list = (
        θ = range(0, 1, length=200),
    )

    design_list = (d = [1,2],)

    data_list = (choice=[true, false],)

    randomizer = Optimizer(;design_list, parm_list, data_list, model,
        approach=Randomize)

    @test mean_post(randomizer)[1] ≈ mean(Beta(α,β)) atol = 5e-3
    @test std_post(randomizer)[1] ≈ std(Beta(α,β)) atol = 5e-3

    update!(randomizer, false)
    @test mean_post(randomizer)[1] ≈ mean(Beta(α,β+1)) atol = 5e-3
    @test std_post(randomizer)[1] ≈ std(Beta(α,β+1)) atol = 5e-3

    update!(randomizer, false)
    @test mean_post(randomizer)[1] ≈ mean(Beta(α,β+2)) atol = 5e-3
    @test std_post(randomizer)[1] ≈ std(Beta(α,β+2)) atol = 5e-3

    update!(randomizer, true)
    @test mean_post(randomizer)[1] ≈ mean(Beta(α+1,β+2)) atol = 5e-3
    @test std_post(randomizer)[1] ≈ std(Beta(α+1,β+2)) atol = 5e-3
end

@safetestset "Updating Default Prior" begin
    using Test, Distributions, AdaptiveDesignOptimization

    α = 1
    β = 1
    loglike(θ, d, data) = logpdf(Bernoulli(θ), data)
    model = Model(; loglike)

    parm_list = (
        θ = range(0, 1, length=200),
    )

    design_list = (d = [1,2],)

    data_list = (choice=[true, false],)

    randomizer = Optimizer(;design_list, parm_list, data_list, model,
        approach=Randomize)

    @test mean_post(randomizer)[1] ≈ mean(Beta(α,β)) atol = 5e-3
    @test std_post(randomizer)[1] ≈ std(Beta(α,β)) atol = 5e-3

    update!(randomizer, false)
    @test mean_post(randomizer)[1] ≈ mean(Beta(α,β+1)) atol = 5e-3
    @test std_post(randomizer)[1] ≈ std(Beta(α,β+1)) atol = 5e-3

    update!(randomizer, false)
    @test mean_post(randomizer)[1] ≈ mean(Beta(α,β+2)) atol = 5e-3
    @test std_post(randomizer)[1] ≈ std(Beta(α,β+2)) atol = 5e-3

    update!(randomizer, true)
    @test mean_post(randomizer)[1] ≈ mean(Beta(α+1,β+2)) atol = 5e-3
    @test std_post(randomizer)[1] ≈ std(Beta(α+1,β+2)) atol = 5e-3
end

@safetestset "Updating Prior Dynamic Model" begin
    using Test, Distributions, AdaptiveDesignOptimization

    struct State1 end

    α = 5
    β = 5
    prior = [Beta(α, β)]
    loglike(θ, d, data, args...) = logpdf(Bernoulli(θ), data)
    model = Model(;prior, loglike)

    parm_list = (
        θ = range(0, 1, length=200),
    )

    design_list = (d = [1,2],)

    data_list = (choice=[true, false],)

    update_state!(args...) = nothing 

    randomizer = Optimizer(;design_list, parm_list, data_list, model,
    update_state!, state_type=State1, model_type=Dynamic)

    @test mean_post(randomizer)[1] ≈ mean(Beta(α,β)) atol = 5e-3
    @test std_post(randomizer)[1] ≈ std(Beta(α,β)) atol = 5e-3

    update!(randomizer, false)
    @test mean_post(randomizer)[1] ≈ mean(Beta(α,β+1)) atol = 5e-3
    @test std_post(randomizer)[1] ≈ std(Beta(α,β+1)) atol = 5e-3

    update!(randomizer, false)
    @test mean_post(randomizer)[1] ≈ mean(Beta(α,β+2)) atol = 5e-3
    @test std_post(randomizer)[1] ≈ std(Beta(α,β+2)) atol = 5e-3

    update!(randomizer, true)
    @test mean_post(randomizer)[1] ≈ mean(Beta(α+1,β+2)) atol = 5e-3
    @test std_post(randomizer)[1] ≈ std(Beta(α+1,β+2)) atol = 5e-3
end

@safetestset "Dimension Check" begin
    using Test, Distributions, AdaptiveDesignOptimization

    loglike(μ, σ, data, args...) = logpdf(Normal(μ, σ), data)
    model = Model(; loglike)

    parm_list = (
        μ = range(0, 10, length=10),
        σ = range(1, 5, length=10),
    )

    design_list = (
        d = range(1, 5, length=10),
        z = range(1, 5, length=10),
        v = range(1, 5, length=10)
    )

    data_list = (
        choice=[true, false],
        v = 1:3    
    )

    optimizer = Optimizer(;design_list, parm_list, data_list, model)

    @test optimizer.parm_grid |> size == (10,10)
    @test optimizer.design_grid |> size == (10,10,10)
    @test optimizer.data_grid |> size == (2,3)
    @test optimizer.log_like |> size == (100,1000,6)
    @test optimizer.log_post |> size == (100,)
    @test optimizer.priors |> size == (10,10)
    @test optimizer.mutual_info |> size == (1000,)
    @test optimizer.marg_log_like |> size == (1,1000,6) 
    @test optimizer.marg_entropy |> size == (1000,)
    @test optimizer.cond_entropy |> size == (1000,)
end
