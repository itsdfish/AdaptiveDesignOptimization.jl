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

