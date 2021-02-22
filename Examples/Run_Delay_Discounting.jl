cd(@__DIR__)
include("../src/structs.jl")

discount(t, κ) = 1/(1 + κ*t)

function loglike(κ, τ, t_ss, t_ll, r_ss, r_ll, data)
    u_ll = r_ll * discount(t_ll, κ)
    u_ss = r_ss * discount(t_ss, κ)
    p = 1/(1 + exp(-τ * (u_ll - u_ss)))
    p = p == 1 ? 1 - eps() : p
    p = p == 0 ? eps() : p
    LL = data ? log(p) : log(1 - p)
    # println(" choice ", data, " kappa ", κ, " tau ", τ, " t_ss ", t_ss,
    #     " t_ll ", t_ll, " r_ss ", r_ss, " r_ll ", r_ll, " LL ", LL)
    return LL
end

prior = [Uniform(-5, 5), Uniform(-5, 50)]

model = Model(;prior, loglike)

parm_grid = (κ = range(-5, 0, length=50) .|> x->10^x, 
   τ = range(0, 5, length=11)[2:end])

# parm_grid = (κ = [.1,.2,.7], 
#    τ = [.2,.5,.7])

design_grid = (
    t_ss = [0.0], 
    t_ll = [0.43, 0.714, 1, 2, 3,
        4.3, 6.44, 8.6, 10.8, 12.9,
        17.2, 21.5, 26, 52, 104,
        156, 260, 520], 
    r_ss = 12.5:12.5:787.5,
    r_ll = [800.0]
)

# design_grid = (
#     t_ss = [0.0], 
#     t_ll = [5.0, 10.0], 
#     r_ss = [12.0, 20.0],
#     r_ll = [80.0]
# )

data_grid = (choice=[true, false],)


# parm_grid = product(parm_grid...)
# design_grid = product(design_grid...)
# data_grid = product(data_grid...)


optimizer = Optimizer(;design_grid, parm_grid, data_grid, model);

