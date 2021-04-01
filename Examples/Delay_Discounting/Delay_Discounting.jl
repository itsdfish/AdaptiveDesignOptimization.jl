discount(t, κ) = 1/(1 + κ*t)

function prob(κ, τ, t_ss, t_ll, r_ss, r_ll)
    u_ll = r_ll * discount(t_ll, κ)
    u_ss = r_ss * discount(t_ss, κ)
    return 1/(1 + exp(-τ * (u_ll - u_ss)))
end

function loglike(κ, τ, t_ss, t_ll, r_ss, r_ll, data)
    p = prob(κ, τ, t_ss, t_ll, r_ss, r_ll)
    p = max(p, eps())
    p = min(1-eps(), p)
    LL = data ? log(p) : log(1 - p)
    # println(" choice ", data, " kappa ", κ, " tau ", τ, " t_ss ", t_ss,
    #     " t_ll ", t_ll, " r_ss ", r_ss, " r_ll ", r_ll, " LL ", LL)
    return LL
end

function simulate(κ, τ, t_ss, t_ll, r_ss, r_ll)
    p = prob(κ, τ, t_ss, t_ll, r_ss, r_ll)
    return rand() ≤ p ? true : false
end
