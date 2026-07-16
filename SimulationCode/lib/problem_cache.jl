## Functions required for asocial learning

# Calculate the value function with the fast relaxation approximation that the values reach the stable values much faster than the policy
function stable_value_exact(p::AbstractVector{eltype}, γ) where eltype
    n = length(p)
    A = [j==i ? -one(eltype) : j==i+1 ? γ*p[i] : j==i-1 ? γ*(1-p[i]) : zero(eltype) for i in 1:n, j in 1:n]
    B = zeros(eltype, n)
    B[end] = -p[n]
    lu(A) \ B
end

function stable_value_exact(p::AbstractVector{eltype}, n, i, γ) where eltype
    if i == 0
        zero(eltype)
    elseif i == n+1
        one(eltype)/γ
    else
        stable_value_exact(p, γ)[i]
    end
end

# Calculate the value function with the fast relaxation and small γ approximation that the values reach the stable values much faster than the policy
function stable_value_approx(p::AbstractVector{eltype}, n, i, γ) where eltype
    i==0 && return zero(eltype)
    i==n+1 && return one(eltype)/γ

    v = γ^(n-i)
    for j in i:n
        v *= p[j]
    end
    
    return v
end

function ACRHS_approx(p, k, n, i, γ)
    k[i]*(stable_value_approx(p, n, i+1, γ) - stable_value_approx(p, n, i-1, γ))*p[i]^2*(1-p[i])^2
end

function ACRHS(p, k, n, i, γ)
    k[i]*(stable_value_exact(p, n, i+1, γ) - stable_value_exact(p, n, i-1, γ))*p[i]^2*(1-p[i])^2
end

abstract type ProblemCache end
mutable struct UBProblemCache <: ProblemCache
    UBSL_k_caches::Vector{DerivativeCache}
    UBSL_pk_caches::Vector{DerivativeCache}
    AC_x_caches::Vector{DerivativeCache}
    AC_s_caches::Matrix{DerivativeCache}
end

function UBProblemCache(n, γ)
    UBSL_k_caches = [generate_G_derivative_funcs((x, p) -> x[i], n) for i in 1:n]
    UBSL_pk_caches = [generate_G_derivative_funcs((x, p) -> x[i]*p[i], n) for i in 1:n]
    AC_x_caches = [generate_G_derivative_funcs_num((x, p) -> ACRHS(p, x, n, i, γ), n) for i in 1:n]
    AC_s_caches = [generate_G_derivative_funcs_num((x, p) -> p[j]*ACRHS(p, x, n, i, γ) + p[i]*ACRHS(p, x, n, j, γ), n) for i in 1:n, j in 1:n]
    UBProblemCache(UBSL_k_caches, UBSL_pk_caches, AC_x_caches, AC_s_caches)
end

# Functions required for performance biased social learning
function expected_reward(p::AbstractVector{eltype}) where {eltype}
    n = length(p)
    numerator = one(eltype)
    denominator = one(eltype)
    prod = one(eltype)
    ninv = 1/n
    for k in 1:n-1
        prod *= (1-p[k])/p[k]
        numerator += (1-k*ninv)*prod
        denominator += prod
    end
    prod *= (1-p[n])/p[n]
    denominator += prod
    return numerator / denominator
end

function expected_reward(p::AbstractVector{eltype}, a::AbstractVector) where {eltype}
    n = length(p)
    numerator = one(eltype)
    denominator = one(eltype)
    prod = one(eltype)
    minus_sum = 0.
    for k in 1:n-1
        prod *= (1-p[k])/p[k]
        minus_sum += a[k]
        numerator += (1-minus_sum)*prod
        denominator += prod
    end
    prod *= (1-p[n])/p[n]
    denominator += prod
    return numerator / denominator
end

function generate_reward_function(start_probs::AbstractVector)
    if allsame(start_probs)
        expected_reward
    else
        p -> expected_reward(p, start_probs)
    end
end

mutable struct PBProblemCache <: ProblemCache
    PBSL_r_cache::DerivativeCache
    PBSL_rk_caches::Vector{DerivativeCache}
    PBSL_rpk_caches::Vector{DerivativeCache}
    AC_x_caches::Vector{DerivativeCache}
    AC_s_caches::Matrix{DerivativeCache}
end

function PBProblemCache(n, γ, start_probs)
    reward_function = generate_reward_function(start_probs)
    PBSL_r_cache = generate_G_derivative_funcs((x, p) -> reward_function(p), n)
    PBSL_rk_caches = [generate_G_derivative_funcs((x, p) -> x[i]*reward_function(p), n) for i in 1:n]
    PBSL_rpk_caches = [generate_G_derivative_funcs((x, p) -> x[i]*p[i]*reward_function(p), n) for i in 1:n]
    AC_x_caches = [generate_G_derivative_funcs_num((x, p) -> ACRHS(p, x, n, i, γ), n) for i in 1:n]
    AC_s_caches = [generate_G_derivative_funcs_num((x, p) -> p[j]*ACRHS(p, x, n, i, γ) + p[i]*ACRHS(p, x, n, j, γ), n) for i in 1:n, j in 1:n]
    PBProblemCache(PBSL_r_cache, PBSL_rk_caches, PBSL_rpk_caches, AC_x_caches, AC_s_caches)
end
