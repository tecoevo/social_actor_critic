abstract type GlobalCache end
struct UBGlobalCache <: GlobalCache
    hessian_matrix::Matrix{Float64}
    SL_k_expectations::Vector{Float64}
    SL_pk_expectations::Vector{Float64}
    AC_x_expectations::Vector{Float64}
    AC_s_expectations::Matrix{Float64}
    reward_function::Function
end

function UBGlobalCache(n, start_probs)
    hessian_matrix = zeros(n, n)

    SL_k_expectations = zeros(n)
    SL_pk_expectations = zeros(n)
    
    AC_x_expectations = zeros(n)
    AC_s_expectations = zeros(n, n)

    reward_function = generate_reward_function(start_probs)

    return UBGlobalCache(hessian_matrix, SL_k_expectations, SL_pk_expectations, AC_x_expectations, AC_s_expectations, reward_function)
end

function reset!(cache::UBGlobalCache)
    cache.SL_k_expectations .= 0.
    cache.SL_pk_expectations .= 0.
    cache.AC_x_expectations .= 0.
    cache.AC_s_expectations .= 0.
    cache.hessian_matrix .= 0.
    return nothing
end

struct PBGlobalCache <: GlobalCache
    hessian_matrix::Matrix{Float64}
    SL_r_expectation::Float64
    SL_rk_expectations::Vector{Float64}
    SL_rpk_expectations::Vector{Float64}
    AC_x_expectations::Vector{Float64}
    AC_s_expectations::Matrix{Float64}
    reward_function::Function
end

function PBGlobalCache(n, start_probs)
    hessian_matrix = zeros(n, n)

    SL_r_expectation = 0.
    SL_rk_expectations = zeros(n)
    SL_rpk_expectations = zeros(n)
    
    AC_x_expectations = zeros(n)
    AC_s_expectations = zeros(n, n)

    reward_function = generate_reward_function(start_probs)

    return PBGlobalCache(hessian_matrix, SL_r_expectation, SL_rk_expectations, SL_rpk_expectations, AC_x_expectations, AC_s_expectations, reward_function)
end

function reset!(cache::PBGlobalCache)
    cache.SL_rk_expectations .= 0.
    cache.SL_rpk_expectations .= 0.
    cache.AC_x_expectations .= 0.
    cache.AC_s_expectations .= 0.
    cache.hessian_matrix .= 0.
    return nothing
end