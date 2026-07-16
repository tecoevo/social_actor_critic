# Unbiased social learning
function UBSL!(dy, y, ω, β, gc::UBGlobalCache, ic::IterationCache, pc::UBProblemCache)
    k = ic.x
    n = length(k)
    x = view(y, 1:n)
    dx = view(dy, 1:n)
    s = reshape((@view y[n+1:end]), (n, n))
    ds = reshape((@view dy[n+1:end]), (n, n))

    SL_k_expectations = gc.SL_k_expectations
    SL_pk_expectations = gc.SL_pk_expectations
    
    # calculating the required ensemble expectations matrices (only the second order part)
    for i in 1:n
        SL_k_expectations[i] = k[i] + dot(s, hessian_matrix!(gc.hessian_matrix, x, ic, pc.UBSL_k_caches[i]))/2
        SL_pk_expectations[i] = x[i]*k[i] + dot(s, hessian_matrix!(gc.hessian_matrix, x, ic, pc.UBSL_pk_caches[i]))/2
    end
    
    # calculating the change in the policy means
    for i in 1:n
        dx[i] += ω * β * (SL_pk_expectations[i] - x[i]*SL_k_expectations[i])
    end

    # calculating the change in the policy covariances
    for i in 1:n
        ds[i, i] += ω * (β^2*(SL_pk_expectations[i])*(1-2*x[i]) + (SL_k_expectations[i])*(β^2 * x[i]^2 - β*(2-β)*s[i, i]) )
        for j in i+1:n
            δ = ω * β * (SL_k_expectations[i] + SL_k_expectations[j]) * s[i,j]
            ds[i, j] -= δ
            ds[j, i] -= δ
        end
    end
    return nothing
end

# Performance biased social learning
function PBSL!(dy, y, ω, β, gc::PBGlobalCache, ic::IterationCache, pc::PBProblemCache)
    k = ic.x
    n = length(k)
    x = view(y, 1:n)
    dx = view(dy, 1:n)
    s = reshape((@view y[n+1:end]), (n, n))
    ds = reshape((@view dy[n+1:end]), (n, n))
    
    SL_r_expectation = gc.SL_r_expectation
    SL_rk_expectations = gc.SL_rk_expectations
    SL_rpk_expectations = gc.SL_rpk_expectations
    
    # calculating the required ensemble expectations matrices (only the second order part)
    r = gc.reward_function(x)
    SL_r_expectation = r + dot(s, hessian_matrix!(gc.hessian_matrix, x, ic, pc.PBSL_r_cache))/2
    for i in 1:n
        SL_rk_expectations[i] = r*k[i] + dot(s, hessian_matrix!(gc.hessian_matrix, x, ic, pc.PBSL_rk_caches[i]))/2
        SL_rpk_expectations[i] = r*x[i]*k[i] + dot(s, hessian_matrix!(gc.hessian_matrix, x, ic, pc.PBSL_rpk_caches[i]))/2
    end
    
    # calculating the change in the policy means
    for i in 1:n
        dx[i] += ω * β * (SL_rpk_expectations[i] - x[i]*SL_rk_expectations[i]) / SL_r_expectation
    end

    # calculating the change in the policy covariances
    for i in 1:n
        ds[i, i] += ω * ( β^2*(SL_rpk_expectations[i])*(1-2*x[i]) + (SL_rk_expectations[i])*(β^2 * x[i]^2 - β*(2-β)*s[i, i]) )/SL_r_expectation
        for j in i+1:n
            δ = ω * β * (SL_rk_expectations[i] + SL_rk_expectations[j]) / SL_r_expectation * s[i,j]
            ds[i, j] -= δ  
            ds[j, i] -= δ  
        end
    end
    return nothing
end

# Asocial actor critic
function AC!(dy, y, ω, α, γ, gc::GlobalCache, ic::IterationCache, pc::ProblemCache)
    k = ic.x
    n = length(k)
    x = view(y, 1:n)
    dx = view(dy, 1:n)
    s = reshape((@view y[n+1:end]), (n, n))
    ds = reshape((@view dy[n+1:end]), (n, n))
    
    AC_x_expectations = gc.AC_x_expectations
    AC_s_expectations = gc.AC_s_expectations

    for i in 1:n
        AC_x_expectations[i] = ACRHS(x, k, n, i, γ) + dot(s, hessian_matrix!(gc.hessian_matrix, x, ic, pc.AC_x_caches[i]))/2
        AC_s_expectations[i, i] = 2*x[i]*ACRHS(x, k, n, i, γ) + dot(s, hessian_matrix!(gc.hessian_matrix, x, ic, pc.AC_s_caches[i, i]))/2
        for j in i+1:n
            AC_s_expectations[i, j] = x[j]*ACRHS(x, k, n, i, γ) + x[i]*ACRHS(x, k, n, j, γ) + dot(s, hessian_matrix!(gc.hessian_matrix, x, ic, pc.AC_s_caches[i,j]))/2
        end
    end

    for i in 1:n
        dx[i] += (1-ω) * α * γ * AC_x_expectations[i]
        ds[i, i] += (1-ω) * α * γ * (AC_s_expectations[i,i] - 2*x[i]*AC_x_expectations[i])
        for j in i+1:n
            δ = (1-ω) * α * γ * (AC_s_expectations[i,j] - x[i]*AC_x_expectations[j] - x[j]*AC_x_expectations[i])
            ds[i, j] += δ
            ds[j, i] += δ
        end
    end
    return nothing
end

function SAC_unbiased_n_state!(dy, y, n, ω, α, β, γ, full_demonstrator_choice, global_cache::UBGlobalCache, iteration_cache::IterationCache, problem_cache::UBProblemCache)
    x = view(y, 1:n)

    updateIterationCache!(iteration_cache, x)
    reset!(global_cache)

    # zeroing the output variables
    dy .= 0.

    ## Unbiased social learning part
    if ω != 0.
        ω_new = full_demonstrator_choice ? ω*(1-ω) : ω
        UBSL!(dy, y, ω_new, β, global_cache, iteration_cache, problem_cache)
    end

    ## Asocial actor-critic part
    if ω != 1.
        AC!(dy, y, ω, α, γ, global_cache, iteration_cache, problem_cache)
    end

    return dy
end

function SAC_unbiased_n_state!(dy, y, (n, ω, α, β, γ, full_demonstrator_choice, global_cache, iteration_cache, problem_cache), t)
    SAC_unbiased_n_state!(dy, y, n, ω, α, β, γ, full_demonstrator_choice, global_cache, iteration_cache, problem_cache)
end

function SAC_perfbiased_n_state!(dy, y, n, ω, α, β, γ, full_demonstrator_choice, global_cache::PBGlobalCache, iteration_cache::IterationCache, problem_cache::PBProblemCache)
    x = view(y, 1:n)

    updateIterationCache!(iteration_cache, x)
    reset!(global_cache)

    # zeroing the output variables
    dy .= 0.

    ## Unbiased social learning part
    if ω != 0.
        ω_new = full_demonstrator_choice ? ω*(1-ω) : ω
        PBSL!(dy, y, ω_new, β, global_cache, iteration_cache, problem_cache)
    end

    ## Asocial actor-critic part
    if ω != 1.
        AC!(dy, y, ω, α, γ, global_cache, iteration_cache, problem_cache)
    end

    return dy
end

function SAC_perfbiased_n_state!(dy, y, (n, ω, α, β, γ, full_demonstrator_choice, global_cache, iteration_cache, problem_cache), t)
    SAC_perfbiased_n_state!(dy, y, n, ω, α, β, γ, full_demonstrator_choice, global_cache, iteration_cache, problem_cache)
end

function ensemble_average_reward(y, n, gc::GlobalCache)
    x = view(y, 1:n)
    s = reshape((@view y[n+1:end]), (n, n))

    ForwardDiff.hessian!(gc.hessian_matrix, p -> gc.reward_function(p), x)
    ens_reward = gc.reward_function(x) + dot(s, gc.hessian_matrix)/2
    return ens_reward
end