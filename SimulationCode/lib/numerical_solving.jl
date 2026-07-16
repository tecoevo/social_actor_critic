convergence_affect!(integrator) = terminate!(integrator, ReturnCode.Terminated)
function generate_n_state_stop_condition(n, gc, rtol)
    convergence_condition = (u, t, integrator) -> ensemble_average_reward(u, n, gc) - (1. - rtol)
    converge_cb = ContinuousCallback(convergence_condition, convergence_affect!)
end

function generate_1_state_stop_condition(rtol)
    convergence_condition = (u, t, integrator) -> u[1] - (1 - rtol)
    ContinuousCallback(convergence_condition, convergence_affect!)
end

mutable struct StabilityCheck
    n::Int
    prev_state::Vector{Float64}
    Δ_state::Vector{Float64}
    prev_reward::Float64
    prev_time::Float64
    threshold::Float64
    reward_function::Function
    function StabilityCheck(u::Vector, n::Int, gc::Union{GlobalCache, Nothing}, threshold::Real = 10.)
        reward_function = if n == 1
            u -> u[1]
        else
            u -> ensemble_average_reward(u, n, gc)
        end
        r = reward_function(u)
        prev_state = deepcopy(u)
        Δ_state = similar(prev_state)

        new(n, prev_state, Δ_state, r, 0., threshold, reward_function)
    end
end

function boundscheck_means(u, r, st::StabilityCheck)
    n = st.n
    # reward must lie between 0 and 1
    0. <= r <= 1. || return false

    # all means must lie between 0 and 1
    for idx in 1:n
        0. <= u[idx] <= 1. || return false
    end
    return true

end

function boundscheck_variances(u, st::StabilityCheck)
    n = st.n
    x = view(u, 1:n)
    s = reshape((@view u[n+1:end]), (n, n)) 
    # the limits for the covariances depends on the means
    for i in 1:n
        # variances should lie in [0, μ(1-μ)]
        0 <= s[i, i] <= x[i]*(1-x[i]) || return false
        for j in i+1:n
            # covariances should lie in [max(μ₁ + μ₂ - 1, 0) - μ₁μ₂ , min(μ₁, μ₂) - μ₁μ₂]
            max(x[i] + u[j] - 1., 0.) <= s[i, j] + x[i]*x[j] <= min(x[i], x[j])  || return false
        end
    end
    return true

end

function (st::StabilityCheck)(dt, u, p, t)
    new_reward = st.reward_function(u)
    boundscheck_means(u, new_reward, st) || return true
    boundscheck_variances(u, st) || return true
    @. st.Δ_state = u - st.prev_state
    Δt = t - st.prev_time
    reward_rate_of_change = abs(new_reward - st.prev_reward) / Δt
    max_u_rate_of_change = mapreduce(abs, max, st.Δ_state) / Δt
    st.prev_state .= u
    st.prev_reward = new_reward
    st.prev_time = t
    return reward_rate_of_change/max_u_rate_of_change > st.threshold
end

function create_y0(n, x, s, cov_factor)
    x0 = x .* ones(n)
    s0 = s*cov_factor .* ones(n, n)
    s0[diagind(s0)] .= s
    y0 = vcat(x0, s0[:])
    return y0
end

function create_pars(n, ω, α, β, γ, full_demonstrator_choice, y0, create_matrix, bias = :un, start_probs = ones(n)/n)
    if bias == :un 
        global_cache = UBGlobalCache(n, start_probs)
        problem_cache = UBProblemCache(n, γ)
    else
        global_cache = PBGlobalCache(n, start_probs)
        problem_cache = PBProblemCache(n, γ, start_probs)
    end
    iteration_cache = IterationCache(y0[1:n], create_matrix)

    p = (n, ω, α, β, γ, full_demonstrator_choice, global_cache, iteration_cache, problem_cache)
    return p
end

