function learning_time_n_state(n::Integer, y0::Vector, ω::Float64, α::Float64, β::Float64, γ::Float64, full_demonstrator_choice::Bool, bias::Symbol = :un, start_probs::Vector=ones(n)/n; rtol = 0.1, Tmax=1_000_000, stability_threshold = 10., soltol = 1e-9)
    fname = eval(Symbol("SAC_$(bias)biased_n_state!"))
    p = create_pars(n, ω, α, β, γ, full_demonstrator_choice, y0, generate_matrix_function(start_probs), bias, start_probs) 
    global_cache = p[7] 
    prob = ODEProblem(fname, y0, (0, Tmax), p)
    cb = generate_n_state_stop_condition(n, global_cache, rtol)
    uc = StabilityCheck(y0, n, global_cache, stability_threshold)
    sol = solve(prob, RadauIIA5(; autodiff = AutoFiniteDiff()); callback = cb, abstol = soltol, reltol = soltol, unstable_check = uc)
    learning_time = if sol.retcode == ReturnCode.Terminated
        sol.t[end]
    else
        -1.
    end
    return learning_time
end

function learning_time_n_state_multiple_rtol(n::Integer, y0::Vector, ω::Float64, α::Float64, β::Float64, γ::Float64, full_demonstrator_choice::Bool, rtols::Vector, bias::Symbol = :un, start_probs::Vector=ones(n)/n; Tmax=1_000_000, stability_threshold = 10., soltol = 1e-9)
    fname = eval(Symbol("SAC_$(bias)biased_n_state!"))
    pars = create_pars(n, ω, α, β, γ, full_demonstrator_choice, y0, generate_matrix_function(start_probs), bias, start_probs) 
    global_cache = pars[7]
    uc = StabilityCheck(y0, n, global_cache, stability_threshold)
    learning_times = zeros(size(rtols))
    tstart = 0.
    u0 = deepcopy(y0) 
    for (idx, rtol) in enumerate(rtols)
        prob = ODEProblem(fname, u0, (tstart, Tmax), pars)
        cb = generate_n_state_stop_condition(n, global_cache, rtol)
        try
            sol = solve(prob, RadauIIA5(; autodiff = AutoFiniteDiff()); callback = cb, abstol = soltol, reltol = soltol, unstable_check = uc)
            if sol.retcode == ReturnCode.Terminated
                learning_times[idx] = sol.t[end]
            else
                learning_times[idx:end] .= -1.
                break
            end
            u0 .= sol.u[end]
            tstart = sol.t[end]
        catch
            learning_times[idx:end] .= -1.
            break
        end
    end

    return learning_times
end

function learning_time_1_state(y0, ω, α, β, bias; rtol=0.1, Tmax=1_000_000, stability_threshold = 10., soltol = 1e-9)
    fname = eval(Symbol("SAC_$(bias)biased_1_state!"))
    prob = ODEProblem(fname, y0, (0, Tmax), (ω, α, β))
    cb = generate_1_state_stop_condition(rtol)
    uc = StabilityCheck(y0, 1, nothing, stability_threshold)
    sol = solve(prob, RadauIIA5(); callback = cb, unstable_check = uc, abstol = soltol, reltol = soltol)
    learning_time = if sol.retcode == ReturnCode.Terminated
        sol.t[end]
    else
        -1.
    end
    return learning_time
end

function learning_time_1_state_multiple_rtol(y0, ω, α, β, full_demonstrator_choice, bias, rtols; Tmax=1_000_000, stability_threshold = 10., soltol = 1e-9)
    fname = eval(Symbol("SAC_$(bias)biased_1_state!"))
    pars = (ω, α, β, full_demonstrator_choice)
    uc = StabilityCheck(y0, 1, nothing, stability_threshold)
    learning_times = zeros(size(rtols))
    tstart = 0.
    u0 = deepcopy(y0) 
    for (idx, rtol) in enumerate(rtols)
        prob = ODEProblem(fname, u0, (tstart, Tmax), pars)
        cb = generate_1_state_stop_condition(rtol)
        try
            sol = solve(prob, RadauIIA5(); callback = cb, unstable_check = uc, abstol = soltol, reltol = soltol)
            if sol.retcode == ReturnCode.Terminated
                learning_times[idx] = sol.t[end]
            else
                learning_times[idx:end] .= -1.
                break
            end
            u0 .= sol.u[end]
            tstart = sol.t[end]
        catch 
            learning_times[idx:end] .= -1.
            break
        end
    end
    return learning_times
end

function relative_learning_times(n::Integer, ωs, y0::Vector, α::Float64, β::Float64, γ::Float64, bias::Symbol = :un, start_probs = ones(n)/n; ex = ThreadedEx(), kwargs...)
    if n == 1
        return relative_learning_times_1_state(ωs, y0, α, β, bias; kwargs...)
    end
    base_learning_time = learning_time_n_state(n, y0, ωs[1], α, β, γ, bias, start_probs; kwargs...)
    if base_learning_time < 0 
        return Float64[], Float64[]
    end
    learning_times = Folds.map(ωs, ex) do ω
        learning_time_n_state(n, y0, ω, α, β, γ, bias, start_probs; kwargs...)
    end
    converged_indices = learning_times .> 0
    learning_times = learning_times[converged_indices]
    ωs_new = ωs[converged_indices]
    relative_learning_times = learning_times ./ base_learning_time
    return ωs_new, relative_learning_times
end

function relative_learning_times_1_state(ωs, y0::Vector, α::Float64, β::Float64, bias::Symbol; kwargs...)
    base_learning_time = learning_time_1_state(y0, 0., α, β, bias; kwargs...)
    learning_times = map(ωs) do ω
        learning_time_1_state(y0, ω, α, β, bias; kwargs...)
    end
    converged_indices = learning_times .> 0
    learning_times = learning_times[converged_indices]
    ωs_new = ωs[converged_indices]
    relative_learning_times = learning_times ./ base_learning_time
    return ωs_new, relative_learning_times
end