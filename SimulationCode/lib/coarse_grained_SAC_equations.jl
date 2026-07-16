using LinearAlgebra
using ForwardDiff
using LinearSolve
using Symbolics
using LoopVectorization
using DifferentialEquations
using Folds
using Distributions

include("utilities.jl")
include("transition_matrix.jl")
include("derivative_cache.jl")
include("iteration_cache.jl")
include("hessian_matrix.jl")
include("problem_cache.jl")
include("global_cache.jl")
include("equations.jl")
include("numerical_solving.jl")
include("one_state_SAC.jl")
include("learning_time.jl")
include("steady_state.jl")

function cg_UBSL!(dy, y, n, ω, β)
    x = view(y, 1:n)
    dx = view(dy, 1:n)
    s = @view y[n+1:end]
    ds = @view dy[n+1:end]

    for i in axes(s, 1)
       ds[i] += - ω * 2 * β * (1-β) * s[i]
    end

    return nothing
end

function ensemble_average(f::Function, x, s)
    mean_value = f(x)
    hess = ForwardDiff.hessian(f, x)
    hess_sum = dot(hess, s)
    return mean_value + hess_sum/2
end

function cg_PBSL!(dy, y, n, ω, β, reward_function)
    x = view(y, 1:n)
    dx = view(dy, 1:n)
    s = reshape((@view y[n+1:end]), (n, n))
    ds = reshape((@view dy[n+1:end]), (n, n))
    
    Er = ensemble_average(reward_function, x, s)
    Epr = [ensemble_average(p -> p[i]*reward_function(p), x, s) for i in 1:n]
    Epqr = zeros(n, n)
    for i in 1:n
        for j in i:n
            Epqr[i, j] = ensemble_average(p -> p[i]*p[j]*reward_function(p), x, s)
        end
    end

    for i in 1:n
        dx[i] += ω * β * (Epr[i]/Er - x[i])
    end

    for i in 1:n
        for j in i:n
            δ = β^2 * (Epqr[i, j]/Er - x[i]*Epr[j]/Er - x[j]*Epr[i]/Er + x[i]*x[j]) - β*(2-β)*s[i,j] 
            δ *= ω
            ds[i, j] += δ
            ds[j, i] += δ
        end
    end

    return nothing
end

function one_state_AC!(dy, y, ω, α)
    x, s = y
    dy[1] = (1-ω) * α * ( x^2*(1-x)^2 + s*x*(1-6x+6x^2) )
    dy[2] = (1-ω) * ( 4*α*s*(1-x)*x*(1-2x) )
    return nothing
end

function SAC_coarse_grained_unbiased_n_state!(dy, y, n, ω, α, β, γ, global_cache::UBGlobalCache, iteration_cache::IterationCache, problem_cache::UBProblemCache)
    x = view(y, 1:n)
    s = reshape((@view y[n+1:end]), (n, n))
    ds = reshape((@view dy[n+1:end]), (n, n))
    updateIterationCache!(iteration_cache, x)
    reset!(global_cache)

    # zeroing the output variables
    dy .= 0.

    ## Unbiased social learning part
    if ω != 0.
        cg_UBSL!(dy, y, n, ω, β)
    end

    ## Asocial actor-critic part
    if ω != 1.
        AC!(dy, y, ω, α, γ, global_cache, iteration_cache, problem_cache)
    end

    return dy
end

function SAC_coarse_grained_unbiased_n_state!(dy, y, (n, ω, α, β, γ, global_cache, iteration_cache, problem_cache), t)
    SAC_coarse_grained_unbiased_n_state!(dy, y, n, ω, α, β, γ, global_cache, iteration_cache, problem_cache)
end

function SAC_coarse_grained_perfbiased_n_state!(dy, y, n, ω, α, β, γ, global_cache::PBGlobalCache, iteration_cache::IterationCache, problem_cache::PBProblemCache)
    x = view(y, 1:n)
    s = reshape((@view y[n+1:end]), (n, n))
    ds = reshape((@view dy[n+1:end]), (n, n))
    updateIterationCache!(iteration_cache, x)
    reset!(global_cache)

    # zeroing the output variables
    dy .= 0.

    ## Unbiased social learning part
    if ω != 0.
        cg_PBSL!(dy, y, n, ω, β, global_cache.reward_function)
    end

    ## Asocial actor-critic part
    if ω != 1.
        AC!(dy, y, ω, α, γ, global_cache, iteration_cache, problem_cache)
    end

    for i in 1:n
        if s[i,i] + ds[i, i] < 0.
            ds[i, i] = -s[i,i]
        end
    end

    return dy
end

function SAC_coarse_grained_perfbiased_n_state!(dy, y, (n, ω, α, β, γ, global_cache, iteration_cache, problem_cache), t)
    SAC_coarse_grained_perfbiased_n_state!(dy, y, n, ω, α, β, γ, global_cache, iteration_cache, problem_cache)
end

function SAC_coarse_grained_unbiased_1_state!(dy, y, (ω, α, β), t)
    dy .= 0.

    if ω != 0.
        cg_UBSL!(dy, y, 1, ω, β)
    end

    if ω != 1.
        one_state_AC!(dy, y, ω, α)
    end

    return dy
end

function SAC_coarse_grained_perfbiased_1_state!(dy, y, (ω, α, β), t)
    dy .= 0.

    reward_function = x -> x[1]
    if ω != 0.
        cg_PBSL!(dy, y, 1, ω, β, reward_function)
    end

    if ω != 1.
        one_state_AC!(dy, y, ω, α)
    end

    return dy
end

function create_pars_CG(n, ω, α, β, γ, y0, create_matrix, bias = :un, start_probs = ones(n)/n)
    if bias == :un 
        global_cache = UBGlobalCache(n, start_probs)
        problem_cache = UBProblemCache(n, γ)
    else
        global_cache = PBGlobalCache(n, start_probs)
        problem_cache = PBProblemCache(n, γ, start_probs)
    end
    iteration_cache = IterationCache(y0[1:n], create_matrix)

    p = (n, ω, α, β, γ, global_cache, iteration_cache, problem_cache)
    return p
end

function generate_CG_projection_callback(n)
    function projection_condition(u, t, integrator)
        s = reshape((@view u[n+1:end]), (n, n))
        for i in 1:n
            if s[i,i] < 0.
                return true
            end
        end
        return false
    end

    function projection_affect!(integrator)
        s = reshape((@view integrator.u[n+1:end]), (n, n))
        for i in 1:n
            if s[i,i] < 0.
                s[i, i] = 0.
            end
        end
        return nothing
    end

    DiscreteCallback(projection_condition, projection_affect!)
end

function learning_time_n_state_CG(n::Integer, y0::Vector, ω::Float64, α::Float64, β::Float64, γ::Float64, bias::Symbol = :un, start_probs::Vector=ones(n)/n; rtol = 0.1, Tmax=1_000_000, stability_threshold = 10., soltol = 1e-9)
    fname = eval(Symbol("SAC_coarse_grained_$(bias)biased_n_state!"))
    p = create_pars_CG(n, ω, α, β, γ, y0, generate_matrix_function(start_probs), bias, start_probs) 
    global_cache = p[6] 
    prob = ODEProblem(fname, y0, (0, Tmax), p)
    cb1 = generate_n_state_stop_condition(n, global_cache, rtol)
    cb2 = generate_CG_projection_callback(n)
    cb = CallbackSet(cb1, cb2)
    uc = StabilityCheck(y0, n, global_cache, stability_threshold)
    sol = solve(prob, RadauIIA5(; autodiff = AutoFiniteDiff()); callback = cb, abstol = soltol, reltol = soltol, unstable_check = uc)
    learning_time = if sol.retcode == ReturnCode.Terminated
        sol.t[end]
    else
        -1.
    end
    return learning_time
end

function learning_time_n_state_CG_multiple_rtol(n::Integer, y0::Vector, ω::Float64, α::Float64, β::Float64, γ::Float64, rtols::Vector, bias::Symbol = :un, start_probs::Vector=ones(n)/n; Tmax=1_000_000, stability_threshold = 10., soltol = 1e-9)
    fname = eval(Symbol("SAC_coarse_grained_$(bias)biased_n_state!"))
    pars = create_pars_CG(n, ω, α, β, γ, y0, generate_matrix_function(start_probs), bias, start_probs) 
    global_cache = pars[6]
    uc = StabilityCheck(y0, n, global_cache, stability_threshold)
    learning_times = zeros(size(rtols))
    tstart = 0.
    u0 = deepcopy(y0) 
    cb2 = generate_CG_projection_callback(n)
    for (idx, rtol) in enumerate(rtols)
        prob = ODEProblem(fname, u0, (tstart, Tmax), pars)
        cb1 = generate_n_state_stop_condition(n, global_cache, rtol)
        cb = CallbackSet(cb1, cb2)
        try
            sol = solve(prob; callback = cb, abstol = soltol, reltol = soltol, unstable_check = uc)
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

function learning_time_1_state_CG(y0, ω, α, β, bias; rtol=0.1, Tmax=1_000_000, stability_threshold = 10., soltol = 1e-9)
    fname = eval(Symbol("SAC_coarse_grained_$(bias)biased_1_state!"))
    prob = ODEProblem(fname, y0, (0, Tmax), (ω, α, β))
    cb = generate_1_state_stop_condition(rtol)
    uc = StabilityCheck(y0, 1, nothing, stability_threshold)
    sol = solve(prob, RadauIIA5(; autodiff = AutoFiniteDiff()); callback = cb, unstable_check = uc, abstol = soltol, reltol = soltol)
    learning_time = if sol.retcode == ReturnCode.Terminated
        sol.t[end]
    else
        -1.
    end
    return learning_time
end

function learning_time_1_state_CG_multiple_rtol(y0, ω, α, β, bias, rtols; Tmax=1_000_000, stability_threshold = 10., soltol = 1e-9)
    fname = eval(Symbol("SAC_coarse_grained_$(bias)biased_1_state!"))
    pars = (ω, α, β)
    uc = StabilityCheck(y0, 1, nothing, stability_threshold)
    learning_times = zeros(size(rtols))
    tstart = 0.
    u0 = deepcopy(y0) 
    for (idx, rtol) in enumerate(rtols)
        prob = ODEProblem(fname, u0, (tstart, Tmax), pars)
        cb = generate_1_state_stop_condition(rtol)
        try
            sol = solve(prob, RadauIIA5(; autodiff = AutoFiniteDiff()); callback = cb, unstable_check = uc, abstol = soltol, reltol = soltol)
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