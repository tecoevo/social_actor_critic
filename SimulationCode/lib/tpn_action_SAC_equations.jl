using ForwardDiff
using LinearAlgebra
using DifferentialEquations
using Statistics: mean
include("utilities.jl")

# Functions required for performance biased social learning
function tpnA_expected_reward(action::Int, n::Int)
    if action > 2^n
        error("Action $action > 2^$n")
    elseif action <= 0
        error("Action should be positive")
    elseif action > 2^(n-1)
        return 0
    else
        1 - floor(log2(2*action-1))/n
    end
end

function tpnA_expected_reward(action::Int, start_probs::AbstractVector{<:AbstractFloat})
    n = length(start_probs)
    if action > 2^n
        error("Action $action > 2^$n")
    elseif action <= 0
        error("Action should be positive")
    elseif action > 2^(n-1)
        return 0
    else
        J = Int(floor(log2(2*action-1))) + 1
        sum(view(start_probs, J:n))
    end
end

function tpnA_expected_reward(policy::AbstractVector{T}) where T
    n_action = length(policy)
    n = Int(log2(n_action))
    return mapreduce(+, 1:n_action) do action
        policy[action]*tpnA_expected_reward(action, n)
    end
end

function tpnA_expected_reward(policy::AbstractVector{T}, start_probs::AbstractVector{<:AbstractFloat}) where T
    n = length(start_probs)
    n_action = 2^n
    if length(policy) != n_action
        error("Length of policy $(length(policy)) does not match 2^$n")
    end
    return mapreduce(+, 1:n_action) do action
        policy[action]*tpnA_expected_reward(action, start_probs)
    end
end

function tpnA_UBSL!(dy, y, n, ω, β)
    n_action = 2^n
    x = @view y[1:n_action]
    s = reshape((@view y[n_action+1:end]), (n_action, n_action))
    ds = reshape((@view dy[n_action+1:end]), (n_action, n_action))

    for i in 1:n_action
        ds[i, i] += ω *( β^2 * x[i]*(1-x[i]) - β*(2-β)*s[i, i] )
        for j in i+1:n_action
            δ = ω *( -β^2 * x[i]*x[j] - β*(2-β)*s[i, j] )
            ds[i, j] += δ
            ds[j, i] += δ
        end
    end
    return nothing
end

function tpnA_PBSL!(dy, y, n, ω, β, start_probs)
    n_action = 2^n
    x = @view y[1:n_action]
    dx = @view dy[1:n_action]
    s = reshape((@view y[n_action+1:end]), (n_action, n_action))
    ds = reshape((@view dy[n_action+1:end]), (n_action, n_action))

    reward_function = p -> tpnA_expected_reward(p, start_probs)

    # population average expected reward
    Er = reward_function(x)
    # population average of expected reward * policy
    Epr = map(1:n_action) do i
        x[i]*Er + reward_function(@view s[i, :])
    end

    for i in 1:n_action
        dx[i] += ω * β * (Epr[i]/Er - x[i] )
        ds[i, i] += ω * ( β^2 * ( (1-2x[i])*Epr[i]/Er + x[i]^2 ) - β*(2-β)*s[i, i] )
        for j in i+1:n_action
            δ = ω * β * ((x[i]^2 + x[j]^2 - (2-β)*x[i]*x[j] - (2-β)s[i, j]) +
                         ((1-β)*x[j] - x[i])*Epr[i]/Er +
                         ((1-β)*x[i] - x[j])*Epr[j]/Er   )
            ds[i, j] += δ
            ds[j, i] += δ
        end
    end
    return nothing
end

function tpnA_ACRHS(p, n, i, start_probs)
    term1 = p[i]*tpnA_expected_reward(i, n)
    term2 = norm(p)^2
    term3 = p[i]
    term4 = 0.
    r = 0.
    for j in 1:2^n
        reward_prob = tpnA_expected_reward(j, start_probs)
        r += reward_prob * p[j]
        term4 += reward_prob * p[j]^2
    end
    return p[i]*(term1 + r*term2 - r*term3 - term4)
end

function ensemble_average_mean_tpn_action(x, s, i, n, start_probs)
    n_action = 2^n
    r = tpnA_expected_reward
    rp = r(x, start_probs)
    pnorm = norm(x)^2
    rsi = r(view(s, :, i), start_probs)

    first_order_term = x[i]^2*r(i, n) + x[i]*rp*pnorm - x[i]^2*rp - x[i]*r(x .^ 2, start_probs)
    
    term1 = 2*s[i, i]*r(i, n)
    term2 = 2*pnorm*rsi + 4*rp*dot(view(s, :, i), x) + 4*x[i]*(x' * s * r.(1:n_action, n)) + 2*x[i]*rp*tr(s)
    term3 = 2*rp*s[i, i] + 4*x[i]*rsi
    term4 = 4*r(view(s, :, i) .* x, start_probs) + 2*x[i]*r((@view s[diagind(s)]), start_probs)
    return first_order_term + (term1 + term2 - term3 - term4)/2
end

function ensemble_average_covariance_tpn_action(x, s, i, j, n, start_probs)
    n_action = 2^n
    r = tpnA_expected_reward
    R = r.(1:n_action, n)
    rp = dot(x, R)
    pnorm = norm(x)^2
    rsi = r(view(s, :, i), start_probs)
    rsj = r(view(s, :, j), start_probs)

    first_order_term = x[j]*(x[i]^2*r(i, n) + x[i]*rp*pnorm - x[i]^2*rp - x[i]*r(x .^ 2, start_probs))

    term1 = 4*x[i]*r(i, n)*s[i, j] + 2*x[j]*r(i, n)*s[i, i]
    term2 = 2*rp*pnorm*s[i, j] + 2*x[i]*pnorm*rsj + 2*x[j]*pnorm*rsi + 4*x[i]*rp*dot(view(s, :, j), x) + 4*x[j]*rp*dot(view(s, :, i), x) + 4*x[i]*x[j]*(x' * s * R) + 2*x[i]*x[j]*rp*tr(s)
    term3 = 4*x[i]*rp*s[i, j] + 2*x[i]^2*rsj + 2*x[j]*rp*s[i, i] + 4*x[i]*x[j]*rsi
    term4 = 2*r(x .^ 2, start_probs)*s[i, j] + 4*x[i]*r(view(s, :, j) .* x, start_probs) + 4*x[j]*r(view(s, :, i) .* x, start_probs) + 2*x[i]*x[j]*r((@view s[diagind(s)]), start_probs)
    return first_order_term + (term1 + term2 - term3 - term4)/2
end

function tpnA_AC!(dy, y, n, ω, α, start_probs)
    n_action = 2^n
    x = @view y[1:n_action]
    dx = @view dy[1:n_action]
    s = reshape((@view y[n_action+1:end]), (n_action, n_action))
    ds = reshape((@view dy[n_action+1:end]), (n_action, n_action))

    # population average of single individual expected change
    ep = [ensemble_average_mean_tpn_action(x, s, i, n, start_probs) for i in 1:n_action]
    # population average of p[j]*dp[i]
    eqdp = [ensemble_average_covariance_tpn_action(x, s, i, j, n, start_probs) for i in 1:n_action, j in 1:n_action]

    for i in 1:n_action
        dx[i] += (1-ω) * α * ep[i] 
        for j in i:n_action
            δ = (1-ω) * α * (eqdp[i, j] + eqdp[j, i] - x[j]*ep[i] - x[i]*ep[j])
            ds[i, j] += δ
            if j != i
                ds[j, i] += δ
            end
        end
    end

    return nothing
end

function SAC_unbiased_tpn_action!(dy, y, n, ω, α, β, start_probs)
    dy .= 0.

    if ω != 0.
        tpnA_UBSL!(dy, y, n, ω, β)
    end

    if ω != 1.
        tpnA_AC!(dy, y, n, ω, α, start_probs)
    end

    n_action = 2^n
    dx = @view dy[1:n_action]
    dx .-= mean(dx)

    return dy
end

function SAC_unbiased_tpn_action!(dy, y, (n, ω, α, β, start_probs), t)
    SAC_unbiased_tpn_action!(dy, y, n, ω, α, β, start_probs)
end

function SAC_perfbiased_tpn_action!(dy, y, n, ω, α, β, start_probs)
    dy .= 0.

    if ω != 0.
        tpnA_PBSL!(dy, y, n, ω, β, start_probs)
    end

    if ω != 1.
        tpnA_AC!(dy, y, n, ω, α, start_probs)
    end

    n_action = 2^n
    dx = @view dy[1:n_action]
    dx .-= mean(dx)

    return dy
end

function SAC_perfbiased_tpn_action!(dy, y, (n, ω, α, β, start_probs), t)
    SAC_perfbiased_tpn_action!(dy, y, n, ω, α, β, start_probs)
end

function project_xs!(y, n)
    n_action = 2^n
    x = @view y[1:n_action]
    s = reshape((@view y[n_action+1:end]), (n_action, n_action))

    if sum(x) != 1.
        project_simplex!(x)
    end
    
    # make rows sum to 1
    P = I - fill(1/n_action, (n_action, n_action)) 
    s .= P * s * P

    # symmetrize
    s ./= 2
    s .+= transpose(s)
    
    project_covariance!(s)

    return nothing
end

stop_condition_convergence_affect!(integrator) = terminate!(integrator, ReturnCode.Terminated)
function generate_tpn_action_stop_condition(n, start_probs, rtol)
    n_action = 2^n
    convergence_condition = (u, t, integrator) -> tpnA_expected_reward(u[1:n_action], start_probs) - (1. - rtol)
    ContinuousCallback(convergence_condition, stop_condition_convergence_affect!)
end

function generate_tpn_action_projection_callback(n)
    n_action = 2^n
    projection_condition = (u, t, integrator) -> true
    function projection_affect!(integrator) 
        x = view(integrator.u, 1:n_action)
        s = reshape((@view integrator.u[n_action+1:end]), (n_action, n_action))
        project_simplex!(x)
        project_covariance!(s)
        return nothing
    end
    DiscreteCallback(projection_condition, projection_affect!; save_positions = (false, true))
end

function isoutofdomain_tpnA(u, p, t)
    n = p[1]
    n_action = 2^n
    s = @view u[n_action+1:end]
    return any(x -> abs(x) > 0.25, s)
end


function create_tpn_action_uniform_y0(n, s)
    n_action = 2^n
    x0 = ones(n_action) / n_action

    var = (1/n_action) * (1-1/n_action) * s
    covar = -var / (n_action - 1)

    s0 = covar*ones(n_action, n_action)
    s0[diagind(s0)] .= var
    y0 = vcat(x0, s0[:]) 
end

function create_tpn_action_monotone_y0(n::Int, s::AbstractFloat)
    n_action = 2^n
    x0 = zeros(n_action)
    for i in 0:n
        x0[2^i] = 1/(n+1)
    end
    x0 .+= 1e-4/(n+1)
    x0 ./= sum(x0)

    var = 1/(n+1) * (1-1/(n+1)) * s
    covar = -var / n
    s0 = zeros(n_action, n_action)
    for i in 0:n
        s0[2^i, 2^i] = var
        for j in i+1:n
            s0[2^i, 2^j] = covar
            s0[2^j, 2^i] = covar
        end
    end
    y0 = vcat(x0, s0[:]) 
end

function create_tpn_action_polar_y0(n, s)
    n_action = 2^n
    x0 = zeros(n_action)
    x0[1] = 0.5
    x0[end] = 0.5
    x0 .+= eps(Float64)
    x0 ./= sum(x0)

    var = (1/n_action) * (1-1/n_action) * s
    covar = -var / (n_action - 1)

    s0 = covar*ones(n_action, n_action)
    s0[diagind(s0)] .= var
    y0 = vcat(x0, s0[:]) 
end

function coarse_boundscheck(y, n)
    n_action = 2^n
    S = @view y[n_action+1:end]
    return all(u -> abs(u) <= 0.25, S) 
end

function learning_time_tpn_action(n::Integer, y0::Vector, ω::Float64, α::Float64, β::Float64, bias::Symbol = :un, start_probs::Vector=ones(n)/n; rtol = 0.1, Tmax=1_000_000, soltol = 1e-9, project_manifold = true)
    fname = eval(Symbol("SAC_$(bias)biased_tpn_action!"))
    pars = (n, ω, α, β, start_probs)
    prob = ODEProblem(fname, y0, (0, Tmax), pars)
    cb1 = generate_tpn_action_stop_condition(n, start_probs, rtol)
    cb = if project_manifold
        cb2 = generate_tpn_action_projection_callback(n)
        CallbackSet(cb1, cb2)
    else
        cb1
    end
    sol = solve(prob; callback = cb, abstol = soltol, reltol = soltol, save_everystep = false, isoutofdomain = isoutofdomain_tpnA)
    learning_time = if sol.retcode == ReturnCode.Terminated && coarse_boundscheck(sol.u[end], n)
        sol.t[end]
    else
        -1.
    end
    return learning_time
end

function learning_time_tpn_action_multiple_rtol(n::Integer, y0::Vector, ω::Float64, α::Float64, β::Float64, rtols::Vector{<:AbstractFloat}, bias::Symbol = :un, start_probs::Vector=ones(n)/n; Tmax=1_000_000, soltol = 1e-9, project_manifold = true)
    fname = eval(Symbol("SAC_$(bias)biased_tpn_action!"))
    pars = (n, ω, α, β, start_probs)
    sort!(rtols; rev = true)
    learning_times = zeros(size(rtols))
    tstart = 0.
    u0 = deepcopy(y0) 
    for (idx, rtol) in enumerate(rtols)
        prob = ODEProblem(fname, u0, (tstart, Tmax), pars)
        cb1 = generate_tpn_action_stop_condition(n, start_probs, rtol)
        cb = if project_manifold
            cb2 = generate_tpn_action_projection_callback(n)
            CallbackSet(cb1, cb2)
        else
            cb1
        end
        try
            sol = solve(prob; callback = cb, abstol = soltol, reltol = soltol, save_everystep = false, isoutofdomain = isoutofdomain_tpnA)
            if sol.retcode == ReturnCode.Terminated && coarse_boundscheck(sol.u[end], n)
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