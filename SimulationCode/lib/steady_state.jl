using Random
using Memoization
using IntervalSets
using OffsetArrays

@memoize function get_vec_to_mat_indices(n)
    m = UpperTriangular(CartesianIndices((n, n)))[:]
    mat_to_vec_indices = findall(!iszero, m)
    vec_to_mat_indices = Tuple.(m[mat_to_vec_indices])
    return vec_to_mat_indices
end

function variance_index(i)
    binomial(i+1, 2)
end

"""
    draw_means_and_variances(n, scale) -> (μ, σ²)
 
Draw n means uniformly from [0,1], then draw variances uniformly
from (0, scale * μᵢ(1-μᵢ)], the tightest valid range given each mean.
"""
function draw_means_and_variances(n::Int, scale::Real=1.)
    μ = rand(n)                          # means ~ Uniform(0,1)
    var_max = @. scale * μ * (1 - μ)              # upper bound: scale * μᵢ(1-μᵢ)
    σ² = rand(n) .* var_max              # variances ~ Uniform(0, var_max]
    return μ, σ²
end

"""
Compute elementwise Fréchet-Hoeffding bounds on the correlation matrix,
given means μ and standard deviations σ.
"""
function frechet_bounds(μ, σ)
    n = length(μ)
    R_lo = fill(-1.0, n, n)
    R_hi = fill( 1.0, n, n)
    for i in 1:n, j in 1:n
        i == j && continue
        cov_lo =  max(μ[i]+μ[j] - 1., 0.) - μ[i]*μ[j]   # cov lower bound
        cov_hi =  min(μ[i], μ[j])         - μ[i]*μ[j]   # cov upper bound
        R_lo[i,j] = cov_lo / (σ[i]*σ[j])                # corr lower bound
        R_hi[i,j] = cov_hi / (σ[i]*σ[j])                # corr upper bound
    end
    return R_lo, R_hi
end

struct TruncatedMirroredBeta <: ContinuousUnivariateDistribution
    dist::Truncated{<:Beta}
    function TruncatedMirroredBeta(α, lower, upper)
        new(truncated(Beta(α), (lower+1.)/2., (upper+1.)/2.))
    end
end

function Distributions.rand(rng::AbstractRNG, d::TruncatedMirroredBeta)
    2. *rand(rng, d.dist) - 1.
end

function vine_covariance(n::Int, μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real}, β::Real=1.0)
    # Step 2: Compute Fréchet–Hoeffding bounds on R
    R_lo, R_hi = frechet_bounds(μ, σ)

    # Step 3: Vine sampling with constrained partial correlations
    P = zeros(n, n, n-1)
    P = OffsetArray(P, OffsetArrays.Origin(1, 1, 0))
    for i in 0:n-2
        for j in 1:n
            P[j, j, i] = 1.
        end
    end

    for i in 1:n-1
        for j in i+1:n
            α = (n - i + 1)/2
            # Determine the implied range for the partial correlation p_ij
            # from the Fréchet bounds on R[i,j]
            lo = max(-1.0, R_lo[i,j])
            hi = min( 1.0, R_hi[i,j])
            # Sample partial correlation uniformly within valid range
            P[i, j, i-1] = rand(TruncatedMirroredBeta(α, lo, hi))

            if i > 1
                for k in 1:i-1
                    P[i, j, i-k-1] = P[i, j, i-k] * sqrt( (1 - P[i-k, i, i-k-1]^2) * (1 - P[i-k, j, i-k-1]^2) ) + P[i-k, i, i-k-1]*P[i-k, j, i-k-1]
                end
            end
        end
    end

    for i in 1:n-1
        for j in i+1:n
            P[j, i, 0] = P[i, j, 0]
        end
    end

    R = β * P[:, :, 0] + (1-β)*I

    # Step 4: Assemble Σ = D·R·D
    D = Diagonal(σ)
    Σ = Matrix(D * R * D)
    return Σ
end

function random_initial_condition(n::Int, gc=UBGlobalCache(n, ones(n)/n), scale:: NTuple{2, <:AbstractFloat}=(1.0,1.0))
    α, β = scale
    μ, σ² = draw_means_and_variances(n, α)
    σ = sqrt.(σ²)

    passed = false

    u = zeros(n+n^2)
    u[1:n] .= μ

    count = 0
    while !passed && count < 1_000
        Σ = vine_covariance(n, μ, σ, β)
        u[n+1:end] .= view(Σ, :)
        r = ensemble_average_reward(u, n, gc)
        if 0 <= r <= 1
            passed = true
        end
        count += 1
    end
    if passed 
        return u
    else
        return random_initial_condition(n, gc, scale)
    end
end

function steady_state_1_state(y0, ω, α, β, bias = :un; stability_threshold = 10., soltol = 1e-9)
    fname = eval(Symbol("SAC_$(bias)biased_1_state!"))
    uc = StabilityCheck(y0, 1, nothing, stability_threshold)
    prob = SteadyStateProblem(fname, y0, (ω, α, β))
    sol = solve(prob, DynamicSS(RadauIIA5()); abstol = soltol, reltol = soltol, unstable_check = uc)
    if sol.retcode == ReturnCode.Success
        sol.u
    else
        [-1., -1.]
    end
end

function steady_state_n_state(n, y0, ω, α, β, γ, bias = :un, start_probs=ones(n)/n;stability_threshold = 10., soltol = 1e-9)
    fname = eval(Symbol("SAC_$(bias)biased_n_state!"))
    p = create_pars(n, ω, α, β, γ, y0, generate_matrix_function(start_probs), bias, start_probs) 
    global_cache = p[6] 
    uc = StabilityCheck(y0, n, global_cache, stability_threshold)
    prob = SteadyStateProblem(fname, y0, p)
    sol = solve(prob, DynamicSS(RadauIIA5(; autodiff = AutoFiniteDiff())); abstol = soltol, reltol = soltol, unstable_check = uc)
    if sol.retcode == ReturnCode.Success
        sol.u
    else
        -1. * ones(length(y0))
    end
end

function random_initial_condition_n1(variance_scale)
    x_rand = rand()
    s_rand = variance_scale * x_rand * (1-x_rand) * rand()
    [x_rand; s_rand]
end

function ensemble_steady_state_1_state(N_ensemble, ω, α, β, bias = :un, ex = ThreadedEx(); stability_threshold = 10., init_scale = 1.0, soltol = 1e-9, uniqtol = 1e-6, seed=1234, channel = nothing)
    Random.seed!(seed)
    solutions = Folds.map(1:N_ensemble, ex) do _
        y0 = random_initial_condition_n1(init_scale)
        res = try 
            steady_state_1_state(y0, ω, α, β, bias; stability_threshold, soltol)
        catch err
            [-1., -1]
        end
        next!(channel)
        res
    end
    filter!(u -> !all(u .== -1), solutions)
    solutions, counts = unduplicate(solutions, uniqtol)
    rewards = map(sol -> sol[1], solutions) 
    return solutions, rewards, counts
end

function ensemble_steady_state(N_ensemble, n, ω, α, β, γ, bias = :un, start_probs = ones(n)/n, ex = ThreadedEx(); stability_threshold = 10., soltol = 1e-9, uniqtol = 1e-6, seed=1234, init_scale = (1.0, 1.0), channel = nothing)
    if n == 1
        return ensemble_steady_state_1_state(N_ensemble, ω, α, β, bias, ex; stability_threshold, init_scale = init_scale[1], soltol, uniqtol, seed, channel)
    end
    Random.seed!(seed)
    global_cache = UBGlobalCache(n, start_probs)
    solutions = Folds.map(1:N_ensemble, ex) do _
        res = try 
            y0 = random_initial_condition(n, global_cache, init_scale)
            steady_state_n_state(n, y0, ω, α, β, γ, bias, start_probs; stability_threshold, soltol)
        catch err
            -1 * ones(n*(n+3)÷2)
        end
        next!(channel)
        res
    end
    filter!(u -> !all(u .== -1), solutions)
    solutions, counts = unduplicate(solutions, uniqtol)
    
    rewards = map(solutions) do sol
        ensemble_average_reward(sol, n, global_cache)
    end
    return solutions, rewards, counts
end