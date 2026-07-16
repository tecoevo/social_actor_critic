## ---------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
N = 2:10

ω = 0.5:0.01:0.99
α = 0.01
β = 0.01
γ = 0.9
bias = [:un]

stability_threshold = 10.
soltol = 1e-9
uniqtol = 1e-6

start_probs_mean = 0.5
start_probs_shape = 1.

start_dist_func = [:start_beta_binomial]

N_ensemble = 1000
random_seed = 1234
variance_scale = [(0.1, 1.0)]

filename = "../Data/SAC_steady_state"

## --------------------------------------------------------------------
# Load packages
# ---------------------------------------------------------------------
using DataFrames
using Base.Iterators: product
using Arrow
import Random

using Distributed
using SlurmClusterManager
if nprocs() == 1
    if haskey(ENV, "SLURM_JOB_ID") || haskey(ENV, "SLURM_JOBID")
        addprocs(SlurmManager())
    else
        addprocs(Sys.CPU_THREADS)
    end
end

@everywhere include("lib/n_state_SAC_equations.jl")

@everywhere using Distributions
@everywhere using ProgressMeter
@everywhere import Logging
@everywhere Logging.disable_logging(Logging.Warn)
@everywhere import Dates

@everywhere function steady_state_wrapped(N_ensemble, n, ω, α, β, γ, bias, start_probs_mean, start_probs_shape, start_dist_func, stability_threshold, soltol, uniqtol, seed, variance_scale, channel)
    start_probs, _ = eval(start_dist_func)(n, start_probs_mean, start_probs_shape)
    return ensemble_steady_state(N_ensemble, n, ω, α, β, γ, bias, start_probs, SequentialEx(); stability_threshold, soltol, uniqtol, seed, init_scale = variance_scale, channel)
end

@everywhere function start_beta_binomial(N, mean, shape)
    dist = BetaBinomial(N-1, 2*mean*shape, 2*(1-mean)*shape)
    return probs(dist), String(Symbol(dist))
end

## --------------------------------------------------------------------
# Create parameter combinations and run the calculations
# ---------------------------------------------------------------------

all_pars = product(N, ω, α, β, γ, bias, start_probs_mean, start_probs_shape, start_dist_func, stability_threshold, soltol, uniqtol, random_seed, variance_scale) 
all_pars = collect(all_pars)[:]
Random.shuffle!(all_pars)

pbar = Progress(length(all_pars)*N_ensemble; dt = 60.0)
channel = RemoteChannel(() -> Channel{Bool}(), 1)

results = @sync begin
    @async while take!(channel)
        next!(pbar; showvalues = [(:Time, Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))])
    end

    @async begin
        results = pmap(all_pars) do pars
            steady_state_wrapped(N_ensemble, pars..., channel)
        end
        put!(channel, false)
        results
    end
end

results = fetch(results)

## ----------------------------------------------------------------------------------
# Store the results in a DataFrame and write it to disk
# -----------------------------------------------------------------------------------
columns = (;
    n = Int64[],
    ω = Float64[],
    α = Float64[],
    β = Float64[],
    γ = Float64[],
    bias = Symbol[],
    start_probs = Vector{Float64}[],
    start_dist = String[],
    stability_threshold = Float64[],
    soltol = Float64[],
    uniqtol = Float64[],
    random_seed = Int64[],
    variance_scale = Tuple{Float64, Float64}[],
    N_ensemble = Int64[],
    steady_state = Vector{Vector{Float64}}[],
    rewards = Vector{Float64}[],
    counts = Vector{Int64}[]
)

df = DataFrame(columns)

for (pars, (steady_state, reward, count)) in zip(all_pars, results)
    start_probs_mean, start_probs_shape, start_dist_func = pars[7:9]
    n = pars[1]

    start_probs, start_probs_name = eval(start_dist_func)(n, start_probs_mean, start_probs_shape)
    push!(df, (pars[1:6]..., start_probs, start_probs_name, pars[10:end]..., N_ensemble, steady_state, reward, count))
end

Arrow.write(filename*".arrow", df)

rmprocs(workers())