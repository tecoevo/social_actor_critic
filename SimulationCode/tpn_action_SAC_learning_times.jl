## ---------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
N = 1:6
init = [:monotone, :uniform, :polar]
s0 = [0.01, 0.05, 0.1]

ω = 0:0.01:0.99
α = 0.01
β = 0.01
bias = [:un, :perf]
rtols = [[0.4, 0.3, 0.2, 0.1]] # must be in descending order and a vector of vector or :relative_difference or :relative_ratio
project_manifold = true

Tmax = 1_000_000
soltol = 1e-6

start_probs_mean = 0.5 
start_probs_shape = 1.

start_dist_func = [:start_beta_binomial]

filename = "SAC_tpnA_learning_times"

## --------------------------------------------------------------------
# Load packages
# ---------------------------------------------------------------------
using Distributed
using SlurmClusterManager
if nprocs() == 1
    if haskey(ENV, "SLURM_JOB_ID") || haskey(ENV, "SLURM_JOBID")
        addprocs(SlurmManager())
    else
        addprocs(Sys.CPU_THREADS)
    end
end

println("Launched $(nworkers()) processes")

using DataFrames
using Base.Iterators: product
using Arrow
import Random

@everywhere include("lib/tpn_action_SAC_equations.jl")

@everywhere using Distributions
@everywhere using ProgressMeter
@everywhere import Logging
@everywhere Logging.disable_logging(Logging.Warn)
@everywhere import Dates

@everywhere function learning_time_wrapped(n, init, s0, ω, α, β, bias, rtols, Tmax, soltol, project_manifold, start_probs_mean, start_probs_shape, start_dist_func)
    y0 = if init == :monotone
        create_tpn_action_monotone_y0(n, s0)
    elseif init == :uniform
        create_tpn_action_uniform_y0(n, s0)
    elseif init == :polar
        create_tpn_action_polar_y0(n, s0)
    else
        error("Invalid initial state type $init")
    end
    start_probs, start_probs_name = eval(start_dist_func)(n, start_probs_mean, start_probs_shape)
    new_rtols = if rtols == :relative_ratio
        x0 = y0[1:(2^n)]
        base_r = tpnA_expected_reward(x0, start_probs)
        rtols_1 = map([0.2, 0.4, 0.6, 0.8]) do perc
            (1. - base_r)*(1 - perc)
        end
        rtols_1
    elseif rtols == :relative_difference
        x0 = y0[1:(2^n)]
        base_r = tpnA_expected_reward(x0, start_probs)
        rtols_1 = map([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) do perc
            1. - (base_r + perc)
        end
        filter!(x -> x > 0, rtols_1)
        rtols_1
    else
        rtols
    end
    return learning_time_tpn_action_multiple_rtol(n, y0, ω, α, β, new_rtols, bias, start_probs; Tmax, soltol, project_manifold)
end

println("Loaded all packages.")

## --------------------------------------------------------------------
# Create parameter combinations and run the calculations
# ---------------------------------------------------------------------

all_pars = product(N, init, s0, ω, α, β, bias, rtols, Tmax, soltol, project_manifold, start_probs_mean, start_probs_shape, start_dist_func) 
all_pars = collect(all_pars)[:]
Random.shuffle!(all_pars)

pbar = Progress(length(all_pars); dt = 60.)
channel = RemoteChannel(() -> Channel{Bool}(), 1)

println("Starting parallel calculations.")

results = @sync begin
    @async while take!(channel)
        next!(pbar; showvalues = [(:Time, Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))])
    end

    @async begin
        results = pmap(all_pars) do pars
            res = learning_time_wrapped(pars...)
            put!(channel, true)
            res
        end
        put!(channel, false)
        results
    end
end

results = fetch(results)

println("Finished parallel calculations. Saving to DataFrame.")

## ----------------------------------------------------------------------------------
# Store the results in a DataFrame and write it to disk
# -----------------------------------------------------------------------------------
columns = (;
    n = Int64[],
    init = Symbol[],
    s0 = Float64[],
    ω = Float64[],
    α = Float64[],
    β = Float64[],
    bias = Symbol[],
    rtol = Union{Symbol, Vector{Float64}}[],
    Tmax = Float64[],
    soltol = Float64[],
    project_manifold = Bool[],
    start_probs = Vector{Float64}[],
    start_dist = String[],
    learning_time = Vector{Float64}[]
)

df = DataFrame(columns)

for ((n, pars..., start_probs_mean, start_probs_shape, start_dist_func), learning_time) in zip(all_pars, results)
    start_probs, start_probs_name = eval(start_dist_func)(n, start_probs_mean, start_probs_shape)
    push!(df, (n, pars..., start_probs, start_probs_name, learning_time))
end

Arrow.write(filename*".arrow", df)

println("Successfully saved results. Ending.")

rmprocs(workers())