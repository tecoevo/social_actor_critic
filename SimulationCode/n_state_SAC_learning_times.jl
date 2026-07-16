## ---------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
N = 1:10
x0 = 0.5
s0 = 0:0.01:0.05
cov_factor = -0.2:0.1:0.2

ω = 0:0.01:0.99 
α = 0.01
β = 0.01
γ = 0.9
full_demonstrator_choice = false
bias = [:un, :perf]
rtols = [[0.4, 0.3, 0.2, 0.1]] # must be in descending order and a vector of vector

Tmax = 100_000_000
stability_threshold = 10.
soltol = 1e-6

start_probs_mean = 0.1:0.1:0.9
start_probs_shape = [1., 2., 4., 8., 16.]

start_dist_func = [:start_beta_binomial]

filename = "../Data/SAC_learning_times"

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

@everywhere function learning_time_wrapped(n, x0, s0, cov, ω, α, β, γ, full_demonstrator_choice, bias, rtols, Tmax, stability_threshold, soltol, start_probs_mean, start_probs_shape, start_dist_func)
    if n == 1
        y0 = [x0, s0]
        return learning_time_1_state_multiple_rtol(y0, ω, α, β, full_demonstrator_choice, bias, rtols; Tmax, stability_threshold, soltol)
    end
    y0 = create_y0(n, x0, s0, cov)
    start_probs, start_probs_name = eval(start_dist_func)(n, start_probs_mean, start_probs_shape)
    return learning_time_n_state_multiple_rtol(n, y0, ω, α, β, γ, full_demonstrator_choice, rtols, bias, start_probs; Tmax, stability_threshold, soltol)
end

@everywhere function start_beta_binomial(N, mean, shape)
    dist = BetaBinomial(N-1, 2*mean*shape, 2*(1-mean)*shape)
    return probs(dist), String(Symbol(dist))
end

## --------------------------------------------------------------------
# Create parameter combinations and run the calculations
# ---------------------------------------------------------------------

all_pars = product(N, x0, s0, cov_factor, ω, α, β, γ, full_demonstrator_choice, bias, rtols, Tmax, stability_threshold, soltol, start_probs_mean, start_probs_shape, start_dist_func) 
all_pars = collect(all_pars)[:]
# Random.shuffle!(all_pars)

pbar = Progress(length(all_pars); dt = 60.0)
channel = RemoteChannel(() -> Channel{Bool}(), 1)

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

## ----------------------------------------------------------------------------------
# Store the results in a DataFrame and write it to disk
# -----------------------------------------------------------------------------------
columns = (;
    n = Int64[],
    x0 = Float64[],
    s0 = Float64[],
    cov = Float64[],
    ω = Float64[],
    α = Float64[],
    β = Float64[],
    γ = Float64[],
    full_demonstrator_choice = Bool[],
    bias = Symbol[],
    rtol = Vector{Float64}[],
    Tmax = Float64[],
    stability_threshold = Float64[],
    soltol = Float64[],
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

rmprocs(workers())