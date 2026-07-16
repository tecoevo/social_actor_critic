using DataFrames
using Arrow
using DataFramesMeta
using Distributions
using CairoMakie
using Chain
using Base.Iterators: product
using LinearAlgebra: dot
using StatsBase

import CairoMakie.ColorTypes

theme = Theme(
    font = "Poppins Regular" ,
    Axis = (;
        xticklabelsize = 18, 
        xticklabelfont = "Poppins Regular",
        yticklabelsize = 18, 
        yticklabelfont = "Poppins Regular",
        xlabelsize = 21, 
        xlabelfont = "Poppins Regular",
        ylabelsize = 21, 
        ylabelfont = "Poppins Regular",
        titlefont = "Poppins Medium", 
        titlesize = 20
        ), 
    Label = (; labelfont = "Poppins Regular"),
    Colorbar = (;labelfont = "Poppins Regular", labelsize = 21, ticklabelsize = 18),
    axislegend = (;labelfont = "Poppins Regular", labelsize = 18)
)
set_theme!(theme)

Distributions.BetaBinomial(; n, α, β) = Distributions.BetaBinomial(n, α, β)
function scaled_beta_dist(mean, shape)
    Distributions.Beta(2*mean*shape, 2*(1-mean)*shape)
end

function parse_beta_binomial_dist(df)
    if "start_dist" ∈ names(df) && "start_probs" ∈ names(df)
        @chain df begin
            @rtransform(:start_dist = eval(Meta.parse(replace(:start_dist, "{Float64}" => ""))))
            @rtransform(:p = :start_dist.α/(:start_dist.α + :start_dist.β))
            @rtransform(:θ = :start_dist.α / (2 * :p))
            @rtransform(:θ = 0.1 < :θ < 1 ? 0.1 : :θ )
            @select(Not([:start_probs, :start_dist]))
        end
    else
        df
    end
end

function preprocess_df(df::DataFrame)
    @chain df begin
        # convert initial distributions to mean and shape
        parse_beta_binomial_dist()  
        # Remove those parameter combinations where ω=0 does not complete learning for all rtols
        sort(:ω)
        @groupby(:n, :x0, :s0, :cov, :α, :β, :γ, :full_demonstrator_choice, :bias, :Tmax, :stability_threshold, :soltol, :p, :θ )
        @subset(all(first(:learning_time) .> 0.))
        # flatten vector of different rtols to their own rows
        flatten([:rtol, :learning_time])
        # removing duplicates
        sort([:Tmax, :soltol, :stability_threshold]; rev = [true, false, true]) 
        @groupby(Not(:Tmax, :stability_threshold, :soltol, :learning_time))
        combine(first)
        # removing entries which does not converge for pure asocial learning
        sort(:ω)
        @groupby(:n, :x0, :s0, :cov, :α, :β, :γ, :bias, :rtol, :p, :θ)
        @subset(:learning_time[1] > 0)
        @rsubset(:learning_time > 0)
        # calculating relative learning speedup
        sort(:ω)
        @groupby(:n, :x0, :s0, :cov, :α, :β, :γ, :bias, :rtol, :p, :θ)
        @transform(:relative_learning_time = :learning_time ./ :learning_time[1])
        @rtransform(:learning_speedup = 1/:relative_learning_time)
    end
end

function process_df_rtols(df::DataFrame)
    @chain df begin
        @rtransform(:reward_target = 0.5 - :rtol)
        @rtransform(:expertise_target = :reward_target * 200)
        sort([:reward_target, :ω])
        @groupby(:n, :x0, :s0, :cov, :ω, :α, :β, :γ, :bias, :p, :θ)
        @transform(:learning_time_incremental = [:learning_time[1]; :learning_time[2:end] .- :learning_time[1:end-1]]; ungroup = true)
        @groupby(:n, :x0, :s0, :cov, :α, :β, :γ, :bias, :reward_target, :p, :θ)
        @transform(:relative_learning_time_incremental = :learning_time_incremental ./ :learning_time_incremental[1])
        @rtransform(:learning_speedup_incremental = 1/:relative_learning_time_incremental)
        @groupby(:n, :x0, :s0, :cov, :α, :β, :γ, :bias, :reward_target, :p, :θ)
        @transform(:speedup_change = [:learning_speedup_incremental[1]; :learning_speedup_incremental[2:end] .- :learning_speedup_incremental[1:end-1]]; ungroup = false)
        @rsubset(:ω < 0.9 || :speedup_change < 0)
        @select(Not([:speedup_change]))
    end
end

D(arr) = arr .- [arr[1]; arr[1:end-1]]

function find_second_minimum(arr)
    first_maximum = arr[1]
    for i in axes(arr, 1)[2:end]
        if arr[i] > first_maximum
            maximum = arr[i]
        elseif arr[i] < first_maximum # after it has crossed the first maximum
            if arr[i] > arr[i-1] # if is starts increasing again
                return i-1
            end
        end 
    end
    return axes(arr, 1)[end]
end

function find_derivative_minimum(arr)
    second_derivative = D(D(arr))
    first_maximum = arr[1]
    for i in axes(arr, 1)[2:end]
        if arr[i] > first_maximum
            maximum = arr[i]
        elseif arr[i] < first_maximum # after it has crossed the first maximum
            if second_derivative[i] > 0 # if the decrease starts slowing down
                return i-1
            end
        end 
    end
    return axes(arr, 1)[end]
end