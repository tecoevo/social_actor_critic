## Load packages and required functions
include("common.jl")

## Load data and process

df = mapreduce(vcat, ["problem_size", "variance", "covariance", "start_distribution"]) do name
    DataFrame(Arrow.Table("Data/SAC_learning_times_$name.arrow"))
end

df = @chain df begin
    parse_beta_binomial_dist()  # convert initial distributions to mean and shape
    flatten([:rtol, :learning_time]) # flatten vector of different rtols to their own rows
    # removing duplicates
    sort([:Tmax, :soltol, :stability_threshold]; rev = [true, false, true]) 
    @groupby(Not(:Tmax, :stability_threshold, :soltol, :learning_time))
    combine(first)
    # removing entries which does not converge
    @rsubset(:learning_time > 0)
end

optimal_SL_df = @chain df begin
    sort(:learning_time)
    @groupby(:n, :x0, :s0, :cov, :α, :β, :γ, :bias, :rtol, :p, :θ)
    combine(first)
    sort([:n, :s0, :cov, :bias])
    rename(:ω => :optimal_ω)
end

## Make the figures

fig = Figure(size = (1300, 650))

colormap = :cmr_fusion #vik, cmr_prinsenvlag, roma, cmr_fusion
subset_s0 = 0.01:0.01:0.05

p = 0.5
θ = 1.0
γ = 0.9
rtol = 0.4

df2 = @rsubset(optimal_SL_df, :p == p, :θ == θ, :γ == γ, :rtol == rtol)

for (k, bias) in enumerate([:un, :perf])
    local df3 = @rsubset(df2, :bias == bias)
    for (i, s0) in enumerate(subset_s0)
        kwargs = (yticks = ([0., 0.5, 1.0], ["0", "0.5", "1"]), xticks = [1, 2, 4, 6, 8, 10])
        (i == 1) && (kwargs = (kwargs..., ylabel = rich("Optimal ", rich("ω", font = :italic), offset = (-8, 0))))
        (i == 3 && k == 2) && (kwargs = (; kwargs..., xlabel = rich("Problem size ", rich("n", font = :italic))))
        (k == 1 && i == 1) && (kwargs = (kwargs..., title = "A. Unbiased social learning", titlealign = :left))
        (k == 2 && i == 1) && (kwargs = (kwargs..., title = "B. Performance biased social learning", titlealign = :left))
        local ax = Axis(fig[k, i]; kwargs...)
        local df4 = @rsubset(df3, :s0 == s0)
        all_cov = sort(unique(df4.cov))
        colors = cgrad(colormap, length(all_cov); categorical = true)
        for (j, cov) in enumerate(all_cov)
            local df5 = @rsubset(df4, :cov == cov)
            lines!(ax, df5.n, df5.optimal_ω; color = colors[j], linewidth = 3)
        end
        (i == 1) || hideydecorations!(ax; grid = false)
        (k == 1) && hidexdecorations!(ax; grid = false)
        (k == 2) && hideydecorations!(ax; grid = false, ticks = false, ticklabels = false)
        ylims!(ax, -0.05, 1.05)
        xlims!(ax, 0.5, 10.5)
    end
end

topax = Axis(fig[0, :]; height = 0, xlabel = "Population variance", xticks = subset_s0, xaxisposition = :top)
δ = subset_s0[2] - subset_s0[1]
xlims!(topax, first(subset_s0) - δ/2, last(subset_s0) + δ/2)
hidespines!(topax, :b, :r, :l)
hidedecorations!(topax; ticks = false, ticklabels = false, label = false)
hideydecorations!(topax)

Colorbar(fig[1:end, length(subset_s0)+1]; colormap, colorrange = (-0.2, 0.2), label = "Covariance factor")
rowgap!(fig.layout, 5)
colgap!(fig.layout, 10)

save("Figure_S10.pdf", fig)
display(fig)