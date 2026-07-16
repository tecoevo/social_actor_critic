## Load packages and required functions
include("common.jl")

## Load and process the data
df = preprocess_df(DataFrame(Arrow.Table("Data/SAC_learning_times_covariance.arrow")))

## Make the figure
p = 0.5
θ = 1.
rtol = 0.4
bias = :perf

colormap = :cmr_fusion #cmr_prinsenvlag #cmr_pride
ylims = (0.08, 20.)

df3 = @rsubset(df, :p == p, :θ == θ, :rtol == rtol, :bias == bias)
all_s0 = sort(unique(df3.s0))
all_cov = sort(unique(df3.cov))
subset_ns = 2:2:10

colors = cgrad(colormap, length(all_cov); categorical = true)
yticknums = Any[0.01, 0.1, 1, 10]

fig = Figure(size = (1100, 900))


for (i, n) in enumerate(subset_ns)
    for (j, s0) in enumerate(all_s0)
        kwargs = (yscale = log10, yticks = (yticknums, (u -> string(u)*"×").(yticknums)), yaxisposition = :right, xaxisposition = :top, xticks = ([0, 0.5, 1], ["0", "0.5", "1"]))
        (j == 3 && n == first(subset_ns)) && (kwargs = (kwargs..., xlabel = rich("Social learning propensity ", rich("ω", font = :italic))))
        (s0 == last(all_s0) && i == 3) && (kwargs = (kwargs..., ylabel = rich("Initial relative adaptation speed")))
        
        local ax = Axis(fig[i, j]; kwargs...)
        for (k, cov) in enumerate(sort(all_cov))
            local df4 = @rsubset(df3, :cov == cov, :n == n, :s0 == s0)
            lines!(ax, df4.ω, df4.learning_speedup; color = colors[k], linewidth = 3)
        end
        (s0 == last(all_s0)) || hideydecorations!(ax; grid = false)
        (n == first(subset_ns)) || hidexdecorations!(ax; grid = false)
        ylims!(ax, ylims...)
    end
end

colgap!(fig.layout, 10)
rowgap!(fig.layout, 10)

leftax = Axis(fig[:, 0]; width = 0, ylabel = rich("Problem size ", rich("n", font = :italic)), yreversed = true, yticks = 2:2:10)
ylims!(leftax, 11, 1)
hidespines!(leftax, :t, :r, :b)
hidedecorations!(leftax; ticks = false, ticklabels = false, label = false)
hidexdecorations!(leftax)

bottomax = Axis(fig[length(subset_ns)+1, 1:end]; height = 0, xlabel = "Population variance", xticks = all_s0)
δ = all_s0[2] - all_s0[1]
xlims!(bottomax, first(all_s0) - δ/2, last(all_s0) + δ/2)
hidespines!(bottomax, :t, :r, :l)
hidedecorations!(bottomax; ticks = false, ticklabels = false, label = false)
hideydecorations!(bottomax)


Colorbar(fig[1:end-1, length(all_s0)+1], colormap=colors, colorrange = extrema(all_cov) .+ (-0.01, 0.01), label = "Covariance factor")

save("Figure_S4.pdf", fig)
display(fig)