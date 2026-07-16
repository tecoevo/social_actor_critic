## Load packages and required functions
include("common.jl")

## Load and process the data
df = DataFrame(Arrow.Table("data/SAC_learning_times_expertise.arrow")) |> preprocess_df |> process_df_rtols


## Make the figure

s0 = 0.03
cov = 0.2

df2 = @rsubset(df, :s0 == s0, :cov == cov)
@rtransform!(df2, :expertise_target = :reward_target * 200)

all_reward_target = sort(unique(df2.reward_target))
all_expertise_target = sort(unique(df2.expertise_target))

colormap = :cmr_bubblegum
yticknums = Any[0.01, 0.1, 1, 10]
ylims = (0.005, 20)
yticks = (yticknums, (u -> string(u)*"×").(yticknums))
xticks = ([0., 0.5, 1.0], ["0", "0.5", "1"])
maxn = 10
colors = cgrad(colormap, maxn; categorical = true)
all_rtol = sort(unique(df2.rtol))

fig = Figure(size = (1200, 650))

for (k, bias) in enumerate([:un, :perf])
    local df3 = @rsubset(df2, :bias == bias)
    for (i, target) in enumerate(all_expertise_target)
        kwargs = (yticks = yticks, xticks = xticks, yscale = log10)
        (i == 1) && (kwargs = (kwargs..., xlabel = rich("Social learning propensity ", rich("ω", font = :italic), offset = (36, 0)), ylabel = rich("Relative adaptation speed", offset = (-10, 0))))
        (k == 1 && i == 1) && (kwargs = (kwargs..., title = "A. Unbiased social learning", titlealign = :left))
        (k == 2 && i == 1) && (kwargs = (kwargs..., title = "B. Performance-biased social learning", titlealign = :left))
        local ax = Axis(fig[k, i]; kwargs...)
        
        local df4 = @rsubset(df3, :expertise_target == target)
        for n in 1:maxn
            local df5 = @rsubset(df4, :n == n)
            lines!(ax, df5.ω, df5.learning_speedup_incremental; color = colors[n], linewidth = 3)
        end
        (i == 1) || hideydecorations!(ax; grid = false)
        (k == 1) && hidexdecorations!(ax; grid = false)
        (k == 2) && hideydecorations!(ax; grid = false, ticks = false, ticklabels = false)
        ylims!(ax, ylims...)
    end
end

topxticksnums = round.(Int, all_expertise_target)
topxticks = (topxticksnums, (u -> string(u)*"%").(topxticksnums))
topax = Axis(fig[0, :]; height = 0, xlabel = "Expertise target", xticks = topxticks, xaxisposition = :top)
δ = all_expertise_target[2] - all_expertise_target[1]
xlims!(topax, all_expertise_target[1] - δ/2, all_expertise_target[end] + δ/2)
hidespines!(topax, :b, :r, :l)
hidedecorations!(topax; ticks = false, ticklabels = false, label = false)
hideydecorations!(topax)

Colorbar(fig[1:end, length(all_rtol)+1]; colormap = colors, colorrange = (0.5, maxn+0.5), ticks = 1:maxn, label = rich("Problem size ", rich("n", font = :italic)))
rowgap!(fig.layout, 5)
colgap!(fig.layout, 10)

save("Figure_S2.pdf", fig)
display(fig)