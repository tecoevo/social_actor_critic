## Load packages and required functions
include("common.jl")

## Load and process the data
df = "Data/SAC_learning_times_expertise.arrow" |> Arrow.Table |> DataFrame |> preprocess_df |> process_df_rtols

optimal_SL_df = @chain df begin
    sort(:learning_time_incremental)
    @groupby(:n, :x0, :s0, :cov, :α, :β, :γ, :bias, :reward_target, :p, :θ)
    combine(first)
    sort(:reward_target)
    sort([:n, :s0, :cov, :bias])
end

## Make the figure

s0 = 0.03
cov = 0.1
maxn = 10

ylims = (-0.05, 0.65)
colormap = :cmr_bubblegum # cmr_amber, cmr_ember, cmr_bubblegum, cmr_lavender, cmr_cosmic

df3 = @rsubset(optimal_SL_df, :s0 == s0, :cov == cov)
all_expertise_targets = sort(unique(df3.expertise_target))

fig = Figure(size = (850, 400))
ax1 = Axis(fig[1,1]; xlabel = rich("Expertise target (%)", offset = (16, 0)), ylabel = rich("Optimal ", rich("ω", font = :italic)), title = "Unbiased", xticks = 20:20:80)
ax2 = Axis(fig[1,2]; title = "Performance biased", xticks = 20:20:80)
colors = cgrad(colormap, maxn; categorical = true)

for n in 1:maxn
    local df4 = @rsubset(df3, :n == n, :bias == ^(:un))
    lines!(ax1, df4.expertise_target, df4.ω; color = colors[n], linewidth = 3)

    local df4 = @rsubset(df3, :n == n, :bias == ^(:perf))
    lines!(ax2, df4.expertise_target, df4.ω; color = colors[n], linewidth = 3)
end
ylims!.((ax1, ax2), (ylims,))
hideydecorations!(ax2; ticks = false, grid = false, minorgrid=false)

Colorbar(fig[1,3], colormap=colors, colorrange = (0.5, maxn+0.5), ticks = 1:maxn, label = rich("Problem size ", rich("n", font = :italic)))

# Adding Panel labels
for (pos, label) in zip(1:2, ["A", "B"])
    Label(
        fig[1, pos, TopLeft()], label,
        fontsize = 21,
        font = :bold,
        padding = (-18, 0, 0, 0),
        halign = :right,
        width = 0 
    )
end

save("Figure_5.pdf", fig)
display(fig)