## Load packages and required functions
include("common.jl")

## Load and process the data
df = "data/SAC_learning_times_expertise.arrow" |> Arrow.Table |> DataFrame |> preprocess_df |> process_df_rtols

equivalent_SL_df = @chain df begin
    @rtransform(:reward_target = 0.5 - :rtol)
    sort(:ω)
    @groupby(:n, :x0, :s0, :cov, :α, :β, :γ, :bias, :reward_target, :p, :θ)
    @subset(length(:ω) > 1 && first(:ω) == 0.; ungroup = false)
end

equivalent_idx = [findlast(df.learning_speedup_incremental .>= 1.) for df in equivalent_SL_df]
equivalent_SL_df = map(zip(equivalent_idx, equivalent_SL_df)) do (idx, df)
    df[idx, :]
end

equivalent_SL_df = @chain equivalent_SL_df begin
    DataFrame()
    sort([:n, :s0, :cov, :p, :θ, :reward_target])
end

## Make the figure

s0 = 0.03
cov = 0.1
maxn = 10

ylims = (-0.05, 1.05)
xticks = [1; 2:2:10]
colormap = :RdYlBu_4 # 

df3 = @rsubset(equivalent_SL_df, :s0 == s0, :cov == cov)
all_expertise_targets = sort(unique(df3.expertise_target))

fig = Figure(size = (850, 400))
ax1 = Axis(fig[1,1]; xlabel = rich("Problem size ", rich("n", font = :italic), offset = (16.5, 0)), ylabel = rich("Equivalent ", rich("ω", font = :italic)), title = "Unbiased", xticks)
ax2 = Axis(fig[1,2]; title = "Performance biased", xticks)
colors = cgrad(colormap, length(all_expertise_targets); categorical = true)

for (i, target) in enumerate(all_expertise_targets)
    local df4 = @rsubset(df3, :expertise_target == target, :bias == ^(:un))
    lines!(ax1, df4.n, df4.ω; color = colors[i], linewidth = 3)

    local df4 = @rsubset(df3, :expertise_target == target, :bias == ^(:perf))
    lines!(ax2, df4.n, df4.ω; color = colors[i], linewidth = 3)
end
ylims!.((ax1, ax2), (ylims,))
hideydecorations!(ax2; ticks = false, grid = false, minorgrid=false)

Colorbar(fig[1,3], colormap=colors, colorrange = (10, 90), ticks = 20:20:80, label = "Expertise target (%)")

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

save("Figure_S8.pdf", fig)
display(fig)