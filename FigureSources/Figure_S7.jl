## Load packages and required functions
include("common.jl")

## Load and process the data
df = "data/SAC_learning_times_expertise.arrow" |> Arrow.Table |> DataFrame |> preprocess_df |> process_df_rtols

optimal_SL_df = @chain df begin
    sort(:learning_time_incremental)
    @groupby(:n, :x0, :s0, :cov, :α, :β, :γ, :bias, :reward_target, :p, :θ)
    combine(first)
    sort(:reward_target)
    sort([:n, :s0, :cov, :bias])
end

optimal_SL_SLS_df = @chain optimal_SL_df begin
    sort(:bias)
    @groupby(:n, :s0, :cov, :expertise_target, :p, :θ, :full_demonstrator_choice)
    @combine(:learning_speedup_incremental_SLS = :learning_speedup_incremental[1] / :learning_speedup_incremental[2])
end;

## Make the plot
s0 = 0.03
cov = 0.1
maxn = 10

ylims = (-0.05, 0.65)
colormap = :RdYlBu_4 #Spectral_4 #RdYlBu_4 # 

yticknums = Any[1, 1.5, 2, 2.5]
yticks = (yticknums, (u -> string(u)*"×").(yticknums))

df3 = @rsubset(optimal_SL_df, :s0 == s0, :cov == cov)
df3_B = @rsubset(optimal_SL_SLS_df, :s0 == s0, :cov == cov)
all_expertise_targets = sort(unique(df3.expertise_target))

fig = Figure(size = (1300, 400))
ax1 = Axis(fig[1,1]; ylabel = rich("Optimal ", rich("ω", font = :italic)), title = "Unbiased", xticks = [1; 2:2:10])
ax2 = Axis(fig[1,2]; title = "Performance biased", xlabel = rich("Problem size ", rich("n", font = :italic)), xticks = [1; 2:2:10])
ax3 = Axis(fig[1,3]; title = "SLS comparison", ylabel = "Relative learning speed", xticks = [1; 2:2:10], yticks)
colors = cgrad(colormap, length(all_expertise_targets); categorical = true)

for (i, target) in enumerate(all_expertise_targets)
    df4 = @rsubset(df3, :expertise_target == target, :bias == ^(:un))
    lines!(ax1, df4.n, df4.ω; color = colors[i], linewidth = 3)

    df4 = @rsubset(df3, :expertise_target == target, :bias == ^(:perf))
    lines!(ax2, df4.n, df4.ω; color = colors[i], linewidth = 3)

    df4_B = @rsubset(df3_B, :expertise_target == target)
    lines!(ax3, df4_B.n, df4_B.learning_speedup_incremental_SLS; color = colors[i], linewidth = 3)
end
ylims!.((ax1, ax2), (ylims,))
hideydecorations!(ax2; ticks = false, grid = false, minorgrid=false)

Colorbar(fig[1,4], colormap=colors, colorrange = (10, 90), ticks = 20:20:80, label = "Expertise target (%)")

# Adding Panel labels
for (pos, label) in zip(1:3, ["A", "B","C"])
    Label(
        fig[1, pos, TopLeft()], label,
        fontsize = 21,
        font = :bold,
        padding = (pos == 3 ? -63 : -18, 0, 0, 0),
        halign = :right,
        width = 0 
    )
end

colgap!(fig.layout, 2, 10.)

save("Figure_S7.pdf", fig)
display(fig)