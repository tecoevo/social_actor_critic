## Load packages and required functions
include("common.jl")

## Load data and process
df_OG = "Data/SAC_learning_times_expertise.arrow" |> Arrow.Table |> DataFrame |> preprocess_df |> process_df_rtols

optimal_SL_df_OG = @chain df_OG begin
    sort(:learning_time_incremental)
    @groupby(:n, :x0, :s0, :cov, :α, :β, :γ, :bias, :reward_target, :p, :θ)
    combine(first)
    sort(:reward_target)
    sort([:n, :s0, :cov, :bias])
end

df_CG = "Data/SAC_learning_times_policy_copying.arrow" |> Arrow.Table |> DataFrame |> preprocess_df |> process_df_rtols

optimal_SL_df_CG = @chain df_CG begin
    sort(:learning_time_incremental)
    @groupby(:n, :x0, :s0, :cov, :α, :β, :γ, :bias, :reward_target, :p, :θ)
    combine(first)
    sort(:reward_target)
    sort([:n, :s0, :cov, :bias])
end

optimal_SL_df_both = innerjoin(optimal_SL_df_OG, optimal_SL_df_CG; on = [:n, :x0, :s0, :cov, :α, :β, :γ, :bias, :reward_target, :expertise_target, :p, :θ, :full_demonstrator_choice], renamecols = "_OG" => "_CG");
@rtransform!(optimal_SL_df_both, :learning_speedup_incremental_ratio = :learning_speedup_incremental_CG / :learning_speedup_incremental_OG);

## Make the figure
s0 = 0.03
cov = 0.1
maxn = 10

ylims = (0.95, 1.65)
colormap = :RdYlBu_4 #Spectral_4  

df3 = @rsubset(optimal_SL_df_both, :s0 == s0, :cov == cov)
all_expertise_targets = sort(unique(df3.expertise_target))

yticknums = Any[1, 1.2, 1.4, 1.6]
yticks = (yticknums, (u -> string(u)*"×").(yticknums))

fig = Figure(size = (850, 400))
ax1 = Axis(fig[1,1]; xlabel = rich("Problem size ", rich("n", font = :italic), offset = (15.5, 0)), ylabel = "Relative learning speed", title = "Unbiased", xticks = [1; 2:2:10], yticks)
ax2 = Axis(fig[1,2]; title = "Performance biased", xticks = [1; 2:2:10], yticks)
colors = cgrad(colormap, length(all_expertise_targets); categorical = true)

for (i, target) in enumerate(all_expertise_targets)
    df4 = @rsubset(df3, :expertise_target == target, :bias == ^(:un))
    lines!(ax1, df4.n, df4.learning_speedup_incremental_ratio; color = colors[i], linewidth = 3)

    df4 = @rsubset(df3, :expertise_target == target, :bias == ^(:perf))
    lines!(ax2, df4.n, df4.learning_speedup_incremental_ratio; color = colors[i], linewidth = 3)
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

save("Figure_S14.pdf", fig)
display(fig)