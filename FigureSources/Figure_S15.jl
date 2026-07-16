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

colormap = :cmr_bubblegum
ylims = (-0.05, 1.05)
xlims = (-0.05, 0.65)

df3 = @rsubset(optimal_SL_df_both, :s0 == s0, :cov == cov)
all_expertise_targets = sort(unique(df3.expertise_target))
all_n = sort(unique(df3.n))
colors = cgrad(colormap, length(all_n); categorical = true)

fig = Figure(size = (1250, 700))

for (k, bias) in enumerate([:un, :perf])
    for (i, target) in enumerate(all_expertise_targets)
        kwargs = (;)

        (k == 1 && i == 1) && (kwargs = (; kwargs..., title = "A. Unbiased social learning", titlealign = :left))
        (k == 2 && i == 1) && (kwargs = (; kwargs..., title = "B. Performance biased social learning", titlealign = :left))
        (k == 2 && i == 1) && (kwargs = (; kwargs..., xlabel = rich("Optimal ", rich("ω", font = :italic), " action copying", offset = (39, 0)), ylabel = rich("Optimal ", rich("ω", font = :italic), " policy copying", offset = (14, 0))))
        i == 1 && (kwargs = (; kwargs..., xticks = 0:0.2:0.6, yticks = 0:0.2:0.6))
        i == 2 && (kwargs = (; kwargs..., xticks = 0:0.2:0.6, yticks = 0:0.2:0.6))
        i == 3 && (kwargs = (; kwargs..., xticks = 0:0.1:0.2, yticks = 0:0.1:0.2))
        i == 4 && (kwargs = (; kwargs..., xticks = 0:0.1:0.2, yticks = 0:0.1:0.2))


        local ax = Axis(fig[k, i]; kwargs...)
        i == 1 && (xlims!(ax, -0.065, 0.65); ylims!(ax, -0.065, 0.65))
        i == 2 && (xlims!(ax, -0.022, 0.22); ylims!(ax, -0.022, 0.22))
        i == 3 && (xlims!(ax, -0.011, 0.11); ylims!(ax, -0.011, 0.11))
        i == 4 && (xlims!(ax, -0.011, 0.11); ylims!(ax, -0.011, 0.11))

        local df4 = @rsubset(df3, :expertise_target == target, :bias == bias)
        scatterlines!(ax, df4.ω_OG, df4.ω_CG; color = df4.n, colormap = colors, linewidth = 3, markersize = 15)
    end
end

Colorbar(fig[1:2,length(all_expertise_targets)+1], colormap=colors, colorrange = extrema(all_n) .+ (-0.5, 0.5), label = rich("Problem size ", rich("n", font = :italic)))

topxticknums = round.(Int, all_expertise_targets)
topxticks = (topxticknums, (u -> string(u)*"%").(topxticknums))
topax = Axis(fig[0, 1:4]; height = 0, xlabel = "Expertise target", xticks = topxticks, xaxisposition = :top)
δ = all_expertise_targets[2] - all_expertise_targets[1]
xlims!(topax, all_expertise_targets[1] - δ/2, all_expertise_targets[end] + δ/2)
hidespines!(topax, :b, :r, :l)
hidedecorations!(topax; ticks = false, ticklabels = false, label = false)
hideydecorations!(topax)

colgap!(fig.layout, 10.)
rowgap!(fig.layout, 1, 5.)
rowgap!(fig.layout, 2, 0.)

save("Figure_S15.pdf", fig)
display(fig)