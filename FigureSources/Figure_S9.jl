## Load packages and required functions
include("common.jl")

## Load and process the data
df_1 = "Data/SAC_learning_times_expertise.arrow" |> Arrow.Table |> DataFrame |> preprocess_df |> process_df_rtols

optimal_SL_df = @chain df begin
    sort(:learning_time_incremental)
    @groupby(:n, :x0, :s0, :cov, :α, :β, :γ, :bias, :reward_target, :p, :θ)
    combine(first)
    sort(:reward_target)
    sort([:n, :s0, :cov, :bias])
end

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

optimal_equivalent_SL_df = innerjoin(optimal_SL_df, equivalent_SL_df; on = [:n, :x0, :s0, :cov, :α, :β, :γ, :bias, :reward_target, :expertise_target, :p, :θ, :rtol, :Tmax, :stability_threshold, :soltol], renamecols = ("_optimal" => "_equivalent"))

## Make the figure

s0 = 0.03
cov = 0.1
maxn = 10

colormap = :cmr_bubblegum
ylims = (-0.05, 1.05)
xlims = (-0.05, 0.65)
scale_markers = x-> @. 10log10(x) + 3

df3 = @rsubset(optimal_equivalent_SL_df, :s0 == s0, :cov == cov)
all_expertise_targets = sort(unique(df3.expertise_target))
all_n = sort(unique(df3.n))
colors = cgrad(colormap, length(all_n); categorical = true)

fig = Figure(size = (1250, 400))

for (i, target) in enumerate(all_expertise_targets)
    kwargs = (;)

    i == 1 && (kwargs = (; kwargs..., xlabel = rich("Optimal ", rich("ω", font = :italic), offset = (39, 0)), ylabel = rich("Equivalent ", rich("ω", font = :italic))))
    i == 2 && (kwargs = (; kwargs..., xticks = 0:0.2:0.6, yticks = 0:0.2:0.6))
    i == 3 && (kwargs = (; kwargs..., xticks = 0:0.1:0.2, yticks = 0:0.1:0.2))
    # i == 4 && (kwargs = (; kwargs..., xticks = 0:0.1:0.2, yticks = 0:0.1:0.2))


    local ax = Axis(fig[1, i]; kwargs...)
    i == 1 && (xlims!(ax, -0.1, 1.1); ylims!(ax, -0.1, 1.1))
    i == 2 && (xlims!(ax, -0.063, 0.63); ylims!(ax, -0.063, 0.63))
    i == 3 && (xlims!(ax, -0.022, 0.22); ylims!(ax, -0.022, 0.22))
    i == 4 && (xlims!(ax, -0.011, 0.11); ylims!(ax, -0.011, 0.11))

    local df4 = @rsubset(df3, :expertise_target == target, :bias == ^(:un))
    scatterlines!(ax, df4.ω_optimal, df4.ω_equivalent; color = df4.n, colormap = colors, linewidth = 3, markersize = 15, label = "Unbiased")

    local df4 = @rsubset(df3, :expertise_target == target, :bias == ^(:perf))
    scatterlines!(ax, df4.ω_optimal, df4.ω_equivalent; color = df4.n, colormap = colors, linewidth = 3, markersize = 15, marker = :utriangle, label = "Performance biased")
    i == 1 && axislegend(ax; position = :rb)
end

Colorbar(fig[1,length(all_expertise_targets)+1], colormap=colors, colorrange = extrema(all_n) .+ (-0.5, 0.5), label = rich("Problem size ", rich("n", font = :italic)))

topxticknums = round.(Int, all_expertise_targets)
topxticks = (topxticknums, (u -> string(u)*"%").(topxticknums))
topax = Axis(fig[0, 1:4]; height = 0, xlabel = "Expertise target", xticks = topxticks, xaxisposition = :top)
δ = all_expertise_targets[2] - all_expertise_targets[1]
xlims!(topax, all_expertise_targets[1] - δ/2, all_expertise_targets[end] + δ/2)
hidespines!(topax, :b, :r, :l)
hidedecorations!(topax; ticks = false, ticklabels = false, label = false)
hideydecorations!(topax)

colgap!(fig.layout, 10.)
rowgap!(fig.layout, 10.)

save("Figure_S9.pdf", fig)
display(fig)