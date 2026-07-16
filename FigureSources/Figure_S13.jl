## Load packages and required functions
include("common.jl")

## Load data and process
df_CG = preprocess_df(DataFrame(Arrow.Table("data/SAC_learning_times_policy_copying.arrow")))

df_OG = preprocess_df(DataFrame(Arrow.Table("data/SAC_learning_times_problem_size.arrow")))

df_both = innerjoin(df_CG, df_OG; on = [:n, :x0, :s0, :cov, :ω, :α, :β, :γ, :bias, :rtol, :p, :θ], renamecols = ("_dem" => "_OG"))
@rtransform!(df_both, :learning_speedup_ratio = :learning_speedup_dem / :learning_speedup_OG);

## Make the figure

s0 = 0.03
cov = 0.1
p = 0.5
θ = 1.
rtol = 0.4
maxn = 10

ylims = (0.15, 7.5)
xlims = (-0.05, 1.05)
colormap = :cmr_ember #  cmr_ember, cmr_toxic
yticknums = Any[0.01, 0.1, 0.2, 1, 5, 10]
yticks = (yticknums, (u -> string(u)*"×").(yticknums))
xticks = ([0., 0.5, 1.0], ["0", "0.5", "1"])

df3 = @rsubset(df_both, :s0 == s0, :cov == cov, :p == p, :θ == θ, :rtol == rtol)

fig = Figure(size = (850, 400))
ax1 = Axis(fig[1,1]; xlabel = rich("Social learning propensity ", rich("ω", font = :italic), offset = (16, 0)), ylabel = "Init. rel. adaptation speed", yscale = log10, title = "Unbiased", yticks, xticks)
ax2 = Axis(fig[1,2]; yscale = log10, title = "Performance-biased", yticks, xticks)
colors = cgrad(colormap, maxn; categorical = true)

# Add background for learning speedup < 1
band!(ax1, [-0.1, 1.1], 0.001, 1; color = (:grey, 0.2) )
band!(ax2, [-0.1, 1.1], 0.001, 1; color = (:grey, 0.2) )

for n in 1:maxn
    local df4 = @rsubset(df3, :n == n, :bias == ^(:un))
    lines!(ax1, df4.ω, df4.learning_speedup_ratio; color = colors[n], linewidth = 3)

    local df4 = @rsubset(df3, :n == n, :bias == ^(:perf))
    lines!(ax2, df4.ω, df4.learning_speedup_ratio; color = colors[n], linewidth = 3)
end
ylims!.((ax1, ax2), (ylims,))
xlims!.((ax1, ax2), (xlims,))
hideydecorations!(ax2; ticks = false, grid = false, minorgrid=false)

Colorbar(fig[1,3], colormap=colors, colorrange = (0.5, maxn+0.5), ticks = 1:maxn, label = rich("Problem size ", rich("n", font = :italic)))

# Adding Panel labels
for (pos, label) in zip(1:2, ["A", "B"])
    Label(
        fig[1, pos, TopLeft()], label,
        fontsize = 21,
        font = :bold,
        padding = (-19, 0, 0, 0),
        halign = :right,
        width = 0 
    )
end

save("Figure_S13.pdf", fig)
display(fig)