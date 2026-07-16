## Load packages and required functions
include("common.jl")

## Load and process the data
df_covar = preprocess_df(DataFrame(Arrow.Table("Data/SAC_learning_times_covariance.arrow")))
df_variance = preprocess_df(DataFrame(Arrow.Table("Data/SAC_learning_times_variance.arrow")))

# Top panels
cov = 0.
n = 8
p = 0.5
θ = 1.
rtol = 0.4

colormap = :cmr_lavender #cmr_ember 
ylims = (0.05, 20)
xlims = (-0.05, 1.05)
yticknums = Any[0.01, 0.1, 1, 10]
yticks = (yticknums, (u -> string(u)*"×").(yticknums))


df3 = @rsubset(df_variance, :cov == cov, :p == p, :n == n, :θ == θ, :rtol == rtol)

all_s0 = sort(unique(df3.s0))

colors = cgrad(colormap, length(all_s0); categorical = true)

fig = Figure(size = (850, 700))
ax = Axis(fig[1,1]; yscale = log10, title = "Unbiased", yticks)
# Add background for learning speedup < 1
band!(ax, [-0.1, 1.1], 0.001, 1; color = (:grey, 0.2) )

for (i, s0) in enumerate(sort(all_s0))
    local df4 = @rsubset(df3, :s0 == s0, :bias == ^(:un))
    lines!(ax, df4.ω, df4.learning_speedup; color = colors[i], linewidth = 3, linestyle = s0 == 0.03 ? :dash : :solid)
end

ax2 = Axis(fig[1,2]; yscale = log10, title = "Performance-biased", yticks)
# Add background for learning speedup < 1
band!(ax2, [-0.1, 1.1], 0.001, 1; color = (:grey, 0.2) )

for (i, s0) in enumerate(sort(all_s0))
    local df4 = @rsubset(df3, :s0 == s0, :bias == ^(:perf))
    lines!(ax2, df4.ω, df4.learning_speedup; color = colors[i], linewidth = 3, linestyle = s0 == 0.03 ? :dash : :solid)
end

ylims!.((ax, ax2), ylims...)
xlims!.((ax, ax2), xlims...)
hideydecorations!(ax2; ticks = false, grid = false, minorgrid=false)
hidexdecorations!.((ax, ax2); ticks = false, grid = false, minorgrid = false)
δ_s0 = all_s0[2]-all_s0[1]
Colorbar(fig[1,3], colormap=colors, colorrange = (first(all_s0)-δ_s0/2, last(all_s0)+δ_s0/2), label = "Population variance")

# Bottom panels

s0 = 0.03
colormap = :cmr_fusion #cmr_prinsenvlag #cmr_pride

df5 = @rsubset(df_covar, :p == p, :θ == θ, :rtol == rtol, :n == n, :s0 == s0)
all_cov = sort(unique(df5.cov))
colors = cgrad(colormap, length(all_cov); categorical = true)[:]
colors[length(colors)÷2+1] = ColorTypes.Gray(0.5)
colors = cgrad(colors; categorical = true)
yticknums = Any[0.01, 0.1, 1, 10]

ax = Axis(fig[2,1], xlabel = rich("Social learning propensity ", rich("ω", font = :italic), offset = (15.5, 0)), ylabel = rich("Initial relative adaptation speed", offset=(14.5, 0.)), yscale = log10, yticks = (yticknums, (u -> string(u)*"×").(yticknums)))
# Add background for learning speedup < 1
band!(ax, [-0.1, 1.1], 0.001, 1; color = (:grey, 0.2) )

for (i, cov) in enumerate(sort(all_cov))
    local df6 = @rsubset(df5, :cov == cov, :bias == ^(:un))
    lines!(ax, df6.ω, df6.learning_speedup; linewidth = 3, color = cov == 0. ? :grey : colors[i], linestyle = cov == 0 ? :dash : :solid)
end

ax2 = Axis(fig[2,2], yscale = log10, yticks = (yticknums, (u -> string(u)*"×").(yticknums)))
# Add background for learning speedup < 1
band!(ax2, [-0.1, 1.1], 0.001, 1; color = (:grey, 0.2) )

for (i, cov) in enumerate(sort(all_cov))
    local df6 = @rsubset(df5, :cov == cov, :bias == ^(:perf))
    lines!(ax2, df6.ω, df6.learning_speedup; linewidth = 3, color = cov == 0. ? :grey : colors[i], linestyle = cov == 0 ? :dash : :solid)
end

ylims!.((ax, ax2), ylims...)
xlims!.((ax, ax2), xlims...)
hideydecorations!(ax2; ticks = false, grid = false, minorgrid=true)
δ_cov = all_cov[2]-all_cov[1]
Colorbar(fig[2,3], colormap=colors, colorrange = (first(all_cov)-δ_cov/2, last(all_cov)+δ_cov/2), label = "Covariance factor")

rowgap!(fig.layout, 1, 20.)
# Adding Panel labels
for (pos, label) in zip([(1, 1), (1, 2), (2, 1), (2, 2)], ["A", "B", "C", "D"])
    Label(
        fig[pos..., TopLeft()], label,
        fontsize = 21,
        font = :bold,
        padding = (-20, 0, pos[1] == 1 ? -15 : 0, 0),
        halign = :right,
        width = 0,
        height = 0
    )
end

save("Figure_4.pdf", fig)