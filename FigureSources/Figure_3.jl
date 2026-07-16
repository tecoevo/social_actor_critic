## Load packages and required functions
include("common.jl")

## Load and process the data
df = preprocess_df(DataFrame(Arrow.Table("Data/SAC_learning_times_problem_size.arrow")))

## Make the plot

# Choose the parameters
s0 = 0.03
cov = 0.1
p = 0.5
θ = 1.
rtol = 0.4
maxn = 10

# Set plot attributes
ylims = (0.05, 20)
xlims = (-0.05, 1.05)
colormap = :cmr_bubblegum # cmr_amber, cmr_ember, cmr_bubblegum, cmr_lavender, cmr_cosmic
yticknums = Any[0.01, 0.1, 1, 10]
yticks = (yticknums, (u -> string(u)*"×").(yticknums))
xticks = ([0., 0.5, 1.0], ["0", "0.5", "1"])

df2 = @rsubset(df, :s0 == s0, :cov == cov, :p == p, :θ == θ, :rtol == rtol)

fig = Figure(size = (850, 400))
ax1 = Axis(fig[1,1]; xlabel = rich("Social learning propensity ", rich("ω", font = :italic), offset = (16, 0)), ylabel = "Init. rel. adaptation speed", yscale = log10, title = "Unbiased", yticks, xticks)
ax2 = Axis(fig[1,2]; yscale = log10, title = "Performance-biased", yticks, xticks)
colors = cgrad(colormap, maxn; categorical = true)

# Add background for learning speedup < 1
band!(ax1, [-0.1, 1.1], 0.001, 1; color = (:grey, 0.2) )
band!(ax2, [-0.1, 1.1], 0.001, 1; color = (:grey, 0.2) )

for n in 1:maxn
    local df3 = @rsubset(df2, :n == n, :bias == ^(:un))
    lines!(ax1, df3.ω, df3.learning_speedup; color = colors[n], linewidth = 3)

    local df3 = @rsubset(df2, :n == n, :bias == ^(:perf))
    lines!(ax2, df3.ω, df3.learning_speedup; color = colors[n], linewidth = 3)
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
        padding = (-20, 0, 0, 0),
        halign = :right,
        width = 0 
    )
end

save("Figure_3.pdf", fig)
display(fig)