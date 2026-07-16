## Load packages and required functions
include("common.jl")

## Load and process the data
df = DataFrame(Arrow.Table("Data/SAC_learning_times_learning_rate.arrow")) |> preprocess_df

## Make the figure

cov = 0.1
s0 = 0.03
n = 8

colormap = :cmr_ember
ylims = (0.05, 20)
yticknums = Any[0.1, 1, 10]
yticks = (yticknums, (u -> string(u)*"×").(yticknums))

df3 = @rsubset(df, :cov == cov, :s0 == s0, :n == n)

all_learning_rate = sort(unique(df3.α))

colors = cgrad(colormap, length(all_learning_rate); categorical = true)

fig = Figure(size = (850, 400))
ax = Axis(fig[1,1]; xlabel = rich("Social learning propensity ", rich("ω", font = :italic), offset = (16, 0)), ylabel = "Init. rel. adaptation speed", yscale = log10, title = "Unbiased", yticks)

for (i, rate) in enumerate(sort(all_learning_rate))
    local df4 = @rsubset(df3, :α == rate, :β == rate, :bias == ^(:un))
    idx = find_derivative_minimum(df4.learning_speedup)
    lines!(ax, df4.ω[1:idx], df4.learning_speedup[1:idx]; color = colors[i], linewidth = 3)
end

ax2 = Axis(fig[1,2]; yscale = log10, title = "Performance-biased", yticks)

for (i, rate) in enumerate(sort(all_learning_rate))
    local df4 = @rsubset(df3, :α == rate, :β == rate, :bias == ^(:perf))
    lines!(ax2, df4.ω, df4.learning_speedup; color = colors[i], linewidth = 3)
end

ylims!.((ax, ax2), ylims...)
hideydecorations!(ax2; ticks = false, grid = false, minorgrid=false)
Colorbar(fig[1,3], colormap=colors, colorrange = extrema(all_learning_rate) .+ (-0.005, 0.005), ticks = [0.01, 0.05, 0.1, 0.15, 0.2], label = rich("Learning rate ", rich("α, β", font = :italic)))

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

save("Figure_S5.pdf", fig)
display(fig)