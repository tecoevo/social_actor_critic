## Load packages and required functions
include("common.jl")

## Load and process the data
df = DataFrame(Arrow.Table("Data/SAC_learning_times_discount_factor.arrow")) |> preprocess_df |> process_df_rtols

## Make the figure

cov = 0.
s0 = 0.03
n = 8
rtol = 0.4

colormap = :cmr_ember
ylims = (0.05, 20)
yticknums = Any[0.1, 1, 10]
yticks = (yticknums, (u -> string(u)*"×").(yticknums))

df3 = @rsubset(df, :cov == cov, :s0 == s0, :n == n, :rtol == rtol)

all_discount_factor = sort(unique(df3.γ))

colors = cgrad(colormap, length(all_discount_factor); categorical = true)

fig = Figure(size = (850, 400))
ax = Axis(fig[1,1]; xlabel = rich("Social learning propensity ", rich("ω", font = :italic), offset = (16, 0)), ylabel = "Init. rel. adaptation speed", yscale = log10, title = "Unbiased", yticks)

for (i, γ) in enumerate(sort(all_discount_factor))
    local df4 = @rsubset(df3, :γ == γ, :bias == ^(:un))
    idx = find_derivative_minimum(df4.learning_speedup)
    lines!(ax, df4.ω[1:idx], df4.learning_speedup_incremental[1:idx]; color = colors[i], linewidth = 3)
end

ax2 = Axis(fig[1,2]; yscale = log10, title = "Performance-biased", yticks)

for (i, γ) in enumerate(sort(all_discount_factor))
    local df4 = @rsubset(df3, :γ == γ, :bias == ^(:perf))
    lines!(ax2, df4.ω, df4.learning_speedup; color = colors[i], linewidth = 3)
end

ylims!.((ax, ax2), ylims...)
hideydecorations!(ax2; ticks = false, grid = false, minorgrid=false)
Colorbar(fig[1,3], colormap=colors, colorrange = extrema(all_discount_factor) .+ (-0.05, 0.05), ticks = 0.1:0.2:0.9, label = rich("Discount factor ", rich("γ", font = :italic)))

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

save("Figure_S6.pdf", fig)
display(fig)