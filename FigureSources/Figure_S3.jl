## Load packages and required functions
include("common.jl")

## Load and process the data
df = preprocess_df(DataFrame(Arrow.Table("Data/SAC_learning_times_start_distribution.arrow")))

## Make the figure

s0 = 0.03
cov = 0.
rtol = 0.4
bias = :perf

colormap = :cmr_bubblegum #cmr_prinsenvlag #cmr_pride
ylims = (0.08, 18.)

df3 = @rsubset(df, :s0 == s0, :cov == cov, :rtol == rtol, :bias == bias)
subset_p = 0.1:0.1:0.5
all_θ = sort(unique(df3.θ))
all_ns = sort(unique(df3.n))
all_ns = 1:10

colors = cgrad(colormap, length(all_ns); categorical = true)
yticknums = Any[0.01, 0.1, 1, 10]

fig = Figure(size = (1100, 900))

for (i, p) in enumerate(subset_p)
    for (j, θ) in enumerate(all_θ)
        kwargs = (yscale = log10, yticks = (yticknums, (u -> string(u)*"×").(yticknums)), yaxisposition = :right, xaxisposition = :top, xticks = ([0, 0.5, 1], ["0", "0.5", "1"]))
        (j == 3 && p == first(subset_p)) && (kwargs = (kwargs..., xlabel = rich("Social learning propensity ", rich("ω", font = :italic))))
        (θ == last(all_θ) && i == 3) && (kwargs = (kwargs..., ylabel = "Initial relative adaptation speed"))
        
        local ax = Axis(fig[i, j]; kwargs...)
        for (k, n) in enumerate(sort(all_ns))
            local df4 = @rsubset(df3, :θ == θ, :n == n, :p == p)
            lines!(ax, df4.ω, df4.learning_speedup; color = colors[k], linewidth = 3)
        end
        (θ == last(all_θ)) || hideydecorations!(ax; grid = false)
        (p == first(subset_p)) || hidexdecorations!(ax; grid = false)
        ylims!(ax, ylims...)

        inset_ax = Axis(fig[i, j]; width=Relative(0.2), height=Relative(0.2), halign=0.1, valign=0.1)
        inset_x = range(0, 1, 100)
        inset_y = pdf(scaled_beta_dist(p, θ), inset_x)
        lines!(inset_ax, inset_x, inset_y; color = :grey)
        hidedecorations!(inset_ax)
        hidespines!(inset_ax)
    end
end

colgap!(fig.layout, 10)
rowgap!(fig.layout, 10)

leftax = Axis(fig[:, 0]; width = 0, ylabel = rich("Start distribution mean ", rich("p", font = :italic)), yreversed = true, yticks = subset_p)
δ = subset_p[2] - subset_p[1]
ylims!(leftax, last(subset_p) + δ/2, first(subset_p) - δ/2)
hidespines!(leftax, :t, :r, :b)
hidedecorations!(leftax; ticks = false, ticklabels = false, label = false)
hidexdecorations!(leftax)

bottomax = Axis(fig[length(subset_p)+1, 1:end]; height = 0, xlabel = rich("Start distribution shape ", rich("θ", font = :italic)), xticks = (1:length(all_θ), string.(Int.(all_θ))))
xlims!(bottomax, 1/2, length(all_θ) + 1/2)
hidespines!(bottomax, :t, :r, :l)
hidedecorations!(bottomax; ticks = false, ticklabels = false, label = false)
hideydecorations!(bottomax)

Colorbar(fig[1:end-1, length(all_θ)+1], colormap=colors, colorrange = extrema(all_ns) .+ (-0.5, 0.5), ticks = all_ns, label = rich("Problem size ", rich("n", font = :italic)))

save("Figure_S3.pdf", fig)
display(fig)