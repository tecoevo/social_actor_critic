## Load packages and required functions
include("common.jl")

function preprocess_df(df::DataFrame)
    @chain df begin
        parse_beta_binomial_dist() # convert initial distributions to mean and shape
        flatten([:reward_target, :learning_time]) # flatten vector of different rtols to their own rows
        # removing entries which does not converge for pure asocial learning
        sort(:ω)
        @groupby(:n, :init, :s0, :α, :β, :bias, :reward_target, :p, :θ, :project_manifold)
        @subset(:learning_time[1] > 0)
        @rsubset(:learning_time > 0)
        # calculating relative learning speedup
        sort(:ω)
        @groupby(:n, :init, :s0, :α, :β, :bias, :reward_target, :p, :θ, :project_manifold)
        @transform(:relative_learning_time = :learning_time ./ :learning_time[1])
        @rtransform(:learning_speedup = 1/:relative_learning_time)
    end
end

function process_df_rtols(df::DataFrame)
    @chain df begin
        sort([:reward_target, :ω])
        @groupby(:n, :init, :s0, :ω, :α, :β, :bias, :p, :θ, :project_manifold)
        @transform(:learning_time_incremental = [:learning_time[1]; :learning_time[2:end] .- :learning_time[1:end-1]]; ungroup = true)
        @groupby(:n, :init, :s0, :α, :β, :bias, :reward_target, :p, :θ, :project_manifold)
        @transform(:relative_learning_time_incremental = :learning_time_incremental ./ :learning_time_incremental[1])
        @rtransform(:learning_speedup_incremental = 1/:relative_learning_time_incremental)
        @groupby(:n, :init, :s0, :α, :β, :bias, :reward_target, :p, :θ, :project_manifold)
        @transform(:speedup_change = [:learning_speedup_incremental[1]; :learning_speedup_incremental[2:end] .- :learning_speedup_incremental[1:end-1]]; ungroup = false)
        @rsubset(:ω < 0.9 || :speedup_change < 0)
        @select(Not([:speedup_change]))
    end
end

## Load data and process
df_mon_pol = @chain "Data/SAC_tpnA_learning_times.arrow" begin
    Arrow.Table()
    DataFrame()
    @rtransform(:reward_target = round.(0.5 .- :rtol; digits=1))
    @select(Not([:rtol]))
    preprocess_df()
    process_df_rtols()
end


df_unif_ratio = @chain "Data/SAC_tpnA_learning_times_uniform_ratio.arrow" begin
    Arrow.Table()
    DataFrame()
    @rtransform(:reward_target = [0.1, 0.2, 0.3, 0.4])
    preprocess_df()
    process_df_rtols()
end;

## Make the figures

s0 = 0.1
reward_target = 0.1
initial_dist = :monotone
project_manifold = false
p = 0.5
θ = 1.

ylims = (0.05, 2)
colormap = :cmr_cosmic # cmr_amber, cmr_ember, cmr_bubblegum, cmr_lavender, cmr_cosmic
yticknums = Any[0.01, 0.1, 1, 10]
yticks = (yticknums, (u -> string(u)*"×").(yticknums))

df3 = @rsubset(df_mon_pol, :s0 == s0, :reward_target == reward_target, :init == initial_dist, :p == p, :θ == θ, :project_manifold == project_manifold)
all_ns = sort(unique(df3.n))
colors = cgrad(colormap, all_ns[end]; categorical = true)

fig = Figure(size = (850, 700))
ax1 = Axis(fig[1,1]; yscale = log10, title = "Unbiased", yticks)
ax2 = Axis(fig[1,2]; yscale = log10, title = "Performance biased", yticks)

for n in all_ns
    local df4 = @rsubset(df3, :n == n, :bias == ^(:un))
    lines!(ax1, df4.ω, df4.learning_speedup_incremental; color = colors[n], linewidth = 3)

    local df4 = @rsubset(df3, :n == n, :bias == ^(:perf))
    lines!(ax2, df4.ω, df4.learning_speedup_incremental; color = colors[n], linewidth = 3)
end
ylims!.((ax1, ax2), (ylims,))
hideydecorations!(ax2; ticks = false, grid = false, minorgrid=false)
hidexdecorations!.((ax1, ax2); ticks = false, grid = false, minorgrid = false)

# Bottom panels

initial_dist = :uniform
project_manifold = true

ylims = (0.005, 2)
colormap = :cmr_cosmic # cmr_amber, cmr_ember, cmr_bubblegum, cmr_lavender, cmr_cosmic

df3 = @rsubset(df_unif_ratio, :s0 == s0, :reward_target == reward_target, :init == initial_dist, :p == p, :θ == θ, :project_manifold == project_manifold)

ax1 = Axis(fig[2,1]; xlabel = rich("Social learning propensity ", rich("ω", font = :italic), offset = (16, 0)), ylabel = rich("Initial relative learning speed", offset = (15, 0)), yscale = log10, yticks)
ax2 = Axis(fig[2,2]; yscale = log10, yticks)

for n in all_ns
    local df4 = @rsubset(df3, :n == n, :bias == ^(:un))
    lines!(ax1, df4.ω, df4.learning_speedup; color = colors[n], linewidth = 3)

    local df4 = @rsubset(df3, :n == n, :bias == ^(:perf))
    lines!(ax2, df4.ω, df4.learning_speedup; color = colors[n], linewidth = 3)
end
ylims!.((ax1, ax2), (ylims,))
hideydecorations!(ax2; ticks = false, grid = false, minorgrid=false)

Colorbar(fig[1:2,3], colormap=colors, colorrange = (0.5, all_ns[end]+0.5), ticks = all_ns, label = rich("Problem size ", rich("n", font = :italic)))

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

save("Figure_S12.pdf", fig)
display(fig)