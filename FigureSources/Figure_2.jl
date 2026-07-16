## Load packages and required functions
include("common.jl")

function merge_close(solutions::Vector{Float64}, counts::Vector{Int}, tol::Float64)
    # sort solutions and reorder counts
    p = sortperm(solutions; rev = true)
    sols = solutions[p]
    cnts = counts[p]

    merged_solutions = Float64[]
    merged_counts = Int[]
    merged_indices = Int[1]

    current_sol = sols[1]
    current_count = cnts[1]

    for i in 2:length(sols)
        if abs(sols[i] - current_sol) ≤ tol
            current_count += cnts[i]
        else
            push!(merged_solutions, current_sol)
            push!(merged_counts, current_count)
            push!(merged_indices, i)
            current_sol = sols[i]
            current_count = cnts[i]
        end
    end

    # push last group
    push!(merged_solutions, current_sol)
    push!(merged_counts, current_count)

    return merged_solutions, merged_counts, merged_indices
end

## Load the data

df = DataFrame(Arrow.Table("Data/SAC_steady_state.arrow"))

## Process the data

df_basins = @chain df begin
    parse_beta_binomial_dist()
    @rtransform(:steady_state = map(u->u[1:(:n)], :steady_state))
    @rtransform(:passes = 0 .< :rewards .<= 1.)
    @rtransform(:rewards = :rewards[:passes], :counts = :counts[:passes], :steady_state = :steady_state[:passes])
    @rsubset(!isempty(:rewards))
    @select(Not([:passes, :α, :β, :γ, :soltol, :uniqtol, :random_seed, :N_ensemble, :stability_threshold, :bias, :check_bounds]))
    @rtransform(:merge_close = merge_close(:rewards, :counts, 1e-2))
    @rtransform(:rewards = :merge_close[1], :counts = :merge_close[2], :steady_state = :steady_state[:merge_close[3]])
    @select(Not(:merge_close))
    @rtransform(:basin_2 = map(((ss, r),) -> any(ss .< 1e-3) && r < 0.9, zip(:steady_state, :rewards)))
    @rtransform(:basin_1 = .!(:basin_2))
    @rtransform(:fraction = :counts ./ sum(:counts))
    @rtransform(:basin_1_rewards = :rewards[:basin_1], :basin_2_rewards = :rewards[:basin_2], :basin_1_fraction = :fraction[:basin_1], :basin_2_fraction = :fraction[:basin_2])
    @rtransform(:total_count = sum(:counts))
    @select(Not([:steady_state, :rewards, :counts, :basin_1, :basin_2]))
    @rtransform(:basin_1_reward = dot(:basin_1_rewards, :basin_1_fraction)/sum(:basin_1_fraction), :basin_2_reward = dot(:basin_2_rewards, :basin_2_fraction)/sum(:basin_2_fraction))
    @rtransform(:basin_1_fraction = sum(:basin_1_fraction), :basin_2_fraction = sum(:basin_2_fraction))
    @select(Not([:fraction, :basin_1_rewards, :basin_2_rewards]))
    @rsubset((:n != 2) || (:n == 2 && :ω != 0.991))
    sort([:n, :ω, :p])
end

df_equilibria = @chain df begin
    parse_beta_binomial_dist()
    @select(Not([:steady_state, :α, :β, :γ, :stability_threshold, :soltol, :uniqtol, :random_seed]))
    flatten([:rewards, :counts])
    @rsubset(-1e-1 <= :rewards <= 1. + 1e-1)
    @rtransform(:fraction = :counts / :N_ensemble)
    @rsubset(:check_bounds == true)
    @select(Not([:check_bounds, :counts, :N_ensemble]))
    @rsubset(:rewards >= 0 )
    sort(:ω)
end


gdf = @chain df_basins begin
    @select(Not([:variance_scale, :p, :θ, :basin_1_reward, :basin_2_reward]))
    @groupby(:n)
end

newrows = map(collect(gdf)) do group
    row = copy(last(group))
    row = (; row..., ω = 1.0, basin_1_fraction = 0., basin_2_fraction = 1.)
end

df_phase_transition = @chain gdf begin
    @transform()
    append!(newrows)
    sort(:ω)
    @groupby(:n, :ω)
    combine(first)
    @groupby(Not([:ω, :basin_1_fraction, :basin_2_fraction, :total_count]))
    @transform(:derivative = [0; (:basin_1_fraction[2:end] .- :basin_1_fraction[1:end-1]) ./ (:ω[2:end] - :ω[1:end-1])]; ungroup = false)
    @combine(:threshold = (:ω[findmax(-1 .* :derivative)[2]] + :ω[findmax(-1 .* :derivative)[2]-1])/2, :threshold_size = (:basin_1_fraction[findmax(-1 .* :derivative)[2]] + :basin_1_fraction[findmax(-1 .* :derivative)[2] - 1])/2)
    push!((1, 1., 0.))
    sort(:n)
end


## Create the plot

fig = Figure(size = (1300, 400))

# Panel A
ax = Axis(fig[1,1], xlabel = rich("Social learning propensity ", rich("ω", font = :italic)), ylabel = "Reward / basin fractions")

# plotting the shaded region for the basins of attraction
df2 = @rsubset(df_basins, :n == 2, :p == 0.5)
band!(ax, df2.ω, df2.basin_2_fraction, ones(length(df2.ω)); color = (:deepskyblue2, 0.3))
band!(ax, df2.ω, zeros(length(df2.ω)), df2.basin_2_fraction; color = (:red, 0.3))

# plotting the adaptive equilibrium
df3 = @rsubset(df_equilibria, :p ≈ 0.5, :θ ≈ 1.0, :n == 2, :rewards > 0.6)
scatter!(ax, df3.ω, df3.rewards; markersize = 12, color = :blue3) # :royalblue
text!(ax, 0.51, 0.9; text = "Adaptive equilibrium", fontsize = 10)

# plotting the non-adaptive equilibrium
df4 = @rsubset(df_equilibria, :p ≈ 0.5, :θ ≈ 1.0, :n == 2, :rewards <0.6)
scatter!(ax, df4.ω, df4.rewards; markersize = 12, color = :crimson)  
xlims!(ax, 0.5, 1)
text!(ax, 0.7, 0.55; text = "Non-adaptive equilibrium", fontsize = 10)

# Adding arrow and text to panel A
A, B = Point2f(0.8, 0.07), Point2f(0.96, 0.3)
annotation!(ax, [A], [B];
    text = [""],
    path = Ann.Paths.Arc(height = -0.4),
    style = Ann.Styles.LineArrow(),
    color = :black, labelspace = :data, shrink = (5.0, 5.0)
)
text!(ax, 0.83, 0.15; text = "Basin of adaptive\nequilibrium shrinks", align = (:center, :bottom), fontsize = 10)

# Panel B
ax2 = Axis(fig[1,2], xlabel = rich("Social learning propensity ", rich("ω", font = :italic)), ylabel = "Basin size of\nadaptive equilibrium")

all_ns = sort(unique(df_basins.n))
all_ns = 2:10
colors_all = cgrad(:cmr_bubblegum, 10; categorical = true)
colors_from_2 = cgrad(colors_all[2:end]; categorical = true)
for (i, n) in enumerate(all_ns)
    local df5 = @rsubset(df_basins, :n == n, :p == 0.5)
    lines!(ax2, df5.ω, df5.basin_1_fraction; linewidth = 3, label = "n = $n", color = colors_from_2[i])
end
scatter!(ax2, df_phase_transition.threshold[2:end], df_phase_transition.threshold_size[2:end]; markersize = 10, color = :black)

text!(ax2, 0.75, 0.38; text = rich("Critical ", rich("ω", font = :italic)), fontsize = 10, align = (:right, :bottom))

maxn = maximum(all_ns)
Colorbar(fig[1,3]; colormap = colors_from_2, colorrange = (all_ns[1]-0.5, maxn+0.5), ticks = all_ns, label = rich("Problem size ", rich("n", font = :italic)), labelsize = 16, labelpadding = -5)

# Panel C
ax3 = Axis(fig[1,4], xlabel = rich("Problem size ", rich("n", font = :italic)), ylabel = rich("Critical ", rich("ω", font = :italic)), xticks = [1; 2; 4; 6; 8; 10], yticks = 0.75:0.05:1)
ylims!(ax3, 0.75, 1.01)
lines!(ax3, df_phase_transition.n, df_phase_transition.threshold; linewidth = 3, colormap = colors_all, colorrange = (0.5, maxn+0.5), color = 1:10)
scatter!(ax3, df_phase_transition.n[2:end], df_phase_transition.threshold[2:end]; markersize = 10, color = :black)

# Adding Panel labels
for (pos, label) in zip([1; 2; 4], ["A", "B", "C"])
    Label(
        fig[1, pos, TopLeft()], label,
        fontsize = 21,
        font = :bold,
        padding = (pos == 4 ? -74 : -62, 0, 0, 0),
        halign = :right,
        width = 0 
    )
end

colgap!(fig.layout, 2, 10.)

save("Figure_2.pdf", fig)