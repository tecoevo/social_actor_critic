## Load packages and required simulation and plotting functions
include("common.jl")
include("../SimulationCode/lib/n_state_SAC_equations.jl")

## Run the simulation and create the plot of the trajetory

n = 2
x0 = 0.3
s0 = 0.0
cov = 0.0
tspan = (0, 1_000_000)

ω = 1
α = 0.01
β = 0.01
γ = 0.9
p = 0.5
θ = 1.0

bias = :un
rtol = 0.4
full_demonstrator_choice = false

start_probs, _ = start_beta_binomial(n, p, θ)

y0 = create_y0(n, x0, s0, cov)
pars = create_pars(n, ω, α, β, γ, full_demonstrator_choice, y0, generate_matrix_function(start_probs), bias, start_probs)
global_cache = pars[7]

fname = eval(Symbol("SAC_$(bias)biased_n_state!"))
prob = ODEProblem(fname, y0, tspan, pars)
sol = solve(prob, RadauIIA5(; autodiff = AutoFiniteDiff()); abstol = 1e-9, reltol = 1e-9)

t = sol.t
u = reduce(hcat, sol.u)
r = map(u -> ensemble_average_reward(u, n, global_cache), eachcol(u))

fig = Figure(size = (600, 500))
ax = Axis(fig[1,1], xlabel = "Time")
for i in 1:2
    lines!(ax, t, u[i, :]; linewidth = 4)
end
lines!(ax, t, r; linewidth = 4, linestyle = :dash)
ylims!(ax, -0.05, 1.05)

text!(ax, 10e5, 0.99; text = rich("Population average\npolicy component ", rich("y", font = :italic)), align = (:right, :top), fontsize = 16)
text!(ax, 10e5, 0.01; text = rich("Population average\npolicy component ", rich("x", font = :italic)), align = (:right, :bottom), fontsize = 16)
text!(ax, 10e5, 0.51; text = rich("Population average reward"), align = (:right, :bottom), fontsize = 16)

save("Figure_S1.pdf", fig)
display(fig)