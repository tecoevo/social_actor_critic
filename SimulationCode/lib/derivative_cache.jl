function generate_T_derivative_func(create_matrix, n)
    P = Symbolics.variables(:p, 1:n)
    X = Symbolics.variables(:x, 1:n)

    T_p_mat_expr = Symbolics.jacobian(create_matrix(P), P)
    T_p_expr = reshape(T_p_mat_expr, (n, n, n))
    T_p_f! = build_function(T_p_expr, P; expression = Val(false))[2]

    return T_p_f!
end

struct DerivativeCache
    g_x_f!::Function
    g_xx_f!::Function
    g_xp_f!::Function
    g_pp_f!::Function
    g_x::Vector{Float64}
    g_xx::Matrix{Float64}
    g_xp::Matrix{Float64}
    g_pp::Matrix{Float64}
end

function generate_G_derivative_funcs(G, n)
    P = Symbolics.variables(:p, 1:n)
    X = Symbolics.variables(:x, 1:n)

    g_x_expr = Symbolics.gradient(G(X, P), X)
    g_x_f! = build_function(g_x_expr, X, P; expression = Val(false))[2]
    g_x = zeros(n)

    g_xp_expr = Symbolics.jacobian(g_x_expr, P)
    g_xp_f! = build_function(g_xp_expr, X, P; expression = Val(false))[2]
    g_xp = zeros(n, n)

    g_xx_expr = Symbolics.hessian(G(X, P), X)
    g_xx_f! = build_function(g_xx_expr, X, P; expression = Val(false))[2]
    g_xx = zeros(n, n)

    # g_pp_expr = Symbolics.hessian(G(X, P), P)
    # g_pp_f! = build_function(g_pp_expr, X, P; expression = Val(false))[2]
    cfg = ForwardDiff.HessianConfig(nothing, zeros(n))
    g_pp_f! = (g_pp, X, P) -> ForwardDiff.hessian!(g_pp, p -> G(X, p), P, cfg)
    g_pp = zeros(n, n)
    return DerivativeCache(g_x_f!, g_xx_f!, g_xp_f!, g_pp_f!, g_x, g_xx, g_xp, g_pp)
end

# Construct the derivative cache but only using numerical automatic differentiation
# Useful for the Actor-critic equations due to the LinearSolve
function generate_G_derivative_funcs_num(G, n)
    cfg1 = ForwardDiff.GradientConfig(nothing, zeros(n))
    g_x_f! = (g_x, X, P) -> ForwardDiff.gradient!(g_x, x -> G(x, P), X, cfg1)
    g_x = zeros(n)

    g_xp_f! = (g_xp, X, P) -> ForwardDiff.jacobian!(g_xp, p -> ForwardDiff.gradient(x -> G(x, p), X), P)
    g_xp = zeros(n, n)

    cfg2 = ForwardDiff.HessianConfig(nothing, zeros(n))
    g_xx_f! = (g_xx, X, P) -> ForwardDiff.hessian!(g_xx, x -> G(x, P), X, cfg2)
    g_xx = zeros(n, n)

    cfg3 = ForwardDiff.HessianConfig(nothing, zeros(n))
    g_pp_f! = (g_pp, X, P) -> ForwardDiff.hessian!(g_pp, p -> G(X, p), P, cfg3)
    g_pp = zeros(n, n)
    return DerivativeCache(g_x_f!, g_xx_f!, g_xp_f!, g_pp_f!, g_x, g_xx, g_xp, g_pp)
end