function hessian_matrix!(hess::AbstractMatrix{Float64}, p::AbstractVector{Float64}, ic::IterationCache, dc::DerivativeCache)
    n = length(p)
    
    x, T_p, F_p, firstOrderProb, secondOrderProbS, secondOrderProbEta = ic.x, ic.T_p, ic.F_p, ic.firstOrderProb, ic.secondOrderProbS, ic.secondOrderProbEta 
    g_x, g_xx, g_xp, g_pp = dc.g_x, dc.g_xx, dc.g_xp, dc.g_pp

    # calculating evaluation function gradients
    dc.g_x_f!(g_x, x, p)
    dc.g_xp_f!(g_xp, x, p)
    g_px = transpose(g_xp)
    dc.g_xx_f!(g_xx, x, p)
    dc.g_pp_f!(g_pp, x, p)
    
    # first order adjoint
    firstOrderRHS = g_x .- dot(x, g_x)
    # firstOrderRHS .-= x * dot(x, firstOrderRHS) # why?
    firstOrderProb.b = firstOrderRHS
    λ = solve!(firstOrderProb)
    
    # second order adjoint method
    for i in 1:n
        secondOrderProbS.b = -F_p[:, i]
        s = solve!(secondOrderProbS)
        y = g_xp[:, i] + g_xx * s - T_p[:, :, i]' * λ
        secondOrderProbEta.b = y .- dot(x, y)
        eta = solve!(secondOrderProbEta)
        T_p_s = tensor3_dot2_vector(T_p, s)
        # T_pp = ForwardDiff.jacobian(p -> ForwardDiff.jacobian(create_matrix, p), p)
        # T_pp = reshape(T_pp, n, n, n, n)
        # T_pp_X = tensor_vector_contraction(T_pp, x, 2)
        # T_pp_X_v = tensor_vector_contraction(T_pp_X, v, 3)
        # Hcol = g_pp*v + g_px * s - (T_p_s + T_pp_X_v)' * λ - F_p' * eta
        Hcol = g_pp[:, i] + g_px * s - T_p_s' * λ - F_p[1:n, :]' * eta
        hess[:, i] .= Hcol
    end
    
    return hess
end

function hessian_matrix(p::AbstractVector{Float64}, ic::IterationCache, derivative_cache::DerivativeCache)
    n = length(p)
    hess = zeros(n, n)
    hessian_matrix!(hess, p, ic, derivative_cache)
end

function hessian_matrix(p::AbstractVector{Float64}, create_matrix::Function, G::Function)
    n = length(p)
    problem_cache = generate_G_derivative_funcs(G, n)
    iteration_cache = IterationCache(p, create_matrix)
    hessian_matrix(p, iteration_cache, problem_cache)
end

function hessian_matrix_vector!(hess::AbstractVector{Float64}, p::AbstractVector{Float64}, ic::IterationCache, dc::DerivativeCache)
    n = length(p)

    x, T_p, F_p, firstOrderProb, secondOrderProbS, secondOrderProbEta = ic.x, ic.T_p, ic.F_p, ic.firstOrderProb, ic.secondOrderProbS, ic.secondOrderProbEta 
    g_x, g_xx, g_xp, g_pp = dc.g_x, dc.g_xx, dc.g_xp, dc.g_pp

    # calculating evaluation function gradients
    dc.g_x_f!(g_x, x, p)
    dc.g_xp_f!(g_xp, x, p)
    g_px = transpose(g_xp)
    dc.g_xx_f!(g_xx, x, p)
    dc.g_pp_f!(g_pp, x, p)
    
    # first order adjoint
    firstOrderRHS = g_x .- dot(x, g_x)
    # firstOrderRHS .-= x * dot(x, firstOrderRHS) # why?
    firstOrderProb.b = firstOrderRHS
    λ = solve!(firstOrderProb)
    
    # second order adjoint method
    previndex = 0
    for i in 1:n
        secondOrderProbS.b = -F_p[:, i]
        s = solve!(secondOrderProbS)
        y = g_xp[:, i] + g_xx * s - T_p[:, :, i]' * λ
        secondOrderProbEta.b = y .- dot(x, y)
        eta = solve!(secondOrderProbEta)
        T_p_s = tensor3_dot2_vector(T_p, s)
        # T_pp = ForwardDiff.jacobian(p -> ForwardDiff.jacobian(create_T_matrix, p), p)
        # T_pp = reshape(T_pp, n, n, n, n)
        # T_pp_X = tensor_vector_contraction(T_pp, x, 2)
        # T_pp_X_v = tensor_vector_contraction(T_pp_X, v, 3)
        # Hcol = g_pp*v + g_px * s - (T_p_s + T_pp_X_v)' * λ - F_p' * eta
        Hcol = g_pp[:, i] + g_px * s - T_p_s' * λ - F_p[1:n, :]' * eta
        hess[(previndex+1):(previndex+i)] .= Hcol[1:i]
        previndex += i
    end

    return hess
end

function hessian_matrix!(hess::AbstractMatrix{Float64}, p::AbstractVector{Float64}, ic::IterationCache, dc::DerivativeCache)
    n = length(p)

    x, T_p, F_p, firstOrderProb, secondOrderProbS, secondOrderProbEta = ic.x, ic.T_p, ic.F_p, ic.firstOrderProb, ic.secondOrderProbS, ic.secondOrderProbEta 
    g_x, g_xx, g_xp, g_pp = dc.g_x, dc.g_xx, dc.g_xp, dc.g_pp

    # calculating evaluation function gradients
    dc.g_x_f!(g_x, x, p)
    dc.g_xp_f!(g_xp, x, p)
    g_px = transpose(g_xp)
    dc.g_xx_f!(g_xx, x, p)
    dc.g_pp_f!(g_pp, x, p)
    
    # first order adjoint
    firstOrderRHS = g_x .- dot(x, g_x)
    # firstOrderRHS .-= x * dot(x, firstOrderRHS) # why?
    firstOrderProb.b = firstOrderRHS
    λ = solve!(firstOrderProb)
    
    # second order adjoint method
    for i in 1:n
        secondOrderProbS.b = -F_p[:, i]
        s = solve!(secondOrderProbS)
        y = g_xp[:, i] + g_xx * s - T_p[:, :, i]' * λ
        secondOrderProbEta.b = y .- dot(x, y)
        eta = solve!(secondOrderProbEta)
        T_p_s = tensor3_dot2_vector(T_p, s)
        # T_pp = ForwardDiff.jacobian(p -> ForwardDiff.jacobian(create_T_matrix, p), p)
        # T_pp = reshape(T_pp, n, n, n, n)
        # T_pp_X = tensor_vector_contraction(T_pp, x, 2)
        # T_pp_X_v = tensor_vector_contraction(T_pp_X, v, 3)
        # Hcol = g_pp*v + g_px * s - (T_p_s + T_pp_X_v)' * λ - F_p' * eta
        Hcol = g_pp[:, i] + g_px * s - T_p_s' * λ - F_p[1:n, :]' * eta
        hess[:,i] .= Hcol
    end

    return hess
end