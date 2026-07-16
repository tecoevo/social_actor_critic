mutable struct IterationCache{T1, T2, T3}
    x::Vector{Float64}
    T_p::Array{Float64, 3}
    F_p::Matrix{Float64}
    firstOrderProb::LinearSolve.LinearCache{T1}
    secondOrderProbS::LinearSolve.LinearCache{T2}
    secondOrderProbEta::LinearSolve.LinearCache{T3}
    T_p_f!::Function
    create_matrix::Function
end

function IterationCache(p, create_matrix)
    n = length(p)
    T = create_matrix(p)
    x = steady_state_distribution(T)

    T_p_f! = generate_T_derivative_func(create_matrix, n)

    # calculating some required partial derivatives
    A = (T - I)
    T_p = zeros(n, n, n)
    T_p_f!(T_p, p)

    T_p_X = tensor3_dot2_vector(T_p, x)
    F_p = vcat(T_p_X, zeros(1, n))::Matrix{Float64}
    
    F_x = vcat(A, ones(1, n))

    # setting up solver for first order adjoint
    firstOrderProb = LinearProblem(permutedims(A), ones(n)) 
    firstOrderProb = init(firstOrderProb, SVDFactorization())
    
    # setting up solvers for second order adjoint method
    secondOrderProbS = LinearProblem(F_x, ones(n+1))
    secondOrderProbS = init(secondOrderProbS, SVDFactorization())
    secondOrderProbEta = LinearProblem(permutedims(A), ones(n))
    secondOrderProbEta = init(secondOrderProbEta, SVDFactorization())
    return IterationCache(x, T_p, F_p, firstOrderProb, secondOrderProbS, secondOrderProbEta, T_p_f!, create_matrix)
end

function updateIterationCache!(cache::IterationCache, p::AbstractVector{Float64})
    n = length(p)
    T = cache.create_matrix(p)
    cache.x .= steady_state_distribution(T)

    A = (T - I)
    cache.T_p_f!(cache.T_p, p)
    T_p_X = tensor3_dot2_vector(cache.T_p, cache.x)

    cache.F_p[1:n, :] .= T_p_X
    
    cache.firstOrderProb.A .= transpose(A)
    cache.firstOrderProb.isfresh = true

    cache.secondOrderProbS.A[1:n, :] .= A
    cache.secondOrderProbS.A[n+1, :] .= 1.
    cache.secondOrderProbS.isfresh = true

    cache.secondOrderProbEta.A .= transpose(A)
    cache.secondOrderProbEta.isfresh = true
    return nothing
end