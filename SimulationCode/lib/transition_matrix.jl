function steady_state_distribution(p::AbstractVector{eltype}) where eltype
    n = length(p)
    Taction = [i==j+1 ? p[j] : i==j-1 ? 1-p[j] : zero(eltype) for i in 1:n, j in 1:n]
    Trestart = [j==1 ? (1-p[1])/n : j==n ? p[n]/n : zero(eltype) for i in 1:n, j in 1:n]
    T = Taction + Trestart
    eigvals, eigvecs = eigen(T)
    idx = findfirst(≈(1.0), eigvals)
    isnothing(idx) && throw(ErrorException("Stochastic matrix $T does not have eigenvalue 1."))
    # _, idx = findmin(x -> abs(x - 1.0), eigvals)
    eigvec = eigvecs[:, idx]
    eigvec = eigvec / sum(eigvec)
end

function steady_state_distribution(T::AbstractMatrix)
    eigvals, eigvecs = eigen(T)
    idx = findfirst(≈(1.0), eigvals)
    # _, idx = findmin(x -> abs(x - 1.0), eigvals)
    isnothing(idx) && throw(ErrorException("Stochastic matrix $T does not have eigenvalue 1."))
    eigvec = eigvecs[:, idx]
    eigvec = eigvec / sum(eigvec)
    return real(eigvec)
end

function create_T_matrix(p::AbstractVector{eltype}) where eltype
    n = length(p)
    Taction = [i==j+1 ? p[j] : i==j-1 ? 1-p[j] : zero(eltype) for i in 1:n, j in 1:n]
    Trestart = [j==1 ? (1-p[1])/n : j==n ? p[n]/n : zero(eltype) for i in 1:n, j in 1:n]
    T = Taction + Trestart
end

function create_T_matrix(p::AbstractVector{eltype}, start_probs::AbstractVector) where eltype
    n = length(p)
    Taction = [i==j+1 ? p[j] : i==j-1 ? 1-p[j] : zero(eltype) for i in 1:n, j in 1:n]
    Trestart = [j==1 ? (1-p[1])*start_probs[i] : j==n ? p[n]*start_probs[i] : zero(eltype) for i in 1:n, j in 1:n]
    T = Taction + Trestart
end

function generate_matrix_function(start_probs::AbstractVector)
    if allsame(start_probs)
        create_T_matrix
    else
        p -> create_T_matrix(p, start_probs)
    end
end