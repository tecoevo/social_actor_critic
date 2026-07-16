using LinearAlgebra
import ProgressMeter
using Distributed: RemoteChannel
using LoopVectorization: @turbo
using Distributions: BetaBinomial, probs

# projection function to correct numerical and approximation error in the derivatives

# ensure that the vector lies in the simplex
function project_simplex!(x)
    u = sort(x; rev = true)
    cssv = 0.
    rho = 0
    for j in axes(u, 1)
        cssv += u[j]
        if u[j] - (cssv - 1.) / j > 0
            rho = j
        end
    end
    θ = (sum(@view u[1:rho]) - 1.)/rho
    for i in axes(x, 1)
        x[i] = max(x[i] - θ, 0.)
    end
end

# ensure that the matrix is a true covariance matrix which is symmetric and positive-semi definite
function project_covariance!(S)
    # symmetrize
    S ./= 2
    S .+= transpose(S)

    # make positive semi definite
    eigvals, eigvecs = eigen(S)
    for i in axes(eigvals, 1)
        eigvals[i] = (eigvals[i] > 0) ? eigvals[i] : 0.
    end
    S .= eigvecs * diagm(eigvals) * eigvecs'

    # symmetrize
    S ./= 2
    S .+= transpose(S)
    return S
end

# Matrix and vector multiplication utilities

function tensor_vector_contraction(tensor, vector, index)
    mapreduce(+, zip(eachslice(tensor; dims = index), vector)) do (slice, v)
        slice * v
    end
end

function tensor3_dot2_vector(tensor, vector)
    n1, n2, n3 = size(tensor)
    T = reshape(permutedims(tensor, (1, 3, 2)), (n1 * n3, n2))
    reshape(T*vector, (n1, n3))
end

function LinearAlgebra.dot(x, y, z)
    s = zero(eltype(x))
    @turbo for i in eachindex(x, y, z)
        s += x[i] * y[i] * z[i]
    end
    return s
end

function upper_triangular_to_vector_indices(n)
    m = UpperTriangular([CartesianIndex(i,j) for i in 1:n, j in 1:n])[:]
    keep_indices = findall(!iszero, m)
end

rightopenrange(start, stop, num) = range(start, stop, num+1)[1:end-1]
leftopenrange(start, stop, num) = range(start, stop, num+1)[2:end]
openrange(start, stop, num) = range(start, stop, num+2)[2:end-1]

function allsame(v::AbstractVector)
    isempty(v) && return true
    @inbounds first_val = first(v)
    @inbounds for i in axes(v, 1)[2:end]
        v[i] == first_val || return false
    end
    return true
end

function unduplicate(sols::Array{<:Vector}, tol=1e-6)
    indicesToUse = Int64[]
    counts = Int64[]
    indicesToReject = Int64[]
    for i in 1:length(sols)
        if !(i in indicesToReject)
            push!(indicesToUse, i)
            push!(counts, 1)
            for j in i+1:length(sols)
                if !(j in indicesToReject) && sum((sols[i] .- sols[j]).^2) < tol
                    push!(indicesToReject, j)
                    counts[end] += 1
                end
            end
        end
    end
    sols[indicesToUse], counts
end

function unduplicate(sols::Matrix{<:Real}, tol=1e-6)
    indicesToUse = Int64[]
    counts = Int64[]
    indicesToReject = Int64[]
    for i in 1:size(sols,2)
        if !(i in indicesToReject)
            push!(indicesToUse, i)
            push!(counts, 1)
            for j in i+1:size(sols,2)
                if !(j in indicesToReject) && @views sum((sols[:,i] .- sols[:,j]).^2) < tol
                    push!(indicesToReject, j)
                    counts[end] += 1
                end
            end
        end
    end
    sols[:,indicesToUse], counts
end

ProgressMeter.next!(::Nothing) = nothing
ProgressMeter.next!(channel::RemoteChannel) = put!(channel, true)

function start_beta_binomial(N, mean, shape)
    dist = BetaBinomial(N-1, 2*mean*shape, 2*(1-mean)*shape)
    return probs(dist), String(Symbol(dist))
end