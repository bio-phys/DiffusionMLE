#= Evaluate quality factor for each trajectory: =#

function Q_factor_analysis(B::AbstractArray{Float64,1}, X::AbstractArray{Array{Float64,2},1}, parameters::Array{Float64,1})
    M = length(X)
    d = size(X[1],2)
    Q = zeros(M)
    @inbounds for m = 1 : M
        N = size(X[m],1) - 1
        Δ = make_increments(d,N,X[m])
        c = zeros(N-1)
        Y = zeros(N)
        α = parameters[1] + parameters[2]*(1.0 - 2*B[m])
        β = -0.5*parameters[1] + parameters[2]*B[m]
        χ2 = 0.0
        @inbounds for n = 1 : d
            data_vec = view(Δ,:,n)
            Thomas_algorithm!(N,α,β,data_vec,c,Y)
            χ2 += dot(data_vec,Y)
        end
        Q_tmp = 1.0 - cdf(Chisq(d*N),χ2)
        if Q_tmp == 1.0
            Q_tmp = 0.9999999999999999
        end
        Q[m] = Q_tmp
    end
    return Q
end



#= Compute Kuiper statistic for a set of quality factors: =#

function Kuiper_statistic!(Q_values::Array{Float64,1})
    M = length(Q_values)
    Q = sort(Q_values)
    D_upper = 0.0
    D_lower = 0.0
    @inbounds for m = 1 : M
        func = Q[m]
        upper_candidate = m/M - func
        lower_candidate = func - (m-1)/M
        if upper_candidate > D_upper
            D_upper = upper_candidate
        end
        if lower_candidate > D_lower
            D_lower = lower_candidate
        end
    end
    return sqrt(M)*(D_upper + D_lower)
end



#= Compute corresponding p-value: =#

function Kuiper_p_value(K::Float64, N_trunc::Int64=100)
    p = 0.0
    @inbounds for i = 1 : N_trunc
        p += (4*i^2*K^2 - 1.0)*exp(-2*i^2*K^2)
    end
    return 2*p
end
