# Code for estimating translational diffusion coefficients from experimental single-particle tracking data
# version 1.0 (14/12/2020)
# Jakob Tómas Bullerjahn (jabuller@biophys.mpg.de)
# Gerhard Hummer

# Please read and cite the associated publication: 
# J. T. Bullerjahn and G. Hummer, "Maximum likelihood estimates of diffusion coefficients from single-molecule tracking experiments", https://arxiv.org/abs/2011.09955

module DiffusionMLE

using Distributions, LinearAlgebra, Optim
# Local modules:
using BasicFunctions

export Kuiper_p_value, Kuiper_statistic!, MLE_errors, MLE_estimator, Q_factor_analysis, sort_trajectories, subpopulation_analysis



#= FUNCTIONS FOR MAXIMUM-LIKELIHOOD ESTIMATION: =#

#= Evaluate the boundary solutions =#
function a2_MLE(d::Int64, M::Int64, N_M::Int64, N::Array{Int64,1}, Δ::Array{Array{Float64,2},1}, c::Array{Array{Float64,1},1}, Y::Array{Array{Float64,1},1})
    ΔΣ_invΔ = 0.0
    logdet = 0.0
    @inbounds for n = 1 : d
        @inbounds for m = 1 : M
            α = 1.0
            β = -0.5
            data_vec = view(Δ[m],:,n)
            Thomas_algorithm!(N[m],α,β,data_vec,c[m],Y[m])
            ΔΣ_invΔ += dot(data_vec,Y[m])
            logdet += log_det(N[m],α)
        end
    end
    a2 = ΔΣ_invΔ/(d*N_M)
    L = 0.5*(d*N_M*(1.0 + log(a2)) + logdet)
    return ([a2,0.0],L/N_M)
end

function σ2_MLE(d::Int64, M::Int64, N_M::Int64, N::Array{Int64,1}, B::AbstractArray{Float64,1}, Δ::Array{Array{Float64,2},1}, c::Array{Array{Float64,1},1}, Y::Array{Array{Float64,1},1})
    ΔΣ_invΔ = 0.0
    logdet = 0.0
    @inbounds for n = 1 : d
        @inbounds for m = 1 : M
            α = 1.0 - 2*B[m]
            β = B[m]
            q = sqrt(1.0 - 4*B[m])/(1.0 - 2*B[m])
            data_vec = view(Δ[m],:,n)
            Thomas_algorithm!(N[m],α,β,data_vec,c[m],Y[m])
            ΔΣ_invΔ += dot(data_vec,Y[m])
            logdet += log_det(N[m],α,q)
        end
    end
    σ2 = ΔΣ_invΔ/(d*N_M)
    L = 0.5*(d*N_M*(1.0 + log(σ2)) + logdet)
    return ([0.0,σ2],L/N_M)
end

#= Search for a general (a2>0,σ2>0)-solution =#
function likelihood(d::Int64, M::Int64, N_M::Int64, N::Array{Int64,1}, ϕ::Float64, B::AbstractArray{Float64,1}, Δ::Array{Array{Float64,2},1}, c::Array{Array{Float64,1},1}, Y::Array{Array{Float64,1},1}, ΔΣ_invΔ::Array{Float64,1})
    ΔΣ_invΔ[1] = 0.0
    logdet = 0.0
    @inbounds for n = 1 : d
        @inbounds for m = 1 : M
            α = 1.0 + ϕ*(1.0 - 2*B[m])
            β = -0.5 + ϕ*B[m]
            q = sqrt(1.0 - 4*β^2 / α^2)
            data_vec = view(Δ[m],:,n)
            Thomas_algorithm!(N[m],α,β,data_vec,c[m],Y[m])
            ΔΣ_invΔ[1] += dot(data_vec,Y[m])
            logdet += log_det(N[m],α,q)
        end
    end
    L = 0.5*d*N_M*log(ΔΣ_invΔ[1]) + 0.5*logdet
    return L
end

function a2_σ2_MLE(d::Int64, M::Int64, N_M::Int64, N::Array{Int64,1}, B::AbstractArray{Float64,1}, Δ::Array{Array{Float64,2},1}, c::Array{Array{Float64,1},1}, Y::Array{Array{Float64,1},1}, interval::Array{Float64,1})
    ΔΣ_invΔ = [0.0]
    res = optimize(ϕ->likelihood(d,M,N_M,N,ϕ,B,Δ,c,Y,ΔΣ_invΔ), interval[1], interval[2], Brent())
    a2 = ΔΣ_invΔ[1]/(d*N_M)
    σ2 = a2*res.minimizer
    L = res.minimum + 0.5*d*N_M*(1.0 - log(d*N_M))
    return ([a2,σ2],L/N_M)
end

#= Evaluate and compare boundary solutions with the general solution
   and report the most favorable values =#
function MLE_estimator(B::AbstractArray{Float64,1}, X::AbstractArray{Array{Float64,2},1}, interval::Array{Float64,1}=[0.0,10000.0])
    M = length(X)
    d = size(X[1],2)
    N = zeros(Int64,M)
    Δ = Array{Array{Float64,2},1}(undef, M)
    @inbounds for m = 1 : M
        N[m] = size(X[m],1) - 1
        Δ[m] = make_increments(d,N[m],X[m])
    end
    N_M = sum(N)
    c = [ zeros(N[m]-1) for m = 1 : M ]
    Y = [ zeros(N[m]) for m = 1 : M ]
    likelihoods = [0.0, 0.0, 0.0]
    solutions = [[0.0,0.0] for i = 1 : 3]
    solutions[1], likelihoods[1] = a2_MLE(d,M,N_M,N,Δ,c,Y)
    solutions[2], likelihoods[2] = σ2_MLE(d,M,N_M,N,B,Δ,c,Y)
    solutions[3], likelihoods[3] = a2_σ2_MLE(d,M,N_M,N,B,Δ,c,Y,interval)
    return solutions[findmin(likelihoods)[2]]
end



#= FUNCTIONS FOR ERROR ANALYSIS: =#
function MLE_errors(B::AbstractArray{Float64,1}, X::AbstractArray{Array{Float64,2},1}, parameters::Array{Float64,1})
    a2 = parameters[1]
    σ2 = parameters[2]
    M = length(X)
    d = size(X[1],2)
    N = [ size(X[m],1) - 1 for m = 1 : M ]
    N_M = sum(N)
    if a2 == 0.0
        return [0.0, sqrt(2/(d*N_M))*σ2]
    elseif σ2 == 0.0
        return [sqrt(2/(d*N_M))*a2, 0.0]
    else
        I_11 = 0.5*d*N_M/a2^2
        ϕ = σ2 / a2
        dlogdet = 0.0
        ddlogdet = 0.0
        for m = 1 : M
            α = 1.0 + ϕ*(1.0 - 2*B[m])
            β = -0.5 + ϕ*B[m]
            q = sqrt(1.0 - 4*β^2 / α^2)
            dlogdet += dlog_det(N[m],α,β,q,ϕ)
            ddlogdet += ddlog_det(N[m],α,β,q,ϕ)
        end
        I_12 = 0.5*d/a2*dlogdet
        I_22 = -0.5*d*ddlogdet
        detI = I_11*I_22 - I_12^2
        δa2 = sqrt(I_22/detI)
        δσ2 = sqrt( (a2^2*I_11 - 2*ϕ*a2*I_12 + ϕ^2*I_22)/detI )
        return [δa2, δσ2]
    end
end



#= FUNCTIONS FOR QUALITY FACTOR ANALYSIS: =#

#= Evaluate quality factor for each trajectory =#
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
        Q[m] = 1.0 - cdf(Chisq(d*N),χ2)
    end
    return Q
end

function sort_trajectories(M::Int64, K::Int64, T::Array{Float64,2}, B::AbstractArray{Float64,1}, X::AbstractArray{Array{Float64,2},1})
    B_sub = [ Array{Float64,1}() for k = 1 : K ]
    X_sub = [ Array{Array{Float64,2},1}() for k = 1 : K ]
    @inbounds for m = 1 : M
        index = findmax(T[:,m])[2]
        push!(B_sub[index], B[m])
        push!(X_sub[index], X[m])
    end
    return B_sub, X_sub
end

function subpopulation_analysis(T::Array{Float64,2}, parameters::Array{Float64,2}, B::AbstractArray{Float64,1}, X::AbstractArray{Array{Float64,2},1})
    M = length(X)
    d = size(X[1])[2]
    K = size(T,1)
    B_sub, X_sub = sort_trajectories(M,K,T,B,X)
    M_sub = length.(X_sub)
    Q_sub = Array{Array{Float64,1},1}(undef,K)
    @inbounds for k = 1 : K
        if length(X_sub[k]) == 0
            Q_sub[k] = []
        else
            Q_sub[k] = Q_factor_analysis(B_sub[k],X_sub[k],parameters[1:2,k])
        end
    end
    return Q_sub
end

#= Compute Kuiper statistic for a set of quality factors =#
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

#= Compute corresponding p-value =#
function Kuiper_p_value(K::Float64, N_trunc::Int64=100)
    p = 0.0
    @inbounds for i = 1 : N_trunc
        p += (4*i^2*K^2 - 1.0)*exp(-2*i^2*K^2)
    end
    return 2*p
end



end # module
