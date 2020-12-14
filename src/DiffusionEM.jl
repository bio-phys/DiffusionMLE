# Code for estimating translational diffusion coefficients from experimental single-particle tracking data
# version 1.0 (14/12/2020)
# Jakob Tómas Bullerjahn (jabuller@biophys.mpg.de)
# Gerhard Hummer

# Please read and cite the associated publication: 
# J. T. Bullerjahn and G. Hummer, "Maximum likelihood estimates of diffusion coefficients from single-molecule tracking experiments", https://arxiv.org/abs/2011.09955

module DiffusionEM

using Base.Threads, LinearAlgebra, Optim, ProgressMeter
# Local modules:
using BasicFunctions

export global_EM_estimator, local_EM_estimator!



#= FUNCTIONS FOR EXPECTATION-MAXIMIZATION ALGORITHM: =#

#= Update T using the parameter values stored in the array 'parameters' =#
function expectation_step!(d::Int64, M::Int64, K::Int64, N::Array{Int64,1}, parameters::Array{Float64,2}, B::AbstractArray{Float64,1}, Δ::Array{Array{Float64,2},1}, c::Array{Array{Float64,1},2}, Y::Array{Array{Float64,1},2}, T::Array{Float64,2})
    @threads for m = 1 : M
        normalization = 0.0
        @inbounds for k = 1 : K
            a2 = parameters[1,k]
            σ2 = parameters[2,k]
            value = parameters[3,k]
            if σ2 == 0.0
                α = a2
                β = -0.5*a2
            else
                α = a2 + σ2*(1.0 - 2*B[m])
                β = -0.5*a2 + σ2*B[m]
                q = sqrt(1.0 - 4*β^2 / α^2)
            end
            @inbounds for n = 1 : d
                data_vec = view(Δ[m],:,n)
                Thomas_algorithm!(N[m],α,β,data_vec,c[k,m],Y[k,m])
                ΔΣ_invΔ = dot(data_vec,Y[k,m])
                if σ2 == 0.0
                    logdet = log_det(N[m],a2)
                else
                    logdet = log_det(N[m],α,q)
                end
                value *= exp( - 0.5*ΔΣ_invΔ - 0.5*logdet - 0.5*N[m]*log(2*π) )
            end
            T[k,m] = value
            normalization += value
        end
        if normalization == 0.0 # Required for numerical stability
            @inbounds for k = 1 : K
                T[k,m] = 1/K
            end
        elseif normalization == Inf # Required for numerical stability
            @inbounds for k = 1 : K
                if isinf(T[k,m])
                    T[k,m] = 1.0
                else
                    T[k,m] = 0.0
                end
            end
        else
            @inbounds for k = 1 : K
                T[k,m] = T[k,m] / normalization
            end
        end
    end
end

#= Evaluate the boundary solutions =#
function a2_MLE(d::Int64, M::Int64, C_k::Float64, N::Array{Int64,1}, Δ::Array{Array{Float64,2},1}, c::AbstractArray{Array{Float64,1},1}, Y::AbstractArray{Array{Float64,1},1}, T::AbstractArray{Float64,1})
    ΔΣ_invΔ = 0.0
    logdet = 0.0
    @inbounds for m = 1 : M
        α = 1.0
        β = -0.5
        @inbounds for n = 1 : d
            data_vec = view(Δ[m],:,n)
            Thomas_algorithm!(N[m],α,β,data_vec,c[m],Y[m])
            ΔΣ_invΔ += T[m]*dot(data_vec,Y[m])
            logdet += T[m]*log_det(N[m],α)
        end
    end
    a2 = ΔΣ_invΔ/C_k
    L = 0.5*(C_k*(1.0 + log(a2)) + logdet)
    return ([a2,0.0],L)
end

function σ2_MLE(d::Int64, M::Int64, C_k::Float64, N::Array{Int64,1}, B::AbstractArray{Float64,1}, Δ::Array{Array{Float64,2},1}, c::AbstractArray{Array{Float64,1},1}, Y::AbstractArray{Array{Float64,1},1}, T::AbstractArray{Float64,1})
    ΔΣ_invΔ = 0.0
    logdet = 0.0
    @inbounds for m = 1 : M
        α = 1.0 - 2*B[m]
        β = B[m]
        q = sqrt(1.0 - 4*B[m])/(1.0 - 2*B[m])
        @inbounds for n = 1 : d
            data_vec = view(Δ[m],:,n)
            Thomas_algorithm!(N[m],α,β,data_vec,c[m],Y[m])
            ΔΣ_invΔ += T[m]*dot(data_vec,Y[m])
            logdet += T[m]*log_det(N[m],α,q)
        end
    end
    σ2 = ΔΣ_invΔ/C_k
    L = 0.5*(C_k*(1.0 + log(σ2)) + logdet)
    return ([0.0,σ2],L)
end

#= Search for a general (a2>0,σ2>0)-solution =#
function likelihood(d::Int64, M::Int64, C_k::Float64, N::Array{Int64,1}, ϕ::Float64, B::AbstractArray{Float64,1}, Δ::Array{Array{Float64,2},1}, c::AbstractArray{Array{Float64,1},1}, Y::AbstractArray{Array{Float64,1},1}, T::AbstractArray{Float64,1}, ΔΣ_invΔ::Array{Float64,1})
    ΔΣ_invΔ[1] = 0.0
    logdet = 0.0
    @inbounds for m = 1 : M
        α = 1.0 + ϕ*(1.0 - 2*B[m])
        β = -0.5 + ϕ*B[m]
        q = sqrt(1.0 - 4*β^2 / α^2)
        @inbounds for n = 1 : d
            data_vec = view(Δ[m],:,n)
            Thomas_algorithm!(N[m],α,β,data_vec,c[m],Y[m])
            ΔΣ_invΔ[1] += T[m]*dot(data_vec,Y[m])
            logdet += T[m]*log_det(N[m],α,q)
        end
    end
    L = 0.5*C_k*log(ΔΣ_invΔ[1]) + 0.5*logdet
    return L
end

function a2_σ2_MLE(d::Int64, M::Int64, C_k::Float64, N::Array{Int64,1}, B::AbstractArray{Float64,1}, Δ::Array{Array{Float64,2},1}, c::AbstractArray{Array{Float64,1},1}, Y::AbstractArray{Array{Float64,1},1}, T::AbstractArray{Float64,1}, interval::Array{Float64,1})
    ΔΣ_invΔ = [0.0]
    res = optimize(ϕ->likelihood(d,M,C_k,N,ϕ,B,Δ,c,Y,T,ΔΣ_invΔ), interval[1], interval[2], Brent())
    a2 = ΔΣ_invΔ[1]/C_k
    σ2 = a2*res.minimizer
    L = res.minimum + 0.5*C_k*(1.0 - log(C_k))
    return ([a2,σ2],L)
end

#= Update parameters using the array T =#
function maximization_step!(d::Int64, M::Int64, K::Int64, N_M::Int64, N::Array{Int64,1}, parameters::Array{Float64,2}, B::AbstractArray{Float64,1}, Δ::Array{Array{Float64,2},1}, c::Array{Array{Float64,1},2}, Y::Array{Array{Float64,1},2}, T::Array{Float64,2},interval::Array{Float64,1})
    L = zeros(K)
    TlnPoverT = 0.0
    @threads for k = 1 : K
        likelihoods = [0.0, 0.0, 0.0]
        solutions = [[0.0, 0.0] for i = 1 : 3]
        T_temp = view(T,k,:)
        c_temp = view(c,k,:)
        Y_temp = view(Y,k,:)
        C_k = 0.0
        @inbounds for m = 1 : M
            C_k += d*T_temp[m]*N[m]
        end
        solutions[1], likelihoods[1] = a2_MLE(d,M,C_k,N,Δ,c_temp,Y_temp,T_temp)
        solutions[2], likelihoods[2] = σ2_MLE(d,M,C_k,N,B,Δ,c_temp,Y_temp,T_temp)
        solutions[3], likelihoods[3] = a2_σ2_MLE(d,M,C_k,N,B,Δ,c_temp,Y_temp,T_temp,interval)
        i_opt = findmin(likelihoods)[2]
        view(parameters,1:2,k) .= solutions[i_opt]
        parameters[3,k] = sum(T_temp)/M
        L[k] = likelihoods[i_opt]
        @inbounds for m = 1 : M
            TlnPoverT += T_temp[m]*log(parameters[3,k]) - log(T_temp[m]^T_temp[m])
        end
    end
    return (sum(L) - TlnPoverT)/N_M
end

#= Function executing the expectation-maximization algorithm =#
function local_EM_estimator!(d::Int64, M::Int64, K::Int64, N_local::Int64, tolerance::Float64, parameters::Array{Float64,2}, B::AbstractArray{Float64,1}, X::AbstractArray{Array{Float64,2},1}, interval::Array{Float64,1}=[0.0,10000.0])
#    p = Progress(N_local, 1) # set progress meter
    M = length(X)
    d = size(X[1],2)
    T = zeros(K,M)
    N = zeros(Int64,M)
    Δ = Array{Array{Float64,2},1}(undef, M)
    @inbounds for m = 1 : M
        N[m] = size(X[m],1) - 1
        Δ[m] = make_increments(d,N[m],X[m])
    end
    N_M = sum(N)
    c = [ zeros(N[m]-1) for k = 1 : K, m = 1 : M ]
    Y = [ zeros(N[m]) for k = 1 : K, m = 1 : M ]
    L_old = 1.0e8
    L_new = 1.0e8
    @inbounds for i = 1 : N_local
        expectation_step!(d,M,K,N,parameters,B,Δ,c,Y,T) # This updates T
        L_new = maximization_step!(d,M,K,N_M,N,parameters,B,Δ,c,Y,T,interval) # This updates parameters
        if isnan(L_new)
            @warn "NaN detected"
            return parameters, L_new, T
        end
        if abs(L_old - L_new) < tolerance
            return parameters, L_new, T
        else
            L_old = L_new
        end
#        next!(p) # update progress meter
    end
    @warn "Local convergence failure"
    return parameters, L_new, T
end

function global_EM_estimator(K::Int64, N_local::Int64, N_global::Int64, tolerance::Float64, a2_range::Array{Float64,1}, σ2_range::Array{Float64,1}, B::AbstractArray{Float64,1}, X::AbstractArray{Array{Float64,2},1}, interval::Array{Float64,1}=[0.0,10000.0])
    p = Progress(N_global, 1) # set progress meter
    M = length(X)
    d = size(X[1],2)
    parameters = zeros(3,K)
    L_old = 1.0e8
    L_new = 1.0e8
    T = zeros(K,M)
    @inbounds for n = 1 : N_global
        a2_values = [ rand()*exp(i) for i in range(log(a2_range[1]), log(a2_range[2]); length=K) ]
        σ2_values = [ rand()*exp(i) for i in range(log(σ2_range[1]), log(σ2_range[2]); length=K) ]
        P_values = [ 1/K for i = 1 : K ]
        parameter_matrix = permutedims(hcat([a2_values,σ2_values,P_values]...))
        estimates, L_new, T_new = local_EM_estimator!(d,M,K,N_local,tolerance,parameter_matrix,B,X,interval)
        if L_new < L_old
            L_old = L_new
            parameters .= parameter_matrix
            T .= T_new
        end
        next!(p) # update progress meter
    end
    return parameters, L_new, T
end



end # module

