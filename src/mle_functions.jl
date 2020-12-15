#= Evaluate the boundary solutions: =#

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



#= Search for a general (a2>0,σ2>0)-solution: =#

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
   and report the most favorable values: =#

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



#= Evaluate the uncertainty behind a particular solution: =#

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
