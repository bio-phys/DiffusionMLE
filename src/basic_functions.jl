#= Make increments Δ from time series X: =#

function make_increments(d::Int64, N::Int64, X::Array{Float64,2})
    Δ = zeros(N,d)
    @inbounds for n = 1 : d
        @inbounds for i = 1 : N
            Δ[i,n] += X[i+1,n] - X[i,n]
        end
    end
    return Δ
end



#= Compute Y = Σ^-1*Δ for a symmetric tridiagonal Toeplitz matrix Σ with diagonal elements α and off-diagonal elements β: =#

function Thomas_algorithm!(N::Int64, α::Float64, β::Float64, Δ::AbstractArray{Float64,1}, c::Array{Float64,1}, Y::Array{Float64,1})
    @inbounds c[1] = β/α
    @inbounds Y[1] = Δ[1]/α
    @inbounds for i = 2 : N-1
        c[i] = β/(α - β*c[i-1])
        Y[i] = (Δ[i] - β*Y[i-1])/(α - β*c[i-1])
    end
    @inbounds Y[N] = (Δ[N] - β*Y[N-1])/(α - β*c[N-1])
    @inbounds for i in reverse(1:N-1)
        Y[i] -= c[i]*Y[i+1]
    end
end



#= Evaluate ln(|Σ|) and its derivatives for q>0 and q=0, respectively: =#

log_det(N::Int64, α::Float64, q::Float64) = N*log(α) + (N+1)*log(0.5*(1+q)) + log(abs(1 - ((1-q)/(1+q))^(N+1))) - log(q)
log_det(N::Int64, α::Float64) = log(N+1) + N*log(0.5*α)

F(N::Int64, α::Float64, β::Float64, q::Float64) = 2*β/(α^3*q^2*(1.0 + q))*(N*q - 1.0 + (N + 1)*exp(N*log(0.5*α*(1-q)) - log_det(N,α,q)) )
F(N::Int64, α::Float64, β::Float64) = 2*β*N*(N - 1)/(3*α^3)
function dF(N::Int64, α::Float64, β::Float64, q::Float64, ϕ::Float64)
    C1 = (0.25/β + 0.5*3/α + β*ϕ/α^3*(3/q^2 + 0.5*α^2*N/β^2))
    C2 = 3*ϕ / (4*α*β*q^2)
    f = F(N,α,β,q)
    f0 = F(N,α,β)
    return - 2/ϕ*(f - C1*f + C2*f0 - 0.5*ϕ*f^2)
end

dlog_det(N::Int64, α::Float64, β::Float64, q::Float64, ϕ::Float64) = N*(α - 1.0)/(α*ϕ) - F(N,α,β,q)
ddlog_det(N::Int64, α::Float64, β::Float64, q::Float64, ϕ::Float64) = - N*(α - 1.0)^2/(α*ϕ)^2 - dF(N,α,β,q,ϕ)
