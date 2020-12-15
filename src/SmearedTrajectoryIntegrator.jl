module SmearedTrajectoryIntegrator

export make_1D_data, make_2D_data



function generate_1D_timeseries(N::Int64,N_sub::Int64,a2::Float64,σ2::Float64)
    a = sqrt(a2)
    σ = sqrt(σ2)
    X = zeros(N+1)
    Y = zeros(N_sub)
    y = 0.0
    @inbounds for i = 1 : N+1
        @inbounds for j = 1 : N_sub
            y += σ/sqrt(N_sub)*randn()
            Y[j] = y
        end
        X[i] = sum(Y)/N_sub + a/sqrt(2)*randn()
    end
    return X
end

function make_1D_data(N::Array{Int64,1},N_sub::Int64,a2::Float64,σ2::Float64)
    M = length(N)
    data = Array{Array{Float64,2},1}(undef, M)
    for n = 1 : M
        data[n] = generate_1D_timeseries(N[n],N_sub,a2,σ2)
    end
    return data
end

function generate_2D_timeseries(N::Int64,N_sub::Int64,a2::Float64,σ2::Float64)
    a = sqrt(a2)
    σ = sqrt(σ2)
    X = zeros(N+1,2)
    Y = zeros(N_sub)
    Z = zeros(N_sub)
    y = 0.0
    z = 0.0
    @inbounds for i = 1 : N+1
        @inbounds for j = 1 : N_sub
            y += σ/sqrt(N_sub)*randn()
            z += σ/sqrt(N_sub)*randn()
            Y[j] = y
            Z[j] = z
        end
        X[i,1] = sum(Y)/N_sub + a/sqrt(2)*randn()
        X[i,2] = sum(Z)/N_sub + a/sqrt(2)*randn()
    end
    return X
end

function make_2D_data(N::Array{Int64,1},N_sub::Int64,a2::Float64,σ2::Float64)
    M = length(N)
    data = Array{Array{Float64,2},1}(undef, M)
    for n = 1 : M
        data[n] = generate_2D_timeseries(N[n],N_sub,a2,σ2)
    end
    return data
end



end # module
