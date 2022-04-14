cd(@__DIR__)
using OhMyREPL
using Distances
using XLEs
using CUDA
using Printf
using BenchmarkTools
using LinearAlgebra
using Statistics

haversine(lat1::Float32,lon1::Float32,lat2::Float32,lon2::Float32) = 2 * 6372.8 * asin(sqrt(sind((lat2-lat1)/2)^2 + cosd(lat1) * cosd(lat2) * sind((lon2 - lon1)/2)^2))

function pairwise_dist(lat::Vector{Float32}, lon::Vector{Float32})
    #Pre-allocate, since size is known
    n = length(lat)
    result = Array{Float32}(undef, n, n)
    #Brute force fill in each cell, ignore that distance [i,j] = distance [j,i]
    for i in 1:n
        for j in 1:n
            @inbounds result[i, j] = haversine(lat[i], lon[i], lat[j], lon[j])
        end
    end
    return result
end


src, trg = EmbeddingData() |> readData;
srcV, X = map(i -> src[i], 1:2)

rng = 1:Int(4e3)
subx = X[:, rng]

x1 = X[:, 1]
x2 = X[:, 2]

@time pairwise_dist(x1, x2)


function kernel_haversine(latpoint::Float32, lonpoint::Float32, lat::AbstractVector{Float32}, lon::AbstractVector{Float32}, Cinv::AbstractMatrix{Float32}, out::AbstractVector{Float32})
    #Thread index
    #Need to do the n-1 dance, since CUDA expects 0 and Julia does 1-indexing
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x

    # out[i] = 2 * 6372.8 * CUDA.CUBLAS.asin(CUDA.CUBLAS.sqrt(CUDA.CUBLAS.sind((latpoint-lat[i])/2)^2 + CUDA.CUBLAS.cosd(lat[i]) * CUDA.CUBLAS.cosd(latpoint) * CUDA.CUBLAS.sind((lonpoint - lon[i])/2)^2))
    out[i] =
    #Return nothing, since we're writing directly to the out array allocated on GPU
    return nothing
end

function distmat(lat::Vector{Float32}, lon::Vector{Float32}, Cinv::AbstractMatrix{Float32}; dev::CuDevice=CuDevice(0))

    #Create a context
    ctx = CuContext(dev)

    #Change to objects with CUDA context
    n = length(lat)
    xgpu = CuArray(x)
    ygpu = CuArray(y)
    d_out = CuArray{Float32}(undef, n)

    #Calculate number of calculations, threads, blocks
    len = n

    maxPossibleThreads = attribute(device(),CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X) # or maybe MAX_THREADS_PER_BLOCK?
    threadsGPU = min(len, maxPossibleThreads)
    blocksGPU = Int(ceil(len/threadsGPU))

    #Julia side accumulation of results to relieve GPU memory pressure
    accum = Array{Float32}(undef, n, n)

    # run and time the test
    secs = CUDA.@elapsed begin
        for i in 1:n
            CUDA.@cuda threads=threadsGPU blocks=blocksGPU kernel_mahalanobis(x[i], y[i], xgpu, ygou, Cinv, d_out)
            accum[:, i] =  Vector{Float32}(d_out)
        end
    end

    #Clean up context
    CUDA.unsafe_destroy!(ctx)

    #Return timing and bring results back to Julia
    return (secs, accum)

end



@time timing, result = distmat(x1, x2)
