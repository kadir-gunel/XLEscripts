cd(@__DIR__)
using OhMyREPL
using Base.Threads
using LinearAlgebra
using Statistics
using Printf
using XLEs
using Distances
using PGFPlotsX
using CUDA
using TransferEntropy
using BSON: @save, @load
using Test

@printf "Setting GPU Device %s :" CUDA.device!(1)

dcov(sub::Matrix) = sub' * sub;

function getDistanceMatrix(M, rev::Bool=false)
    d, w = size(M);
    dist = collect(sortperm(M[:, i], rev=rev) for i in 1:w)
    reduce(hcat, dist)
end

function sqrt_eigen(subE)
      F = svd(subE)
      F.U * diagm(sqrt.(F.S)) * F.Vt
end

topk(voc::Array{String}, dist::Matrix; idx::Int64, k::Int64=10) = voc[dist[:, idx][end-k:end]]


function doubleCenter!(distx::Matrix)
    d, n = size(distx)
    if !isequal(d,n)
        @printf "Matrix is not a square matrix. Exiting ..."
        return 
    end
    μ = fill(mean(distx), n)
    distx .= distx .- mean(distx, dims=1) .- mean(distx, dims=2) .+ μ
end


src, trg, valfile = EmbeddingData() |> readData;
srcV, X = map(i-> src[i], 1:2);
X = X |> normalizeEmbedding;


rng = 1:Int(4e3);
subx = X[:, rng]

# vecmap
vecmapx = subx |> XLEs.correlation |> getDistanceMatrix 

# w/o eigen replacement
dcovx = subx |> dcov |> getDistanceMatrix

# w/ double centering
x = subx' * subx
doubleCenter!(x)
doubleCosx = x * x |> getDistanceMatrix

# from Distance.jl
dcosx = pairwise(CosineDist(), subx) |> XLEs.dist2sim |> getDistanceMatrix

# from Distance.jl with Mahalanobis
dist = Mahalanobis(subx * subx')
dmahx = pairwise(dist, subx) |> XLEs.dist2sim |> getDistanceMatrix

# Kendals' τ 
using StatsBase
Dτ = corkendall(subx) |> getDistanceMatrix;
Dρ = corspearman(subx)|> getDistanceMatrix;

id = 3456;
@printf "%s: \n" srcV[id];

for dist in [vecmapx, dcovx, doubleCosx, dcosx, dmahx, Dτ, Dρ]
    @printf "%s: \n" topk(srcV, dist, idx = id);
end
