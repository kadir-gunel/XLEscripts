# Centered Kernel Alignment 
cd(@__DIR__)
ENV["PYTHON"] = "/home/phd/miniconda3/envs/torch2.0/bin/python"

using PyCall
using LinearAlgebra
using Statistics

using Distances
using XLEs

using BSON: @load

using Test



gram_linear(X::Matrix) = X' * X

function gram_rbf(X::AbstractArray; threshold=1.)
    dot_prod = X' * X 
    sq_norms = diag(dot_prod)
    sq_dists =  -2 * dot_prod .+ sq_norms .+ permutedims(sq_norms)
    sq_median_distance = median(sq_dists)
    return exp.(-sq_dists / (2 * threshold^2 * sq_median_distance))
end


function center_gram(G::Matrix; unbiased=false)
    if !issymmetric(G) 
        @error "Gram Matrix have to be symmetric"
    G = deepcopy(G)
    end
    if unbiased
        n, n = G |> size
        G[diagind(G)] .= 0.
        μs = sum(G, dims=2) / (n-1)
        μs = μs .- (sum(μs) / (2 * (n -1)))
        G = G .- μs .- permutedims(μs) 
        G[diagind(G)] .= 0
    else     
        μs = mean(G, dims=2)
        μs = μs .- mean(μs) / 2
        G = G .- μs .- permutedims(μs) 
    end
    return G    
end




function cka(Gx, Gy; debiased=false)
    Gx = center_gram(Gx, unbiased=debiased)
    Gy = center_gram(Gy, unbiased=debiased)

    hsic = vec(Gx)' * vec(Gy)
    normX, normY = map(norm, [Gx, Gy])

    return hsic / (normX * normY)
end




src_lang = "en"
trg_lang = "it"
vecfile  = "../vecmap/data/embeddings/"
valfile  = "../vecmap/data/dictionaries/en-$(trg_lang).test.txt"

@load "./models/W_$(trg_lang).bson" W src_idx trg_idx


srcV, X = readBinaryEmbeddings(vecfile * src_lang)
trgV, Y = readBinaryEmbeddings(vecfile * trg_lang)


src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
validation = readValidation(valfile, src_w2i, trg_w2i);

trg_indices = collect(values(validation));
src_indices = collect(keys(validation));



X, Y = map(permutedims, [X, Y])
X, Y = map(XLEs.unit, [X, Y])




valx = X[:, src_indices] |> Array
valy = Y[:, first.(trg_indices)] |> Array

subx = X[:, 1:Int(20e3)] |> Array
suby = Y[:, 1:Int(20e3)] |> Array


Sx = gram_linear(subx)
Sy = gram_linear(suby)

cka(Sx, Sy)

Gx = gram_linear(valx)
Gy = gram_linear(valy)

cka(Gx, Gy)

using Distances

cosx = 1 .- pairwise(CosineDist(), valx, valy)

A = getindex.(argmax(cosx, dims=1), 1) |> vec
B = getindex.(argmax(cosx, dims=2), 2) |> vec


Fx = X * X'
Fy = Y * Y'

w, _ = XLEs.mapOrthogonal2(Fx, Fy)

A = gram_linear(Array(w * Fx))
B = gram_linear(Array(Fy))




#######################################################################
using Plots
using LinearAlgebra

pgfplotsx()

Sr = svdvals(Array(W * X) |> XLEs.unit)
S = svdvals(Array(X) |> XLEs.unit)

histogram(log.(Sr), bins=100, color=:green3)
histogram!(log.(S), bins=100, color=:pink)

Gx = gram_rbf(rX[:, src_indices] |> XLEs.unit, threshold=1)
Gxl = gram_linear(rX[:, src_indices] |> XLEs.unit)

Gy = gram_rbf(Y[:, first.(trg_indices)] |> XLEs.unit, threshold=.5)
r = cka(Gx, Gy; debiased=true)



########################################################################

nominator(K, n::Int64) = (permutedims(ones(n)) * K * ones(n))[1]

function HSIC(K::Matrix, L::Matrix)
    n, _ = size(K)
    result = tr(K * L)
    println(result)
    result += (nominator(K, n) * nominator(L, n)) / ((n -1) * (n -2))
    println(result)
    result = result - ((2 / (n -2)) * nominator(permutedims(K * L), n))
    println(result)
    return 1 / (n * (n -3)) * result
end



HSIC()



function gram_rbf(X::AbstractArray; threshold=1.)
    dot_prod = X' * X 
    sq_norms = diag(dot_prod)
    sq_dists =  -2 * dot_prod .+ sq_norms .+ permutedims(sq_norms)
    sq_median_distance = median(sq_dists)
    return exp.(-sq_dists / (2 * threshold^2 * sq_median_distance))
end


subx = X[:, 1:1000]


gram_rbf(subx)