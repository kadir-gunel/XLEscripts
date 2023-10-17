cd(@__DIR__)
using OhMyREPL
using NPZ
using CUDA
using XLEs
using Base.Iterators
using Flux.Losses
using MKL
using Printf
using Statistics
using LinearAlgebra

FT = npzread("./data/FT.npy")
BERT = npzread("./data/Bert.npy")
X, Y = map(permutedims, [FT, BERT])
# X, Y = map(cu, [X, Y])
X, Y = map(normalizeEmbedding, [X, Y])

function mapOrthogonalSentences(X::T, Y::T) where {T}
    F = CUDA.CUBLAS.svd(X * Y')
    W = permutedims(F.U * F.Vt) # * cuinv((X * X') + λ .* CuMatrix{Float32}(I, 300, 300)))
    return W, F.S
end

function align(Ŷ, Y, n; k::Int64=1)
    simŶ = XLEs.correlation(Ŷ; ssize=n)
    simY = XLEs.correlation(Y; ssize=n)
    sim = simŶ' * simY |> cu
    sim = XLEs.csls(sim; k=k)
    src_idx = permutedims(getindex.(argmax(sim, dims=1), 1)) |> vec |> Array
    trg_idx = getindex.(argmax(sim, dims=2), 2) |> vec |> Array
    return src_idx, trg_idx
end


# we have 50k sentences
# lets take the first 40k sentences for rotation
# the rest will be used for testing(10k)

n = Int(30e3)
trn = 1:Int(n)
tst = 1+Int(n):size(X, 2)
trainX, trainY = X[:, trn], Y[:, trn]
testX, testY = X[:, tst], Y[:, tst]

# scores before rotation
@printf "Scores before rotation : \n"
src_idx, trg_idx = align(trainX, trainY, n; k=1)
@printf "Accuracy on train set : %.4f \n" mean(src_idx .== trg_idx)
mean(src_idx .== collect(1:n))
mean(trg_idx .== collect(1:n))

src_idx, trg_idx = align(testX, testY, length(tst); k=10)
@printf "Accuracy on test set : %.4f \n" mean(src_idx .== trg_idx)

mean(src_idx .== collect(1:length(src_idx)))
mean(trg_idx .== collect(1:length(src_idx)))

W, _ = mapOrthogonalSentences(trainX, trainY)

@printf "Scores after rotation from X to Y \n"
src_idx, trg_idx = align((W * trainX) |> normalizeEmbedding, trainY |> normalizeEmbedding, n; k=10)
@printf "Accuracy on train set : %.4f" mean(src_idx .== trg_idx)
@printf " : %.4f" mean(src_idx .== collect(1:length(src_idx)))

src_idx, trg_idx = align((W * testX) |> normalizeEmbedding, testY |> normalizeEmbedding, length(tst); k=10)
@printf "Accuracy on test set : %.4f" mean(src_idx .== trg_idx)

@printf " : %.4f" mean(src_idx .== collect(1:length(src_idx)))


# second rotation but in reverse
Wr, _ = mapOrthogonalSentences(trainY, trainX)
@printf "Scores after rotation from Y to X - transpose \n"
src_idx, trg_idx = align((Wr' * trainX) |> normalizeEmbedding, trainY |> normalizeEmbedding, n; k=1)
@printf "Accuracy on train set : %.4f" mean(src_idx .== trg_idx)
src_idx, trg_idx = align((Wr' * testX) |> normalizeEmbedding, testY |> normalizeEmbedding, length(tst); k=1)
@printf "Accuracy on test set : %.4f" mean(src_idx .== trg_idx)
