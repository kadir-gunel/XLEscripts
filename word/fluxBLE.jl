cd(@__DIR__)

using LinearAlgebra
using Statistics
using OhMyREPL
using Flux
using Flux.Optimise: update!, Descent, ADAM, RADAM
using XLEs
using CUDA
using Printf
using Distances

@printf "Setting GPU Device %s :" CUDA.device!(1)


src, trg, valfile = EmbeddingData() |> readData;
srcV, X = map(i -> src[i], 1:2)
trgV, Y = map(i -> trg[i], 1:2)

src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
validation = readValidation(valfile, src_w2i, trg_w2i);

X, Y = map(normalizeEmbedding, [X, Y]);
X, Y = map(cu, [X, Y])

d, w = size(X)

#======================== Model ===========================#
# model = Chain(Dense(d, d, gelu), Dense(d, d, gelu)) |> gpu;
model = Chain(Dense(d, d, gelu)) |> gpu;
loss(x, y) = Flux.Losses.mse(model(x), y);

Kloss(x,y) = mean(abs2,y - model(x))
loss2(x, y) = pairwise(CosineDist(), model(x) ,y) |> XLEs.dist2sim
#==========================================================#

function cslsLoss(x, y)
    # sims = pairwise(CosineDist(), model(x) ,y) |> XLEs.dist2sim
    sims = parallelCosine(model(x), y) |> XLEs.dist2sim
    revsim  = permutedims(sims)
    bestsim = maximum(revsim, dims=2)
    mean(bestsim)
end

# swish , gelu
datax = collect(keys(validation))
datay = collect(values(validation))
datay = collect(collect(datay[i])[1] for i in 1:length(datay))

trainx = datax[1:500]
trainy = datay[1:500]

testx = datax[501:end]
testy = datay[501:end]

newValidation = Dict(testx[i] => testy[i] for i in 1:length(testy))

function trainMapping(X, Y, src_idx, trg_idx, model, loss; opt=Descent(.1), epochs::Int64=10, validation=validation)
    @printf "initial loss: %.4f \n" loss(X[:, src_idx], Y[:, trg_idx])
    for epoch in 1:epochs
        Flux.train!(loss, params(model), [(X[:, src_idx], Y[:, trg_idx])], opt)
        accuracy, similarity = validate(model(X) |> normalizeEmbedding, Y, validation);
        @printf "Epoch: %1i loss: %.4f,  knn-accuracy: %.4f, similarity: %.4f \n" epoch loss(X[:, src_idx], Y[:, trg_idx]) accuracy similarity
    end

end

trainMapping(X, Y, trainx, trainy, model, loss, validation=newValidation, opt=ADAM(.0048), epochs=40)





D = Iterators.repeated((X[:, trainx], Y[:, trainy]), 100)
Flux.@epochs 30 Flux.train!(loss, params(model), D, ADAM(.0048))


accuracy, similarity = validate(model(X) |> normalizeEmbedding, Y, newValidation)


l = loss(model(X[:, trainx]), Y[:, trainy])

py"""
import numpy as np
def write(words, matrix, file):
    m = np.asarray(matrix)
    print('%d %d' % m.shape, file=file)
    for i in range(len(words)):
        print(words[i] + ' ' + ' '.join(['%.6g' % x for x in m[i]]), file=file)
"""
