using LinearAlgebra
using Statistics
using OhMyREPL
using Flux
using Flux.Optimise: update!, Descent, ADAM, RADAM
using XLEs
using CUDA
using Printf

function trainMapping(X, Y, src_idx, trg_idx, model, loss; opt=Descent(.1), epochs::Int64=10, validation=validation)

    src_idx_forward  = cu(collect(1:Int(20e3)));
    trg_idx_backward = collect(1:Int(20e3));

    @printf "initial loss: %.4f \n" loss(X[:, src_idx], Y[:, trg_idx])

    for epoch in 1:epochs
        Flux.train!(loss, params(model), [(X[:, src_idx], Y[:, trg_idx])], opt)

        trg_idx_forward,  best_sim_forward  = XLEs.update(Y[:, rng], model(X[:, rng]), .1)
        src_idx_backward, best_sim_backward = XLEs.update(model(X[:, rng]), Y[:, rng], .1)

        src_idx = vcat(src_idx_forward, src_idx_backward);
        trg_idx = vcat(trg_idx_forward, trg_idx_backward);

        accuracy, similarity = validate(model(X) |> normalizeEmbedding, Y, validation);

        @printf "Epoch: %1i loss: %.4f , objective: %7f , knn-accuracy: %.4f , similarity: %.4f \n" epoch loss(X[:, src_idx], Y[:, trg_idx]) ((mean(best_sim_forward) + mean(best_sim_backward))/2) accuracy similarity
    end
    return src_idx, trg_idx, model
end

# src, trg, valfile = EmbeddingData(datapath="/Users/kadirgunel/github/FluxVecMap/data/exp_raw/") |> readData;

src, trg, valfile = EmbeddingData() |> readData;
srcV, X = map(i -> src[i], 1:2)
trgV, Y = map(i -> trg[i], 1:2)

src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
validation = readValidation(valfile, src_w2i, src_w2i);

X, Y = map(normalizeEmbedding, [X, Y]);
X, Y = map(cu, [X, Y])
rng = 1:Int(4e3)
subx = X[:, rng]
suby = Y[:, rng]

d, w = size(subx)

model = Chain(Dense(d, d, relu), Dense(d, d, relu)) |> gpu;
loss(x, y) = Flux.Losses.mse(model(x), y);
src_idx, trg_idx = buildSeedDictionary(subx, subx);
src_idx, trg_idx, mparams = trainMapping(X, X, src_idx, trg_idx, model, loss, epochs=1000, opt=ADAM())
