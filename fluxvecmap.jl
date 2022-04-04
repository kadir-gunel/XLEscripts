using LinearAlgebra
using Statistics
using OhMyREPL
using Flux
using Flux.Optimise: update!, Descent
using XLEs
using Printf


src, trg, valfile = EmbeddingData(datapath="/Users/kadirgunel/github/FluxVecMap/data/exp_raw/") |> readData;

srcV, X = map(i -> src[i], 1:2)
trgV, Y = map(i -> trg[i], 1:2)

src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
validation = readValidation(valfile, src_w2i, trg_w2i);

X, Y = map(normalizeEmbedding, [X, Y]);
rng = 1:Int(4e3)
subx = X[:, rng]
suby = Y[:, rng]

d, w = size(subx)

opt = Descent(0.008)
m = Chain(Dense(d, d, relu), Dense(d, d, relu))
loss(x, y) = Flux.Losses.mse(m(x), y)
for epoch = 1:100
  @printf "Epoch: %1i, loss: %.4f \n" epoch loss(subx, suby)
  Flux.train!(loss, params(m), [(subx, suby)], opt)
end
