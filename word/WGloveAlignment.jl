cd(@__DIR__)
using OhMyREPL
using LinearAlgebra
using Statistics
using XLEs
using CUDA
using BSON: @save, @load
using Printf


_, trg, val = EmbeddingData(trgLang="es") |> readData;
trgV, Y = map(i -> trg[i], 1:2)

# @load "./data/RGlove.bson" XW
# @load "./data/RGlove_Voc.bson" srcV
# srcV = srcV .|> String

X, Y = map(normalizeEmbedding, [X, Y]);
X, Y = map(cu, [X, Y]);


src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
validation = readValidation(val, src_w2i, trg_w2i);

rng = 1:Int(4e3)
x = X[:, rng] #|> Matrix
y = Y[:, rng] #|> Matrix



# @time src_idx, trg_idx = XLEs.buildSeedDictionary0(subx, suby)
@time src_idx, trg_idx = buildSeedDictionary(x, y)
@time W, src_idx, trg_idx = XLEs.main(X, Y, src_idx, trg_idx, validation, src_size=Int(10e3), trg_size=Int(10e3));
