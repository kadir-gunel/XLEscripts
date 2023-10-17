using OhMyREPL
using XLEs
using PyCall
using LinearAlgebra
using Statistics
using CUDA
using MKL

@pyimport fasttext as ft

_, _, val = EmbeddingData(trgLang="es") |> readData;

EN = ft.load_model("/home/PhD/cc.en.200.bin")
ES = ft.load_model("/home/PhD/cc.es.300.bin")

srcV = EN.get_words()
trgV = ES.get_words()

X = EN.get_word_vector.(srcV) |> hcat
Y = ES.get_word_vector.(trgV) |> hcat

rng = 1:Int(200e3)

srcV = srcV[rng]
trgV = trgV[rng]

X = reduce(hcat, X[rng])
Y = reduce(hcat, Y[rng])

X, Y = map(normalizeEmbedding, [X, Y]);
X, Y = map(cu, [X, Y]);

src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
validation = readValidation(val, src_w2i, trg_w2i);

rng = 1:Int(4e3)
x = X[:, rng] #|> Matrix
y = Y[:, rng] #|> Matrix



# @time src_idx, trg_idx = XLEs.buildSeedDictionary0(subx, suby)
@time src_idx, trg_idx = buildSeedDictionary(x, y)
@time W, src_idx, trg_idx = XLEs.main(X, Y, src_idx, trg_idx, validation, src_size=Int(5e3), trg_size=Int(5e3));


kacc1, sims1 = validate(W * X |> normalizeEmbedding , Y |> normalizeEmbedding, validation)
kacc2, sims2 = validateCSLS(W * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)

XW, YW = advancedMapping(permutedims(W * X), permutedims(Y), src_idx, trg_idx);
kacc1, sims1 = validate(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)
kacc1, sims1 = validateCSLS(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)
