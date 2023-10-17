cd(@__DIR__)

using Pkg
Pkg.activate("/raid/vecmap_jl")

using LinearAlgebra
using Statistics
using StatsBase
using XLEs
using CUDA
using BSON: @save, @load
using Printf



# getSentences(file) = file |> readlines .|> l -> split(l) .|> i -> String(i)

vecmap = "/run/media/phd/PhD/vecmap/data/embeddings/" 

# srcfile = vecmap * "Glove"
# trgfile = vecmap * "it"



# @time srcV, X = readBinaryEmbeddings(srcfile);
# @time trgV, Y = readBinaryEmbeddings(trgfile);


srcfile = "/raid/glove/glove-unit-text8_seed-6666.txt"
trgfile = "/raid/glove/glove-unit-text8_seed-9876.txt"

@time srcV, X = readEmbeddings(srcfile);
@time trgV, Y = readEmbeddings(trgfile);



# X, Y = map(permutedims, [X, Y])

# X, Y = map(normalizeEmbedding, [X, Y]);
# X, Y = map(XLEs.center, [X, Y]);
X, Y = map(cu, [X, Y]);

val = "/raid/vecmap/data/dictionaries/en-en.test.txt"

src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
validation = readValidation(val, src_w2i, trg_w2i);

# Fs = svd(X);
# Ft = svd(Y);

# snew = log.((Fs.S + Ft.S)) |> diagm |> cu

# Xnew = cu(diagm((log.(Fs.S)).^2)) * Fs.Vt
# Ynew = cu(diagm((cbrt.(Ft.S)).^2)) * Ft.Vt

rng = 1:Int(4e3)
x = X[:, rng]  #|> Matrix
y = Y[:, rng]  #|> Matrix


# @time src_idx, trg_idx = XLEs.buildSeedDictionary0(subx, suby)
# @time src_idx, trg_idx = XLEs.buildRBFSeedDictionary(x, y, threshold=0.6, k=40);
@time src_idx, trg_idx = XLEs.buildSeedDictionary(x, y, sim_size=length(rng));
# @time src_idx3, trg_idx3 = XLEs.buildCosineDictionary(x |> Matrix, y |> Matrix)
# @time src_idx3, trg_idx3 = XLEs.buildMahalanobisDictionary(x |> Matrix, y |> Matrix)
# @time src_idx, trg_dix = XLEs.mahalanobisGPU(subx, suby);
@time W, src_idx, trg_idx = XLEs.main(X, Y, src_idx, trg_idx, validation, src_size=Int(20e3), trg_size=Int(20e3));

kacc1, sims1 = validate(W * X |> normalizeEmbedding , Y |> normalizeEmbedding, validation)
kacc2, sims2 = validateCSLS(W * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)


XW, YW = advancedMapping(permutedims(W * X), permutedims(Y), src_idx, trg_idx);
kacc1, sims1 = validate(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)
kacc2, sims2 = validateCSLS(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)


# this is normal advanced mapping like the above ones.
Postprocessing(Matrix(X), Matrix(Y), Array(src_idx), Array(trg_idx), (x) -> (x), info, validation, srcV, trgV ) |> validateModel
# this is where we replace the singular values
info = SplitInfo();
Postprocessing(Matrix(X), Matrix(Y), Array(src_idx), Array(trg_idx), replaceSingulars, info, validation, srcV, trgV ) |> validateModel

W = W |> Array

@save "./models/Glove_en.bson" W 



W_es = W |> Array |> deepcopy
W_it = W |> Array |> deepcopy
W_de = W |> Array |> deepcopy
W_fi = W |> Array |> deepcopy


@save "./W_es.bson" W_es
@save "./W_it.bson" W_it
@save "./W_de.bson" W_de
@save "./W_fi.bson" W_fi

Wall = (W_it + W_de + W_fi) |> cu




W, src_idx, trg_idx = XLEs.main(X, Y, src_idx, trg_idx, validation, Wt=Wall);

kacc1, sims1 = validate(W * X |> normalizeEmbedding , Y |> normalizeEmbedding, validation)
kacc2, sims2 = validateCSLS(W * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)



XW, YW = advancedMapping(permutedims(W * X), permutedims(Y), src_idx, trg_idx);
kacc1, sims1 = validate(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)
kacc1, sims1 = validateCSLS(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)
