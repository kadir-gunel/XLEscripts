cd(@__DIR__)
using OhMyREPL
using LinearAlgebra
using Statistics
using XLEs
using CUDA
using BSON: @save, @load
using Printf

# getSentences(file) = file |> readlines .|> l -> split(l) .|> i -> String(i)




src, trg, val = EmbeddingData(trgLang="es") |> readData;
srcV, X = map(i -> src[i],  1:2)
trgV, Y = map(i -> trg[i],  1:2)

X, Y = map(normalizeEmbedding, [X, Y]);
X, Y = map(cu, [X, Y]);

src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
validation = readValidation(val, src_w2i, trg_w2i);

rng = 1:Int(4e3)
x = X[:, rng] #|> Matrix
y = Y[:, rng] #|> Matrix


# @time src_idx, trg_idx = XLEs.buildSeedDictionary0(subx, suby)
@time src_idx, trg_idx = buildSeedDictionary(x, y)
# @time src_idx, trg_idx = XLEs.buildMahalanobisDictionary(subx |> Matrix, suby |> Matrix);
# @time src_idx, trg_dix = XLEs.mahalanobisGPU(subx, suby);
W, src_idx, trg_idx = XLEs.main(X, Y, src_idx, trg_idx, validation);

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

kacc1, sims1 = validate(W * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)
kacc2, sims2 = validateCSLS(W * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)


kacc1, sims1 = validate(Wall * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)
kacc2, sims2 = validateCSLS(Wall * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)



#=
val2 = "/home/PhD/github/MUSE/data/crosslingual/dictionaries/en-it.5000-6500.txt"
validation2 = readValidation(val2, src_w2i, trg_w2i)

validate(W * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation2)
validate(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation2)
validateCSLS(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation2)

XW, YW = advancedMapping(permutedims(W * X), permutedims(Y), src_idx, trg_idx);
kacc3, sims3 = validate(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)


kacc4, sims4= validateCSLS(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)


@printf "           |Accuracy | Similarity"
@printf "==========================================="
@printf "KNN        | %.4f  |  %.4f" kacc1 sims2
@printf "CSLS       | %.4f  |  %.4f" kacc2 sims2
@printf "------------------------------------------"
@printf "KNN_{adv}  | %.4f  |  %.4f" kacc3 sims3
@printf "CSLS_{adv} | %.4f  |  %.4f" kacc4 sims4


=#





#=
list = [W, src_idx, trg_idx];
W, src_idx, trg_idx = map(Array, list)

@save "./W_vecmap.bson" W
@save "./src_vecmap.bson" src_idx
@save "./trg_vecmap.bson" trg_idx


info = SplitInfo();
Postprocessing(Matrix(X), Matrix(Y), Array(src_idx), Array(trg_idx), replaceSingulars, info, validation, srcV, trgV ) |> validateModel
Postprocessing(Matrix(X), Matrix(Y), Array(src_idx), Array(trg_idx), (x) -> (x), info, validation, srcV, trgV ) |> validateModel

info2 = SplitInfo(change=true)
Postprocessing(Matrix(X), Matrix(Y), Array(src_idx), Array(trg_idx), replaceSingulars, info2, validation, srcV, trgV ) |> validateModel
Postprocessing(Matrix(X), Matrix(Y), Array(src_idx), Array(trg_idx), (x) -> (x), info2, validation, srcV, trgV ) |> validateModel


=#
