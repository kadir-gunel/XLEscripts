using OhMyREPL
using LinearAlgebra
using Statistics
using BSON: @load
using XLEs
using Flux
using Printf

@load "W_es.bson" W_es
@load "W_it.bson" W_it
@load "W_fi.bson" W_fi
@load "W_de.bson" W_de 


@info "Loading Embedding Data"
src, es, valfileES = EmbeddingData() |> readData
_, it, valfileIT = EmbeddingData(trgLang="it") |> readData
_, fi, valfileFI = EmbeddingData(trgLang="fi") |> readData
_, de, valfileDE = EmbeddingData(trgLang="de") |> readData


enV, X = map(i -> src[i], 1:2) 
esV, Y_es = map(i -> es[i], 1:2)
itV, Y_it = map(i -> it[i], 1:2)
fiV, Y_fi = map(i -> fi[i], 1:2) 
deV, Y_de = map(i -> de[i], 1:2) 


@info "Summing all the rotation matrices"
W_global = W_es .+ W_it .+ W_fi .+ W_de |> normalizeEmbedding

# we only have W_global for rotating the X matrix.
# this W_global * X matrix is our XLEs for source language.


en_w2i, es_w2i, it_w2i, fi_w2i, de_w2i = map(word2idx, [enV, esV, itV, fiV, deV]);

valES = readValidation(valfileES, en_w2i, es_w2i);
valIT = readValidation(valfileIT, en_w2i, it_w2i);
valFI = readValidation(valfileFI, en_w2i, fi_w2i);
valDE = readValidation(valfileDE, en_w2i, de_w2i);


X, Y_es, Y_fi, Y_it, Y_de = map(normalizeEmbedding, [X, Y_es, Y_fi, Y_it, Y_de])


acc_es, sim_es = validate(W_es * X |> normalizeEmbedding |> cu , Y_es |> normalizeEmbedding |> cu, valES)
acc_it, sim_it = validate(W_it * X |> normalizeEmbedding |> cu , Y_it |> normalizeEmbedding |> cu, valIT)
acc_fi, sim_fi = validate(W_fi * X |> normalizeEmbedding |> cu , Y_fi |> normalizeEmbedding |> cu, valFI)
acc_de, sim_de = validate(W_de * X |> normalizeEmbedding |> cu , Y_de |> normalizeEmbedding |> cu, valDE)

@printf "Knn accuracy for ES: %.4f \n" acc_es
@printf "Knn accuracy for IT: %.4f \n" acc_it
@printf "Knn accuracy for FI: %.4f \n" acc_fi
@printf "Knn accuracy for DE: %.4f \n" acc_de


@info "XLEs accuracy values:"
acc_es, sim_es = validate(W_global * X |> normalizeEmbedding |> cu , Y_es |> normalizeEmbedding |> cu, valES)
acc_it, sim_it = validate(W_global * X |> normalizeEmbedding |> cu , Y_it |> normalizeEmbedding |> cu, valIT)
acc_fi, sim_fi = validate(W_global * X |> normalizeEmbedding |> cu , Y_fi |> normalizeEmbedding |> cu, valFI)
acc_de, sim_de = validate(W_global * X |> normalizeEmbedding |> cu , Y_de |> normalizeEmbedding |> cu, valDE)

@printf "Knn accuracy for ES: %.4f \n" acc_es
@printf "Knn accuracy for IT: %.4f \n" acc_it
@printf "Knn accuracy for FI: %.4f \n" acc_fi
@printf "Knn accuracy for DE: %.4f \n" acc_de



@info "Reading Training Data"
valTRN_ES = "../vecmap/data/dictionaries/en-es.train.shuf.txt"
valTRN_IT = "../vecmap/data/dictionaries/en-it.train.shuf.txt"
valTRN_FI = "../vecmap/data/dictionaries/en-fi.train.shuf.txt"
valTRN_DE = "../vecmap/data/dictionaries/en-de.train.shuf.txt"

valTRN_ES = readValidation(valTRN_ES, en_w2i, es_w2i);
valTRN_IT = readValidation(valTRN_IT, en_w2i, it_w2i);
valTRN_FI = readValidation(valTRN_FI, en_w2i, fi_w2i);
valTRN_DE = readValidation(valTRN_DE, en_w2i, de_w2i);



nn_es, _ = getIDX_NN(W_es * X |> normalizeEmbedding, Y_es |> normalizeEmbedding, valTRN_ES)
nn_it, _ = getIDX_NN(W_it * X |> normalizeEmbedding, Y_it |> normalizeEmbedding, valTRN_IT)
nn_fi, _ = getIDX_NN(W_fi * X |> normalizeEmbedding, Y_fi |> normalizeEmbedding, valTRN_FI)
nn_de, _ = getIDX_NN(W_de * X |> normalizeEmbedding, Y_de |> normalizeEmbedding, valTRN_DE)

@info "Forming new training set for Global Rotation Matrix"

x_src_es = valTRN_ES |> keys |> collect
x_src_it = valTRN_IT |> keys |> collect
x_src_fi = valTRN_FI |> keys |> collect
x_src_de = valTRN_DE |> keys |> collect

# source values for training 
X_es = X[:, x_src_es]
X_it = X[:, x_src_it]
X_fi = X[:, x_src_fi]
X_de = X[:, x_src_de]

X_trn = hcat(X_es, X_it, X_fi, X_de)

#  PREDICTED! target values for training
Ytrn_es = Y_es[:, nn_es |> vec]
Ytrn_it = Y_it[:, nn_it |> vec]
Ytrn_fi = Y_fi[:, nn_fi |> vec]
Ytrn_de = Y_de[:, nn_de |> vec]


Y_trn = hcat(Ytrn_es, Ytrn_it, Ytrn_fi, Ytrn_de)


# GOLDEN target values for training
Ytrn_es = Y_es[:, valTRN_ES |> values.|> first]
Ytrn_it = Y_it[:, valTRN_IT |> values.|> first]
Ytrn_fi = Y_fi[:, valTRN_FI |> values.|> first]
Ytrn_de = Y_de[:, valTRN_DE |> values.|> first]

Y_trn = hcat(Ytrn_es, Ytrn_it, Ytrn_fi, Ytrn_de)


function procrustes(XP, Y)
      F = svd(XP * Y');
      W = F.U * F.Vt
      return W'
end

R0 = procrustes(X_es |> normalizeEmbedding, Ytrn_es |> normalizeEmbedding)

@info "XLEs accuracy values:"

acc_es, sim_es = validate(R0 * X |> normalizeEmbedding |> cu , Y_es |> normalizeEmbedding |> cu, valES)
acc_it, sim_it = validate(R0 * X |> normalizeEmbedding |> cu , Y_it |> normalizeEmbedding |> cu, valIT)
acc_fi, sim_fi = validate(R0 * X |> normalizeEmbedding |> cu , Y_fi |> normalizeEmbedding |> cu, valFI)
acc_de, sim_de = validate(R0 * X |> normalizeEmbedding |> cu , Y_de |> normalizeEmbedding |> cu, valDE)

@printf "Knn accuracy for ES: %.4f \n" acc_es
@printf "Knn accuracy for IT: %.4f \n" acc_it
@printf "Knn accuracy for FI: %.4f \n" acc_fi
@printf "Knn accuracy for DE: %.4f \n" acc_de



