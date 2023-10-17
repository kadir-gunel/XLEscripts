using OhMyREPL
using LinearAlgebra
using Statistics
using Base.Iterators
using Embeddings
using XLEs
using Flux
using Flux.Losses
using Flux.Data: DataLoader
using Flux.Optimise: update!, Descent, ADAM, RADAM
using Printf
using CUDA


function procrustes(XP, Y)
      F = svd(XP * Y');
      W = F.U * F.Vt
      return W'
end

glove = load_embeddings(GloVe{:en}, 3, max_vocab_size=Int(200e3))
ft = load_embeddings(FastText_Text{:en}, 1, max_vocab_size=Int(200e3))

srcVGlove, G = glove.vocab, glove.embeddings
srcVFT, F = ft.vocab, ft.embeddings

G, F = map(normalizeEmbedding, [G, F])

src_w2iGlove, src_w2iFT = map(word2idx, [srcVGlove, srcVFT]);
trainVal = "../en_data_train.txt"
validation = readValidation(trainVal, src_w2iGlove, src_w2iFT);


src_idx = validation |> keys |> collect
trg_idx = validation |> values .|> first


@printf "MSE [Glove → FT] before train: %.4f \n" Flux.mse(G[:, src_idx], F[:, trg_idx])
R0 = procrustes(G[:, src_idx], F[:, trg_idx]) |> Matrix

# @printf "MSE [Glove → FT] before train: %.4f\n" Flux.mse(G[:, src_idx], F[:, trg_idx])
@printf "MSE [Glove → FT] after train: %.4f \n" Flux.mse(R0 * G[:, src_idx], F[:, trg_idx])


knn_acc, sim = validateCSLS(R0 * G |> cu, F |>  cu, validation)
@printf "Accuracy on train set: %.4f\n" knn_acc


testVal = "../en_data_test.txt"
validationTest = readValidation(testVal, src_w2iGlove, src_w2iFT);
knn_acc, sim = validateCSLS(R0 * G |> cu, F |>  cu, validationTest)
@printf "Accuracy on test set: %.4f\n" knn_acc


# now let's build a NN for eliminating the 0 values inside the rotation matrix R0 * G

function fine_tune(Ŷtrn, Ytrn, Ŷtst, Ytst, loss,  model; opt=ADAM(.005), epochs::Int64=10, validationTRN=validation, validationTST=validationTest)
    @printf "Validation Scores before Training : \n"
    D = repeated((Ŷtrn, Ytrn), 100)
    for epoch in 1:epochs
        Flux.train!(loss, Flux.params(model), D, opt);
        accuracyTRN, _ = validate(model(Ŷtst), Ytst, validationTRN);
        accuracyTST, _ = validate(model(Ŷtst), Ytst, validationTST);
        @printf "Epoch: %1i loss(train): %.4f, accuracy(train): %.4f, loss(test): %.4f, accuracy(test): %.4f \n" epoch loss(Ŷtrn, Ytrn) accuracyTRN  loss(Ŷtst, Ytst) accuracyTST
    end
    return model
end

Xtrn = G[:, src_idx] |> cu
Ytrn = F[:, trg_idx] |> cu

X_tst = G |> cu 
Y_tst = F |> cu

# d  = size(x, 1)
# model = Chain(Dense(100, 300, tanh)) |> gpu # if we don't use R0, it takes too much time and the result is sub optimal!
# model = Chain(Dense(R0, true, elu), Dense(W, true, tanh)) |> gpu

model = Chain(Dense(R0, true, elu), Dense(W, true, elu)) |> gpu


# model = Chain(Dense(300, 300, tanh)) |> gpu
loss(x, y) = Flux.Losses.mse(model(x), y);
#loss2(x, y) = Flux.Losses.siamese_contrastive_loss(model(x), y; margin=.5);
#loss3(x, y) = Flux.Losses.huber_loss(model(x), y)
# loss4(x, y) = Flux.Losses.msle(model(x), y)

model = fine_tune(Xtrn, Ytrn, X_tst, Y_tst, loss, model; opt=ADAM(5e-4), epochs=7, validationTRN=validation, validationTST=validationTest)


@info "Before NN Training:"

knn_r0, _ = validate(cu(R0) * X_tst, Y_tst, validation)
@printf "Train Accuracy : %.4f , MSE: %.4f\n" knn_r0 mse(R0 * G, F)
knn_r0, _ = validate(cu(R0) * X_tst, Y_tst, validationTest)
@printf "Test Accuracy : %.4f\n" knn_r0



@info "After NN training:"
knn_nn, _ = validateCSLS(model(X_tst), Y_tst, validation)
@printf "Train Accuracy : %.4f, MSE: %.6f\n" knn_nn mse(G |> cu |> model, F |> cu)
knn_nn, _ = validateCSLS(model(X_tst), Y_tst, validationTest)
@printf "Test Accuracy : %.4f \n" knn_nn



"""
NOTES:
  1. G → F 'e gecerken eger sadece NN kullanirsak, maximum alabilecegimiz deger ≈ 64%
 veya 65%. Yani sanki sadece rotasyon matrisini kullanarak aldigimiz sonuca benziyor.
  2. Ancak eger ilk once Rotasyon matrisini bulursak ve daha sonra nn ile non-linear hale getirirsek aldigimiz test sonuclari +4%, train ise +2% oluyor. Bu durum 100 → 300 icin gecerli.
     - Ancak 50 → 300 yaparsak, bu durumda acc 37% seviyesine iniyor. Ve NN ile sonuc 44%'e kadar cikiyor. Ki bu da +7% lik bir artis demek. 
    - 200 → 300 yaparsak, $1.5% test accuracy artiyor.
  3. α (lr.) degerini oldukca kucuk ayarlamak gerekiyor. 
"""




"""
NOTES-2:
    Rotate edilmis ve "fine-tune" edilmis Glove word embeddingleri ile orjinal glove embeddinglerini karsilastirmak istersek. Rotate edilmis olanlarla almanca ceviriden aldigimiz sonuclarin orjinallere gore daha iyi cikmasini bekleriz.
    Asagida bunun icin hazirlanmis deney bulunmaktadir.
"""

"""
   1. Orjinla Glove embeddinglerinden vecmap almancaya embeddinglerine ceviri:
"""

_, trg, _ = EmbeddingData(trgLang="es") |> readData
trgV, Y = map(i -> trg[i], 1:2)

# glove = load_embeddings(GloVe{:en}, 3, max_vocab_size=Int(200e3))
# srcVGlove, G = glove.vocab, glove.embeddings

G, Y = map(normalizeEmbedding, [G, Y])

src_w2i, trg_w2i = map(word2idx, [srcVGlove, trgV])
trainVal = "../vecmap/data/dictionaries/en-es.train.shuf.txt"
testVal = "../vecmap/data/dictionaries/en-es.test.txt"


valTrain = readValidation(trainVal, src_w2i, trg_w2i)
valTest  = readValidation(testVal, src_w2i, trg_w2i)

src_idx = valTrain |> keys |> collect
trg_idx = valTrain |> values .|> first

"""
   2. Model tarafindan rotate edilmis Glove WE ile Vecmap_DE WE'lerinin sonuclari:

    - 1. Sadece svd ile rotasyon islemi gerceklenirse:

"""

RWG = procrustes(G[:, src_idx] |> cu |>  model, Y[:, trg_idx] |> cu) |> Matrix

# RWG = procrustes(F[:, src_idx] |> cu, Y[:, trg_idx] |> cu) |> Matrix

R = deepcopy(cu(RWG))

@printf "MSE [Glove → Vec] after train: %.4f \n" Flux.mse(R * (G[:, src_idx] |> cu |> model), Y[:, trg_idx]|> cu)

knn_acc, sim = validate(R * (G |> cu |> model) , Y |>  cu , valTrain)
@printf "Accuracy on train Set: %.4f\n" knn_acc

knn_acc, sim = validate(R * (G |> cu |> model), Y |>  cu, valTest)
@printf "Accuracy on test  Set: %.4f\n" knn_acc



"""
    - 2. svd + nn

"""
d,d = size(R)
W = Flux.orthogonal(d, d)
model2 = Chain(Conv((4, 4), 3 => 8, tanh), Dense(R, true, elu)) |> gpu

Xtrn = G[:, src_idx] |> cu |> model
# Xtrn = F[:, src_idx] |> cu
Ytrn = Y[:, trg_idx] |> cu 

X_tst = G |> cu |> model
# X_tst = F |> cu
Y_tst = Y |> cu

loss2(x, y) = Flux.Losses.mse(model2(x), y);

model2 = fine_tune(Xtrn, Ytrn, X_tst, Y_tst, loss2, model2; opt=ADAM(3e-6), epochs=100, validationTRN=valTrain, validationTST=valTest)

@info "Before NN Training:"

knn_r0, _ = validate(cu(RWG) * X_tst |> cu , Y_tst |> cu , valTrain)
@printf "Train Accuracy : %.4f , MSE: %.4f\n" knn_r0 mse(cu(RWG) * Xtrn, Ytrn)
knn_r0, _ = validateCSLS(cu(RWG) * X_tst, Y_tst, valTest)
@printf "Test Accuracy : %.4f\n" knn_r0


@info "After NN training:"
knn_nn, _ = validateCSLS(model2(X_tst) , Y_tst , valTrain)
@printf "Train Accuracy : %.4f\n" knn_nn
knn_nn, _ = validateCSLS(model2(X_tst), Y_tst , valTest)
@printf "Train Accuracy : %.4f \n" knn_nn




RG = G |> cu |> model |> cpu

@printf "MSE [Glove → Vec] after train: %.4f \n" Flux.mse(RG[:, src_idx], Y[:, trg_idx])
knn_acc, sim = validate(RG |> cu, Y |>  cu, valTrain)
@printf "Accuracy on train set: %.4f\n" knn_acc



knn_acc, sim = validate(RG |> cu, Y |>  cu, valTest)
@printf "Accuracy on test set: %.4f\n" knn_acc
