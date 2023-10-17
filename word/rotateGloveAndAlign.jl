using OhMyREPL
using LinearAlgebra
using Statistics
using Printf
using XLEs
using CUDA
using Embeddings
using Flux

function procrustes(XP, Y)
      F = svd(XP * Y');
      W = F.U * F.Vt
      return W'
end


glove = load_embeddings(GloVe{:en}, 4, max_vocab_size=Int(400e3))
ft = load_embeddings(FastText_Text{:en}, 1, max_vocab_size=Int(400e3))

srcVGlove, G = glove.vocab, glove.embeddings;
srcVFT, F = ft.vocab, ft.embeddings;


G, F = map(normalizeEmbedding, [G, F])

src_w2iGlove, src_w2iFT = map(word2idx, [srcVGlove, srcVFT]);
trainVal = "../en_data_train.txt"
validation = readValidation(trainVal, src_w2iGlove, src_w2iFT);


src_idx = validation |> keys |> collect
trg_idx = validation |> values .|> first

R0 = procrustes(G[:, src_idx], F[:, trg_idx])
@printf "MSE [Glove → FT] before train: %.4f\n" Flux.mse(G[:, src_idx], F[:, trg_idx])
@printf "MSE [Glove → FT] after train: %.4f \n" Flux.mse(R0 * G[:, src_idx], F[:, trg_idx])


knn_acc, sim = validate(R0 * G |> cu, F |>  cu, validation)
@printf "Accuracy on train set: %.4f\n" knn_acc




testVal = "../en_data_test.txt"
validationTest = readValidation(testVal, src_w2iGlove, src_w2iFT);
knn_acc, sim = validate(R0 * G |> cu, F |>  cu, validationTest)
@printf "Accuracy on test set: %.4f\n" knn_acc


# now lets do it with Glove and vecmap FT embeddings


src, trg, val = EmbeddingData(trgLang="de") |> readData;
srcV, X = map(i -> src[i],  1:2)
trgV, Y = map(i -> trg[i],  1:2)


X, Y = map(normalizeEmbedding, [X, Y])
src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);



trainVal = "../en_data_train.txt"
validation = readValidation(trainVal, src_w2iGlove, src_w2i);


src_idx = validation |> keys |> collect
trg_idx = validation |> values .|> first

R0 = procrustes(G[:, src_idx], X[:, trg_idx])
@printf "MSE [Glove → Vec] before train: %.4f\n" Flux.mse(G[:, src_idx], X[:, trg_idx])
@printf "MSE [Glove → Vec] after train: %.4f \n" Flux.mse(R0 * G[:, src_idx], X[:, trg_idx])


knn_acc, sim = validate(R0 * G |> cu, X |>  cu, validation)
@printf "Accuracy on train set: %.4f\n" knn_acc


testVal = "../en_data_test.txt"
validationTest = readValidation(testVal, src_w2iGlove, src_w2i);
knn_acc, sim = validate(R0 * G |> cu, X |>  cu, validationTest)
@printf "Accuracy on train set: %.4f\n" knn_acc



@info " (Glove_en → R × Glove_en) → FT_de Supervised Rotation"
RGlove = R0 * G
# srcVGlove


src_w2i, trg_w2i = map(word2idx, [srcVGlove, trgV])
trainVal = "../vecmap/data/dictionaries/en-fi.train.shuf.txt"
testVal = "../vecmap/data/dictionaries/en-fi.test.txt"


valTrain = readValidation(trainVal, src_w2i, trg_w2i)
valTest  = readValidation(testVal, src_w2i, trg_w2i)

src_idx = valTrain |> keys |> collect
trg_idx = valTrain |> values .|> first
W = procrustes(RGlove[:, src_idx], Y[:, trg_idx])

@printf "MSE [Glove → Vec] before train: %.4f\n" Flux.mse(RGlove[:, src_idx], Y[:, trg_idx])
@printf "MSE [Glove → Vec] after train: %.4f \n" Flux.mse(W * RGlove[:, src_idx], Y[:, trg_idx])

knn_acc, sim = validate(W * RGlove |> cu, Y |>  cu, valTrain)
@printf "Accuracy on train set: %.4f\n" knn_acc



knn_acc, sim = validate(W * RGlove |> cu, Y |>  cu, valTest)
@printf "Accuracy on test set: %.4f\n" knn_acc




# what happens to glove vectors if we rotate them by using  original vecmap english word embedding ?

@info "Rotation of Glove_en → Vecmap_en : "

src, _, val = EmbeddingData(trgLang="de") |> readData;
srcV, X = map(i -> src[i],  1:2)

glove = load_embeddings(GloVe{:en}, 4, max_vocab_size=Int(400e3))
srcVGlove, G = glove.vocab, glove.embeddings;


X, G = map(normalizeEmbedding, [X, G])
src_w2i, trg_w2i = map(word2idx, [srcVGlove, srcV]);



trainVal = "../en_data_train.txt"
validation = readValidation(trainVal, src_w2i, trg_w2i);


src_idx = validation |> keys |> collect
trg_idx = validation |> values .|> first

R0 = procrustes(G[:, src_idx], X[:, trg_idx])
@printf "MSE [Glove → Vec] before train: %.4f\n" Flux.mse(G[:, src_idx], X[:, trg_idx])
@printf "MSE [Glove → Vec] after train: %.4f \n" Flux.mse(R0 * G[:, src_idx], X[:, trg_idx])


knn_acc, sim = validate(R0 * G |> cu, X |>  cu, validation)
@printf "Accuracy on train set: %.4f\n" knn_acc


testVal = "../en_data_test.txt"
validationTest = readValidation(testVal, src_w2i, trg_w2i);
knn_acc, sim = validate(R0 * G |> cu, X |>  cu, validationTest)
@printf "Accuracy on train set: %.4f\n" knn_acc




@info " Supervised Rotation of Vecmap_en → Vecmap_de:"


src, trg, val = EmbeddingData(trgLang="de") |> readData;
srcV, X = map(i -> src[i], 1:2)
trgV, Y = map(i -> trg[i], 1:2)


X, Y = map(normalizeEmbedding, [X, Y])
src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);



trainVal = "../vecmap/data/dictionaries/en-de.train.shuf.txt"
validation = readValidation(trainVal, src_w2i, trg_w2i);


src_idx = validation |> keys |> collect
trg_idx = validation |> values .|> first

R0 = procrustes(X[:, src_idx], Y[:, trg_idx])
@printf "MSE [Glove → Vec] before train: %.4f\n" Flux.mse(X[:, src_idx], Y[:, trg_idx])
@printf "MSE [Glove → Vec] after train: %.4f \n" Flux.mse(R0 * X[:, src_idx], Y[:, trg_idx])


knn_acc, sim = validate(R0 * X |> cu, Y |>  cu, validation)
@printf "Accuracy on train set: %.4f\n" knn_acc


testVal = "../vecmap/data/dictionaries/en-de.test.txt"
validationTest = readValidation(testVal, src_w2i, trg_w2i);
knn_acc, sim = validate(R0 * X |> cu, Y |>  cu, validationTest)
@printf "Accuracy on train set: %.4f\n" knn_acc
