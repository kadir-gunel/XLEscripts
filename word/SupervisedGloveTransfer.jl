cd(@__DIR__)
using OhMyREPL
using LinearAlgebra
using Statistics
using XLEs
using Embeddings
using CUDA
using BSON: @save, @load
using Printf


src, _, valfile = EmbeddingData() |> readData;
trgV, Y = map(i -> src[i], 1:2)

# load glove embeddigs
glove = load_embeddings(GloVe{:en}, 4, max_vocab_size=Int(200e3))
srcV, X = glove.vocab, glove.embeddings

X, Y = map(normalizeEmbedding, [X, Y])
X, Y = map(cu, [X, Y])

src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
vWords = readlines(valfile) .|> split .|> first .|> String |> unique # validation words

function readValidation2(vWords, src_w2i, trg_w2i)
    validation = Dict{Int64, Set}();
    oov   = Set{String}()
    vocab = Set{String}()
    for word in vWords
        try
            src_ind = src_w2i[word]
            trg_ind = trg_w2i[word]
            !haskey(validation, src_ind) ? validation[src_ind] = Set(trg_ind) : push!(validation[src_ind], trg_ind)
            push!(vocab, word) # adding word to vocab
        catch KeyError
            push!(oov, word)
        end

    end
    return validation
end

validation = readValidation2(vWords, src_w2i, trg_w2i)

x = X[:, collect(keys(validation))]
y = Y[:, collect(first.(values(validation)))]

W, _ = XLEs.mapOrthogonal2(y, x)

validate(W' * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)
validateCSLS(W' * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)

WX = W * X |> Array

@save "./data/SuperGlove.bson" WX

# there are 3 ways of rotation

W, _ = XLEs.mapOrthogonal2(x, y)
acc, sim = validate(W * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)
@printf "Accuracy of Rotation from X → Y : %.4f \n" acc
acc, sim = validate(X |> normalizeEmbedding, W' * Y |> normalizeEmbedding, validation)
@printf "Accuracy of Rotation^{T} from X → Y : %.4f \n"  acc

W, _ = XLEs.mapOrthogonal2(y, x)
acc, sim = validate(X |> normalizeEmbedding, W * Y |> normalizeEmbedding, validation)
@printf "Accuracy of Rotation from Y → X : %.4f \n" acc
acc, sim = validate(W' * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)
@printf "Accuracy of Rotation^{T} from Y → X : %.4f \n" acc
