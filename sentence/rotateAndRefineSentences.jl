cd(@__DIR__)
using OhMyREPL
using CUDA
using XLEs
using Base.Iterators
using Base.Threads
using Flux
using Flux.Losses
using Flux.Data: DataLoader
using Flux.Optimise: update!, Descent, ADAM, RADAM
using MKL
using Printf
using Statistics
using LinearAlgebra
using BSON: @save, @load
using PyCall
using NPZ

@pyimport sacremoses as moses
@pyimport fasttext as ft
@pyimport sentence_transformers.SentenceTransformer as st

function mapOrthogonalSentences(X::T, Y::T) where {T}
    F = svd(X * Y')
    W = permutedims(F.U * F.Vt) # * cuinv((X * X') + λ .* CuMatrix{Float32}(I, 300, 300)))
    return W, F.S
end


function setValidationSet(samples::Int64)
    validation = Dict{Int64, Set}();
    for (s, t) in zip(1:samples, 1:samples)
        push!(validation, s => Set(t))
    end
    return validation
end

function fine_tune(Ŷtrn, Ytrn, Ŷtst, Ytst, loss, model; opt=ADAM(.005), epochs::Int64=10, validationTRN=validationTRN, validationTST=validationTST)
    @printf "Validation Scores before Training : \n"
    #accuracyTRN, _ = validate(model(Ŷtrn), Ytrn, validationTRN);
    #accuracyTST, _ = validate(model(Ŷtst), Ytst, validationTST);
    # @printf "Epoch: %1i loss(train): %.4f, accuracy(train): %.4f, loss(test): %.4f, accuracy(test): %.4f \n" 0 loss(Ŷtrn, Ytrn) accuracyTRN loss(Ŷtst, Ytst) accuracyTST
    D = repeated((Ŷtrn, Ytrn), 100)
    for epoch in 1:epochs
        Flux.train!(loss, Flux.params(model), D, opt)
        model_cpu = model |> cpu
        accuracyTRN, _ = validate(model_cpu(Ŷtrn |> Array), Ytrn |> Array, validationTRN);
        accuracyTST, _ = validate(model(Ŷtst), Ytst, validationTST);
        @printf "Epoch: %1i loss(train): %.4f, accuracy(train): %.4f, loss(test): %.4f, accuracy(test): %.4f \n" epoch loss(Ŷtrn, Ytrn) accuracyTRN loss(Ŷtst, Ytst) accuracyTST
    end
    return model
end


mt = moses.MosesTokenizer("en")

# bert_model = st.SentenceTransformer("all-MiniLM-L6-v2", device="cuda:0")
bert_model = st.SentenceTransformer("all-mpnet-base-v2", device="cuda:0")
ft_model   = ft.load_model("/home/PhD/github/DATA/FT/cc.en.300.bin")

#read data

#wmt = readlines("/home/PhD/Downloads/europarl-v7.es-en.en")
#wmt_lines = Vector{String}(undef, length(wmt))
@load "./data/W_wmt_normalized.bson" W

# start of fine tuning process
lines = readlines("./data/5000Sents.txt")
sentences = String[]
for (i, sentence) in collect(enumerate(lines))
     push!(sentences, mt.tokenize(sentence, return_str=true) |> lowercase)
end


ft_embeddings = []

for (i, sentence) in collect(enumerate(sentences))
    push!(ft_embeddings, ft_model.get_sentence_vector(sentence))
end

ft_embeddings = reduce(hcat, ft_embeddings) |> normalizeEmbedding;


bert_embeddings = bert_model.encode(sentences, device="cuda:0")
bert_embeddings = bert_embeddings |> permutedims |> normalizeEmbedding

#@save "./data/wmt_bert_all.bson" bert_embeddings
#@save "./data/wmt_ft_all.bson" ft_embedings

# ft_embeddings =   NPZ.npzread("./data/wmt_ft_all.npy")
# bert_embeddings = NPZ.npzread("./data/wmt_bert_all.npy")

# ft_embeddings =   NPZ.npzread("./data/FT.npy") |> permutedims |> normalizeEmbedding
# bert_embeddings = NPZ.npzread("./data/Bert.npy") |> permutedims |> normalizeEmbedding
# W = NPZ.npzread("./data/Wsent.npy") |> permutedims


# W, _ = mapOrthogonalSentences(ft_embeddings, bert_embeddings)
# @load "./data/W_wmt.bson" W

rotX = W * ft_embeddings;
# Y = bert_embeddings;


n = 1:Int(40e3)
t = Int(40e3)+1:Int(50e3)
trainX = W * ft_embeddings[:, n] |> normalizeEmbedding
trainY = bert_embeddings[:, n] |> normalizeEmbedding
testX = W * ft_embeddings[:, t] |> normalizeEmbedding
testY = bert_embeddings[:, t] |> normalizeEmbedding

validationTRN = setValidationSet(length(n))
validationTST = setValidationSet(length(t))


mse(rotX, bert_embeddings)
validate(rotX |> normalizeEmbedding, bert_embeddings |> normalizeEmbedding, validationTRN)

Xtrn, Ytrn, Xtst, Ytst = map(cu, [trainX, trainY, testX, testY])

#Ŷtrn = W * Xtrn;
# Ŷtst = W * Xtst;

accuracy, similarity = validate(Xtrn |> Array, Ytrn |> Array, validationTRN)
accuracy, similarity = validate(Xtst, Ytst, validationTST)

# accuracy, similarity = validate(Xtrn[:, 1:Int(30e3)], Ytrn[:, 1:Int(30e3)], validationTRN)
# accuracy, similarity = validate(Xtst, Ytst, validationTST)

d = size(Xtrn, 1)
model = Chain(Dense(d, d, relu6)) |> gpu
loss(x, y) = Flux.Losses.mse(model(x), y);
loss2(x, y) = Flux.Losses.siamese_contrastive_loss(model(x), y; margin=.5);
model = fine_tune(Xtrn, Ytrn, Xtst, Ytst, loss, model; epochs=5)

accuracy, similarity = validate(W * Xtrn, Ytrn, validationTRN)
accuracy, similarity = validate(model(Ŷtrn) |> normalizeEmbedding, Ytrn |> normalizeEmbedding, validationTRN)

accuracy, similarity = validate(Ŷtst, Ytst, validationTST)
accuracy, similarity = validate(model(Ŷtst) |> normalizeEmbedding, Ytst |> normalizeEmbedding, validationTST)

tstparts  = collect(partition(validationTST, Int(10e3))) .|> Dict ;
accuracy, similarity = validateCSLS(model(@view(Ŷtst[:, 1:Int(10e3)])) |> normalizeEmbedding, @view(Ytst[:, 1:Int(10e3)]) |> normalizeEmbedding, tstparts[1])




model = toTrain(W * Xtrn, Ytrn, W * Xtst, Ytst, loss, validationTRN=validationTRN, validationTST=validationTST)

accTRN, simTRN = validate((W * trainX) |> normalizeEmbedding |> cu , trainY |> normalizeEmbedding |> cu, validationTRN)
accTST, simTST = validate((W * testX) |> normalizeEmbedding |> cu , testY |> normalizeEmbedding |> cu, validationTST)
