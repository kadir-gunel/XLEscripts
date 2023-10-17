cd(@__DIR__)
using OhMyREPL
using Random
using Base.Iterators
using Printf
using Statistics
using LinearAlgebra
using MKL



using CSV

using DataFrames


using PGFPlotsX

using CUDA
using XLEs

using Flux
using Flux.Losses
using Flux.Data: DataLoader
using Flux.Optimise: update!, Descent, ADAM, RADAM

using BSON: @save, @load
using PyCall
using NPZ

@pyimport sacremoses as moses
@pyimport fasttext as ft
@pyimport sentence_transformers.SentenceTransformer as st


function loadFT(sentences, path::String="/home/PhD/github/DATA/FT/cc.en.300.bin")
    ft_embeddings = []
    ft_model   = ft.load_model(path)
    for (i, sentence) in collect(enumerate(sentences))
        push!(ft_embeddings, ft_model.get_sentence_vector(sentence))
    end
    @info "Normalizing Embeddings"
    ft_embeddings = reduce(hcat, ft_embeddings) |> normalizeEmbedding;
    return ft_embeddings
end

function loadsBert(sentences)
    @info "Loading ALL-MPNET-BASE-v2 model"
    @time bert_model = st.SentenceTransformer("all-mpnet-base-v2", device="cuda:0")
    @info "Processing Sentences (this might take time)"
    @time bert_embeddings = bert_model.encode(sentences, device="cuda:0", batch_size=256)
    @info "Normalizing Embeddings"
    return bert_embeddings |> permutedims |> normalizeEmbedding
end

function fine_tune(Ŷtrn, Ytrn, Ŷtst, Ytst, loss, model; opt=ADAM(.005), epochs::Int64=10, validationTRN=validationTRN, validationTST=validationTST)
    @printf "Validation Scores before Training : \n"
    #=
    accuracyTRN, _ = validate(Ŷtrn |> Array, Ytrn |> Array, validationTRN);
    accuracyTST, _ = validate(Ŷtst |> Array, Ytst |> Array, validationTST);
    =#
    # @printf "Epoch: %1i loss(train): %.4f, accuracy(train): %.4f, loss(test): %.4f, accuracy(test): %.4f \n" 0 loss(Ŷtrn, Ytrn) accuracyTRN loss(Ŷtst, Ytst) accuracyTST
    D = repeated((Ŷtrn, Ytrn), 100)
    for epoch in 1:epochs
        Flux.train!(loss, Flux.params(model), D, opt)
        # model_cpu = model |> cpu
        accuracyTRN, _ = validate(Ŷtrn |> normalizeEmbedding , Ytrn |> normalizeEmbedding , validationTRN);
        accuracyTST, _ = validate(model(Ŷtst) |> normalizeEmbedding, Ytst |> normalizeEmbedding, validationTST);
        @printf "Epoch: %1i loss(train): %.4f, accuracy(train): %.4f, loss(test): %.4f, accuracy(test): %.4f \n" epoch loss(Ŷtrn, Ytrn) accuracyTRN loss(Ŷtst, Ytst) accuracyTST
    end
    return model
end


function setValidationSet(samples::Int64)
    validation = Dict{Int64, Set}();
    for (s, t) in zip(1:samples, 1:samples)
        push!(validation, s => Set(t))
    end
    return validation
end

function tokenizeFile(path::String="/home/PhD/Downloads/europarl-v7.es-en.en")
    mt = moses.MosesTokenizer("en")
    lines = readlines(path)
    sentences = String[]
    for (i, sentence) in collect(enumerate(lines))
        push!(sentences, mt.tokenize(sentence, return_str=true) |> lowercase)
    end
    return sentences
end

#=
# let's shuffle sentences
rng = MersenneTwister(1234);
idx = randperm(rng, length(sentences))

ft_sembs   = loadFT(sentences[idx])
bert_sembs = loadsBert(sentences[idx])

rft_sembs = W * ft_sembs

@info "fine tuning"
d = size(bert_sembs, 1)
model = Chain(Dense(d, d, relu6)) |> gpu
loss(x, y) = Flux.Losses.mse(model(x), y);

trn = 1:Int(8e3)
tst = 1+Int(8e3):length(sentences)

trnX = rft_sembs[:, trn]
trnY = bert_sembs[:, trn]

tstX = rft_sembs[:, tst]
tstY = bert_sembs[:, tst]

Xtrn, Ytrn, Xtst, Ytst = map(gpu, [trnX, trnY, tstX, tstY])

validationTRN = setValidationSet(length(trn))
validationTST = setValidationSet(length(tst))

accuracyTRN, _ = validate(Xtrn, Ytrn, validationTRN)
accuracyTST, _ = validate(Xtst, Ytst, validationTST)


model = fine_tune(Xtrn, Ytrn, Xtst, Ytst, loss, model; epochs=3) |> cpu

=#



function loss_and_accuracy(data_loader, model)
   acc = 0
   ls = 0.0f0
   num = 0
   for (x, y) in data_loader
       x, y = gpu(x), gpu(y)
       ŷ = model(x)
       ls += logitcrossentropy(ŷ, y, agg=sum)
       acc += sum(Flux.onecold(ŷ) .== Flux.onecold(y)) ## Decode the output of the model
       num +=  size(x)[end]
   end
   return ls / num, acc / num
end

function fine_tune(train_in_sembs, train_out_sembs, test_in_sembs, test_out_sembs)
    d = size(train_in_sembs, 1)
    model = Chain(Dense(d, d, relu6)) |> gpu
    loss(x, y) = Flux.Losses.mse(model(x), y);
    opt = ADAM(0.005)
    validationTRN = setValidationSet(size(train_in_sembs, 2))
    validationTST = setValidationSet(size(test_in_sembs,  2))


    D = repeated((train_in_sembs |> cu, train_out_sembs |> cu), 100)

    for epoch in 1:5
        Flux.train!(loss, Flux.params(model), D, opt)
        # model_cpu = model |> cpu
        #accuracyTRN, _ = validate(model(train_in_sembs |> cu) |> normalizeEmbedding , train_out_sembs |> cu |> normalizeEmbedding , validationTRN);
        #accuracyTST, _ = validate(model(test_in_sembs) |> cu |> normalizeEmbedding, test_out_sembs |> cu |> normalizeEmbedding, validationTST);
        #@printf "Epoch: %1i loss(train): %.4f, accuracy(train): %.4f, loss(test): %.4f, accuracy(test): %.4f \n" epoch loss(train_in_sembs, train_out_sembs) accuracyTRN loss(test_in_sembs, test_out_sembs) accuracyTST
    end
    return model
end

struct TuneData
    ft::Bool
    input::Matrix{Float32}
end

function getdata(data, labels; finetune::TuneData)
    rng = MersenneTwister(1234);
    idx = randperm(rng, size(data, 2))
    n = length(idx)
    tst = div(n, 5)
    trn = n - tst

    xtrain = data[:, idx[1:trn]]
    xtest  = data[:, idx[end-tst:end]]

    if finetune.ft # fine tuning true then
        @info "Fine tuning the rotation matrix W"
        ytrain = finetune.input[:, idx[1:trn]] |> cu
        ytest  = finetune.input[:, idx[end-tst:end]] |> cu
        model = fine_tune(xtrain, ytrain, xtest, ytest)
        xtrain = model(xtrain |> cu) |> cpu
        xtest  = model(xtest  |> cu) |> cpu
    end

    ytrain = Flux.onehotbatch(labels[idx[1:trn]], unique(labels))
    ytest = Flux.onehotbatch(labels[idx[end-tst:end]], unique(labels))

    train_loader = DataLoader((xtrain, ytrain), batchsize=256, shuffle=true)
    test_loader = DataLoader((xtest, ytest), batchsize=256)

    return train_loader, test_loader
end

function kfold(sembs, labels; k::Int64=5, epochs::Int64=5, ft2=TuneData(false, zeros(10, 10)))
    μtest = []
    for i in 1:k
        printstyled("Fold number: ", i , "\n", color=:red)
        traindata, testdata = getdata(sembs |> normalizeEmbedding, labels; finetune=ft2)
        d = size(sembs, 1)
        classifier = Chain(Dense(d, 100, relu),
        # Dense(100, 10, relu),
        Dense(100, length(unique(labels)))) |> gpu


        ps = Flux.params(classifier)

        opt = ADAM(0.005)

        @info "started training"
        acctest  = []
        for epoch in 1:epochs
            for (x, y) in traindata
                x, y = gpu(x), gpu(y) ## transfer data to device
                gs = gradient(() -> logitcrossentropy(classifier(x), y), ps) ## compute gradient
                Flux.Optimise.update!(opt, ps, gs) ## update parameters
            end

            ## Report on train and test
            train_loss, train_acc = loss_and_accuracy(traindata, classifier)
            test_loss,  test_acc  = loss_and_accuracy(testdata, classifier)
            @printf "Epoch : %i \n" epoch
            @printf "train loss : %.6f, train accuracy : %.6f \n" train_loss train_acc
            @printf "test loss  : %.6f, test accuracy  : %.6f \n" test_loss test_acc
            if epoch == epochs
                push!(acctest, test_acc)
            end
        end
        push!(μtest, acctest)
    end
    @printf "Mean test score: %.6f" mean(Iterators.flatten(μtest) |> collect)
end



function kfold2(train_sembs, test_sembs, train_labels, test_labels; k::Int64=5, epochs::Int64=5, ft2=TuneData2(false, zeros(10, 10), zeros(10,10)))
    μtest = []
    labels = vcat(train_labels, test_labels)
    for i in 1:k
        printstyled("Fold number: ", i , "\n", color=:red)

        ytrain = Flux.onehotbatch(train_labels, unique(labels))
        ytest = Flux.onehotbatch(test_labels, unique(labels))

        traindata = DataLoader((train_sembs, ytrain), batchsize=256, shuffle=true)
        testdata  = DataLoader((test_sembs, ytest), batchsize=256)



        d = size(train_sembs, 1)
        classifier = Chain(Dense(d, 100, relu),
                           Dense(100, 10, relu),
                           Dense(10, length(unique(labels)))) |> gpu

        ps = Flux.params(classifier)

        opt = ADAM(0.005)

        @info "started training"
        acctest  = []
        for epoch in 1:epochs
            for (x, y) in traindata
                x, y = gpu(x), gpu(y) ## transfer data to device
                gs = gradient(() -> logitcrossentropy(classifier(x), y), ps) ## compute gradient
                Flux.Optimise.update!(opt, ps, gs) ## update parameters
            end

            ## Report on train and test
            train_loss, train_acc = loss_and_accuracy(traindata, classifier)
            test_loss,  test_acc  = loss_and_accuracy(testdata, classifier)
            @printf "Epoch : %i \n" epoch
            @printf "train loss : %.6f, train accuracy : %.6f \n" train_loss train_acc
            @printf "test loss  : %.6f, test accuracy  : %.6f \n" test_loss test_acc
            if epoch == epochs
                push!(acctest, test_acc)
            end
        end
        push!(μtest, acctest)
    end
    @printf "Mean test score: %.6f" mean(Iterators.flatten(μtest) |> collect)
end





# load pretrained model with WMT
@load "./data/flux_wmt_models.bson" models
W = models[1]
model = models[2]


dataFolder = "../SentEval/data/downstream/"
_, folders, _ = first(walkdir(dataFolder))

root, _, files = first(walkdir(dataFolder * folders[5]))

# train_sents = tokenizeFile(root * "/" * files[2])
# test_sents = tokenizeFile(root * "/" * files[1])

train_sents = readlines(root * "/" * files[2])
test_sents = readlines(root * "/" * files[1])

#=
train_labels = collect(split(train_sents[i], ":")[1] for i in 1:length(train_sents))
test_labels = collect(split(test_sents[i], ":")[1] for i in 1:length(test_sents))

train_sents = collect(join(split(train_sents[i])[2:end], " ") for i in 1:length(train_sents))
test_sents = collect(join(split(test_sents[i])[2:end], " ") for i in 1:length(test_sents))
=#


train_lables = collect(split(train_sents[i])[1] for i in 1:length(train_sents))
test_lables  = collect(split(test_sents[i])[1] for i in 1:length(test_sents))

train_sents = collect(join(split(train_sents[i])[4:end], " ") for i in 1:length(train_sents))
test_sents = collect(join(split(test_sents[i])[4:end], " ") for i in 1:length(test_sents))


train_sents = train_sents[2:end]
test_sents  = test_sents[2:end]

train_sents = train_sents .|> lowercase
test_sents  = test_sents  .|> lowercase

train_labels = train_lables[2:end]
test_labels = test_lables[2:end]


trues  = CSV.read("../DATA/classification/archive-2/True.csv", DataFrame)
falss = CSV.read("../DATA/classification/archive-2/Fake.csv", DataFrame)

sentences = vcat(trues.text, falss.text)
tlabels = repeat(["trues"], outer=length(trues.subject))
flabels = repeat(["falses"], outer=length(falss.subject))
labels  = vcat(tlabels, flabels)

lines = getTrainigData(sentences)
ft_sembs = loadFT(lines)
bert_sembs = loadsBert(lines)


#=
sentences  = map(readlines, root .* "/" .* files)
negLabels = repeat(["neg"], outer=length(sentences[1]))
posLabels = repeat(["pos"], outer=length(sentences[2]))
labels = vcat(negLabels, posLabels)
sentences = sentences |> Iterators.flatten |> collect
=#

struct TuneData2
    ft::Bool
    train::Matrix{Float32}
    test::Matrix{Float32}
end

@info "Classifier"
ft_sembs = loadFT(sentences)
bert_sembs = loadsBert(sentences)

ft1 = TuneData(false, bert_sembs)
ft2 = TuneData(true, bert_sembs)
kfold(bert_sembs |> normalizeEmbedding, labels; ft2=ft1)

#=
train_sents1 = train_sents[1:65]
train_labels1 = train_labels[1:65]

train_sents  = vcat(train_sents1, train_sents[67:end])
train_labels = vcat(train_labels1, train_labels[67:end])
=#

train_sembs = loadFT(train_sents)
test_sembs  = loadFT(test_sents)

kfold2(W * train_sembs |> normalizeEmbedding, W * test_sembs |> normalizeEmbedding, train_labels, test_labels)


#let's fine tune the TREC
ft_train_sembs = loadFT(train_sents)
ft_test_sembs = loadFT(test_sents)

bert_train_sembs = loadsBert(train_sents)
bert_test_sembs = loadsBert(test_sents)
kfold2(bert_train_sembs,  bert_test_sembs, train_labels, test_labels)

model = fine_tune(W * ft_train_sembs |> normalizeEmbedding, bert_train_sembs, W * ft_test_sembs |> normalizeEmbedding, ft_test_sembs)
kfold2(W * ft_train_sembs |> cu |> model |> normalizeEmbedding,  W * ft_test_sembs |> cu |> model |> normalizeEmbedding, train_labels, test_labels)
