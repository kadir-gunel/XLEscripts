cd(@__DIR__)
using OhMyREPL
using Random
using Base.Iterators
using Base.Threads
using Printf
using Statistics
using LinearAlgebra
using MKL

using CSV
using DataFrames
using LIBSVM

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

# @pyimport sacremoses as moses
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
    #=
    accuracyTRN, _ = validate(Ŷtrn |> Array, Ytrn |> Array, validationTRN);
    accuracyTST, _ = validate(Ŷtst |> Array, Ytst |> Array, validationTST);
    =#
    # @printf "Epoch: %1i loss(train): %.4f, accuracy(train): %.4f, loss(test): %.4f, accuracy(test): %.4f \n" 0 loss(Ŷtrn, Ytrn) accuracyTRN loss(Ŷtst, Ytst) accuracyTST
    D = repeated((Ŷtrn, Ytrn), 100)
    for epoch in 1:epochs
        Flux.train!(loss, Flux.params(model), D, opt)
        model_cpu = model |> cpu
        accuracyTRN, _ = validate(model_cpu(Ŷtrn |> Array |> normalizeEmbedding) , Ytrn |> normalizeEmbedding|> Array, validationTRN);
        accuracyTST, _ = validate(model(Ŷtst) |> normalizeEmbedding, Ytst |> normalizeEmbedding, validationTST);
        @printf "Epoch: %1i loss(train): %.4f, accuracy(train): %.4f, loss(test): %.4f, accuracy(test): %.4f \n" epoch loss(Ŷtrn, Ytrn) accuracyTRN loss(Ŷtst, Ytst) accuracyTST
    end
    return model
end


function fine_tune2(Ŷtrn, Ytrn, Ŷtst, Ytst, loss, model; opt=ADAM(4e-4), epochs::Int64=10, validationTRN=validationTRN, validationTST=validationTST)
    data = (Ŷtrn, Ytrn)
    train_loader = DataLoader((data[1], data[2]), batchsize=1024, shuffle=false)
    for epoch in 1:epochs
        Flux.train!(loss, Flux.params(model), train_loader, opt)
        model_cpu = model |> cpu   
        accuracyTRN, _ = validate(model_cpu(Ŷtrn |> Array |> normalizeEmbedding) , Ytrn |> normalizeEmbedding|> Array, validationTRN);
        accuracyTST, _ = validate(model(Ŷtst) |> Array |> normalizeEmbedding, Ytst |> Array |> normalizeEmbedding, validationTST);
        @printf "Epoch: %1i loss(train): %.4f, accuracy(train): %.4f, loss(test): %.4f, accuracy(test): %.4f \n" epoch loss(Ŷtrn, Ytrn) accuracyTRN loss(Ŷtst, Ytst) accuracyTST   
        end 
    return model
end



#=
function getTrainigData(lines)
    mt = moses.MosesTokenizer("en")
    sentences = String[]
    for sentence in lines
        push!(sentences, mt.tokenize(sentence, return_str=true) |> lowercase)
    end
    return sentences
end
=#

function getFTVectors(model, sentences::Vector{String})
    ft_embeddings = []
    for (i, sentence) in collect(enumerate(sentences))
        push!(ft_embeddings, model.get_sentence_vector(sentence))
    end

     return reduce(hcat, ft_embeddings) |> normalizeEmbedding
end


loadEmbeddings(path::String) = npzread(path) |> normalizeEmbedding;


function tokenizeFile(path::String="/home/PhD/Downloads/europarl-v7.es-en.en")
    mt = moses.MosesTokenizer("en")
    lines = readlines(path)
    sentences = String[]
    for (i, sentence) in collect(enumerate(lines))
        push!(sentences, mt.tokenize(sentence, return_str=true) |> lowercase)
    end
    return sentences
end

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
    @time bert_model = st.SentenceTransformer("/home/PhD/github/DATA/all-mpnet-base-v2", device="cuda:0")
    @info "Processing Sentences (this might take time)"
    @time bert_embeddings = bert_model.encode(sentences, device="cuda:0", batch_size=256)
    @info "Normalizing Embeddings"
    return bert_embeddings |> permutedims |> normalizeEmbedding
end

# TrainSentences = tokenizeFile()
# ftSentences = loadFT(TrainSentences)
# npzwrite("./data/ft_wmt_all.npy", ftSentences |> permutedims)
# @time sBertSentences = loadsBert(TrainSentences)
# npzwrite("./data/sbert_wmt_all.npy", sBertSentences |> permutedims)

# @info "Loading calculated sentence embeddings"
# ftSentences = npzread("./data/ft_wmt_all.npy")
# sBertSentences = npzread("./data/sbert_wmt_all.npy")
lines = readlines("/home/PhD/europarl-en.txt")
line_lens = lines .|> split .|> length # 1965734
# lets consider lines that are greater than 4 
longer_lines = findall(x -> x > 10, line_lens)
lines = lines[longer_lines]

#now let's convert each line to sBERT vector!
sBertV = loadsBert(lines)

# now let's convert each line to FT vector!
ftV = loadFT(lines)

ftSentences    = npzread("./data/ftV_wmt.npy") |> permutedims
sBertSentences = npzread("./data/sBertV_wmt.npy") |> permutedims

trn = 1:Int(1.9e6)
tst  = 1+Int(1.9e6):size(ftSentences, 2)

W, s = mapOrthogonalSentences(@view(ftSentences[:, trn]), @view(sBertSentences[:, trn]))

# let's do a quick test
@printf "MSE error on train set: %.6f" mse(W * @view(ftSentences[:, trn]), @view(sBertSentences[:, trn]))
@printf "MSE error on test set : %.6f" mse(W * @view(ftSentences[:, tst]), @view(sBertSentences[:, tst]))


covBert = sBertSentences[:, trn] * sBertSentences[:, trn]'
covFT = ftSentences[:, trn] * ftSentences[:, trn]'

sbert1 = W * ftSentences[:, tst]

W2 = covBert * W
sbert2 = W2 * ftSentences[:, tst]

#let's truncate bert embeddings
B = svd(covBert)
B_trunc = B.U[:, 1:300] * diagm(sqrt.(B.S[1:300])) * B.Vt[1:300, 1:300]
B_trunc |> svdvals
W3 =  B_trunc * covFT
sbert3 = W3 * ftSentences[:, tst]

W4 = W * covFT
sbert4 = W4 * ftSentences[:, tst]


W5 = covBert * W * covFT
sbert5 = W5 * ftSentences[:, tst]

mse1 = mse(sbert1, @view(sBertSentences[:, tst]))
mse2 = mse(sbert2, @view(sBertSentences[:, tst]))
mse3 = mse(sbert3, @view(sBertSentences[:, tst]))
mse4 = mse(sbert4, @view(sBertSentences[:, tst]))
mse5 = mse(sbert5, @view(sBertSentences[:, tst]))


#======================================= DATA FOR FINE_TUNE ===============================================#
# let's first tune the rotation matrix W by using the same training samples from WMT (random 50k sentences)
#==========================================================================================================#

n = 1:Int(60e3)
trainX = W * ftSentences[:, n]
trainY = sBertSentences[:, n]
testX = W * ftSentences[:, tst]
testY = sBertSentences[:, tst]

validationTRN = setValidationSet(length(n))
validationTST = setValidationSet(length(tst))

accuracyTRN, _ = validate(trainX, trainY, validationTRN)
accuracyTST, _ = validate(testX, testY, validationTST)

#====================================FINE_TUNE===================================#
# d = size(trainX, 1)
d = 768
model = Chain(Dense(d, d, elu)) |> gpu
# model = Chain(Dense(W2, true, relu6)) |> gpu
loss(x, y) = Flux.Losses.mse(model(x), y);
loss2(x, y) = Flux.Losses.siamese_contrastive_loss(model(x), y; margin=.5);
Xtrn, Ytrn, Xtst, Ytst = map(cu, [trainX, trainY, testX, testY])
model = fine_tune2(Xtrn, Ytrn, Xtst, Ytst, loss, model; epochs=10, opt=ADAM(7e-4))

models = [W, model |> cpu]

# @save "./data/flux_wmt_models.bson" models
# save both rotation matrix and model parameters
# Wmodel = model.layers[1].weight |> Array |> permutedims
# Bmodel = model.layers[1].bias |> Array |> permutedims
# npzwrite("./data/model2/Wmodel.npy", Wmodel)
# npzwrite("./data/model2/Bmodel.npy", Bmodel)

@load "./data/WTM_model_NN/flux_wmt_models.bson" models
W = models[1]
model = models[2]
#=========================================================================#


# load trainig data
####### imdb data set
imdb = CSV.read("../DATA/classification/IMDB Dataset.csv", DataFrame)
sentences = imdb.review
labels    = imdb.sentiment

# lines = getTrainigData(sentences)
# ft_sembs = loadFT(lines)
# bert_sembs = loadsBert(lines)

@load "./data/imdb_ft.bson" ft_sembs
@load "./data/imdb_bert.bson" bert_sembs


###### load trues-and-falses dataset
#=
trues  = CSV.read("../DATA/classification/archive-2/True.csv", DataFrame)
falses = CSV.read("../DATA/classification/archive-2/Fake.csv", DataFrame)

sentences = vcat(trues.text, falses.text)
tlabels = repeat(["trues"], outer=length(trues.subject))
flabels = repeat(["falses"], outer=length(falses.subject))
labels  = vcat(tlabels, flabels)

lines = getTrainigData(sentences)
ft_sembs = loadFT(lines)
bert_sembs = loadsBert(lines)
=#

#=
@save "./data/true_false_ft_.bson" ft_sembs
@save "./data/true_false_bert.bson" bert_sembs

@load "./data/true_false_ft_.bson" ft_sembs
@load "./data/true_false_bert.bson" bert_sembs
=#

ft_sembs, bert_sembs = map(normalizeEmbedding, [ft_sembs, bert_sembs])

rng = MersenneTwister(1234);
idx = randperm(rng, size(ft_sembs, 2))

trn = idx[1:Int(40e3)]
tst = idx[1+Int(40e3):size(ft_sembs, 2)]

# we have 4 models : FT, sBERT, rotFT, tunedAndRotatedFT.
# for each we will check the svm accuracy
# then, finally, we will fine-tune the rotFT sentence embeddings and check the svm
trnX = W * @view(ft_sembs[:, trn]) |> normalizeEmbedding
tstX = W * @view(ft_sembs[:, tst]) |> normalizeEmbedding

#trnX = (W' * bert_sembs[:, trn]) |> normalizeEmbedding
#tstX = (W' * bert_sembs[:, tst]) |> normalizeEmbedding

trnY = labels[trn]
tstY = labels[tst]

svm_model = svmtrain(trnX, trnY)
ŷ, decision_values = svmpredict(svm_model, tstX);

# trft normalized
@printf "Accuracy score for sBERT %.4f" mean(tstY .== ŷ)


@info "convert ft to sbert with imdb"
d = 768
# model = Chain(Dense(d, d, relu6)) |> gpu
model = Chain(Dense(W, true, relu6)) |> gpu
loss(x, y) = Flux.Losses.mse(model(x), y);
# loss2(x, y) = Flux.Losses.siamese_contrastive_loss(model(x), y; margin=.5);

trnX = @view(ft_sembs[:, trn]) |> normalizeEmbedding |> gpu
tstX = @view(ft_sembs[:, tst]) |> normalizeEmbedding |> gpu

trnY = bert_sembs[:, trn] |> normalizeEmbedding |> gpu
tstY = bert_sembs[:, tst] |> normalizeEmbedding |> gpu

validationTRN = setValidationSet(length(trn))
validationTST = setValidationSet(length(tst))

accuracyTRN, _ = validate(trnX |> Array, trnY |> Array, validationTRN)
accuracyTST, _ = validate(tstX, tstY, validationTST)

Xtrn, Ytrn, Xtst, Ytst = map(gpu, [trnX, trnY, tstX, tstY])
model = fine_tune(Xtrn, Ytrn, Xtst, Ytst, loss, model; epochs=10)


mse(model(Xtrn), Ytrn)
validate(model(Xtst) |> Array, Ytst |> Array , validationTST)

Xtrn = W * @view(ft_sembs[:, trn]) |> gpu |> model |> normalizeEmbedding
# let's retrain svm with this fine-tuned embeddings
trnX = ft_sembs[:, trn] |> gpu |> model |> normalizeEmbedding |> Array
tstX = ft_sembs[:, tst] |> gpu |> model |> normalizeEmbedding |> Array

trnY = labels[trn]
tstY = labels[tst]

svm_model = svmtrain(trnX, trnY)
ŷ, decision_values = svmpredict(svm_model, tstX);

# trft normalized
@printf "Accuracy score for sBERT %.4f" mean(tstY .== ŷ)







# lets observe the singular values between original and rotated ft sentence embeds.
O = svdvals(ft_sembs)
R = svdvals(rft_sembs)
T = svdvals(trft_sembs)
B = svdvals(bert_sembs)

S = [R, O, T, B]




#=
woman = CSV.read("../DATA/classification/Womens Clothing E-Commerce Reviews.csv", DataFrame; header=["hello"])
sentences = woman.Column5[2:end]
labels = woman.Column11[2:end]
=#
#posLines = readlines("/home/PhD/github/SentEval/data/downstream/MR/rt-polarity.pos")
#negLines = readlines("/home/PhD/github/SentEval/data/downstream/MR/rt-polarity.neg")
#tlabels = repeat(["trues"], outer=length(posLines))
#flabels = repeat(["falses"], outer=length(negLines))
#labels  = vcat(tlabels, flabels)

#sentences = vcat(posLines, negLines)




rft_embeds = (W * ft_embeds) |> normalizeEmbedding
gpu_rft_embeds  = cu(rft_embeds)
trft_embeds = model(gpu_rft_embeds) |> normalizeEmbedding
# lets observe the singular values between original and rotated ft sentence embeds.
O = svdvals(ft_embeds)
R = svdvals(rft_embeds)
T = svdvals(trft_embeds |> Array)

S = [O, R, T]

legends = ["rotated", "original",  "fine-tuned", "bert"]

glog = @pgf Axis(
    {
        xlabel = "Singular Value size",
        ylabel = "Log ",
        ymode  = "log",
        legend_pos  = "south west",
    },);

g = @pgf Axis(
    {
        xlabel = "Singular Value size",
        ylabel = "Log ",
        legend_pos  = "south west",
    },);

@pgf for i in 1:length(S)
    push!(glog, Plot(Table(x=collect(1:length(S[i])), y=log2.(S[i]))), LegendEntry(legends[i]))
    push!(g,    Plot(Table(x=collect(1:length(S[i])), y=log2.(S[i]))), LegendEntry(legends[i]))
end

G = @pgf GroupPlot(
           {
               group_style = {group_size="2 by 1"},
           },
       glog, g);
display("image/png", G)

PGFPlotsX.save("./figs/imbd_all.pdf", G)

# since most singular values of the rotated ft_embeds are zero it is impossible to get different results from the originals.
# so one way is to tweak the rotation matrix by using sub-samples from the training data set.
# lets use wmt's subsamples for tweaking the W matrix.
# We will have pipe model that is composed of 2 models at the end:
#  -  1. in closed form
#  -  2. pipe the rotated ones to the NN model


# generate same random numbers for each experiment
rng = MersenneTwister(1234);
idx = randperm(rng, length(lines))

Xtrain = ft_embeds[:, idx[1:Int(20e3)]] |> Array
Ytrain = labels[idx[1:Int(20e3)]]
Xtest = ft_embeds[:, idx[1+Int(20e3):end]] |> Array
Ytest = labels[idx[1+Int(20e3):end]]

svm_model = svmtrain(Xtrain, Ytrain)
ŷ, decision_values = svmpredict(svm_model, Xtest);

# trft normalized
mean(Ytest .== ŷ)

# normalized ft
mean(Ytest .== ŷ)

#d = size(rft_embeds, 1)
#model = Chain(Dense(d, d, relu6)) |> gpu
#loss(x, y) = Flux.Losses.mse(model(x), y);
# loss2(x, y) = Flux.Losses.siamese_contrastive_loss(model(x), y; margin=.5);
# model = fine_tune(Xtrn, Ytrn, Xtst, Ytst, loss, model; epochs=5)
