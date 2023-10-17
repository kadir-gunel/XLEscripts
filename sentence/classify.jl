cd(@__DIR__)
using OhMyREPL
using CSV
using DataFrames
using DelimitedFiles
using LIBSVM
using Random
using Statistics
using PyCall
using BSON: @load
using NPZ
using MKL
using LinearAlgebra

@pyimport sentence_transformers.SentenceTransformer as st
@pyimport sacremoses as moses
@pyimport fasttext as ft

mt = moses.MosesTokenizer("en")
# model = st.SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
# model = st.SentenceTransformer("all-mpnet-base-v2")

# imdb = CSV.read("../DATA/classification/IMDB Dataset.csv", DataFrame)
# sentences = imdb.review
# labels    = imdb.sentiment

trues  = CSV.read("../DATA/classification/archive-2/True.csv", DataFrame)
falses = CSV.read("../DATA/classification/archive-2/True.csv", DataFrame)

sentences = vcat(trues.text, falses.text)
tlabels = repeat(["trues"], outer=length(trues.subject))
flabels = repeat(["falses"], outer=length(falses.subject))
labels  = vcat(tlabels, flabels)


#woman = CSV.read("../DATA/classification/Womens Clothing E-Commerce Reviews.csv", DataFrame; header=["hello"])
#sentences = woman.Column5[2:end]
#labels = woman.Column11[2:end]

reviews = String[]
for sentence in sentences
     push!(reviews, mt.tokenize(sentence, return_str=true) |> lowercase)
end


# embeddings = model.encode(reviews)

EN = ft.load_model("/home/PhD/github/DATA/FT/cc.en.300.bin")
ft_embeddings = []
for sentence in reviews
   push!(ft_embeddings, EN.get_sentence_vector(sentence))
end

ft_embeddings = reduce(hcat, ft_embeddings)
@load "./data/W_wmt.bson" W
rft_embeddings = W * ft_embeddings


idx = randperm(size(ft_embeddings, 2))

Xtrain = ft_embeddings[:, idx[1:Int(40e3)]]
Ytrain = labels[idx[1:Int(40e3)]]
Xtest = ft_embeddings[:, idx[1+Int(40e3):end]]
Ytest = labels[idx[1+Int(40e3):end]]

svm_model = svmtrain(Xtrain, Ytrain)
ŷ, decision_values = svmpredict(svm_model, Xtest);


mean(Ytest .== ŷ)


Xtrain = rft_embeddings[:, idx[1:Int(40e3)]]
Ytrain = labels[idx[1:Int(40e3)]]
Xtest = rft_embeddings[:, idx[1+Int(40e3):end]]
Ytest = labels[idx[1+Int(40e3):end]]

rot_svm_model = svmtrain(Xtrain, Ytrain)
ŷ, decision_values = svmpredict(rot_svm_model, Xtest);

mean(Ytest .== ŷ)


#loading bert model
bert_embeddings = NPZ.npzread("./data/wmt_bert_all.npy")
B = svd(bert_embeddings)
Fr = svd(rft_embeddings)
# now lets exchange singular values
rft_embeddings = Fr.U * diagm(B.S) * Fr.Vt;



# for BERT
Xtrain = rft_embeddings[:, idx[1:Int(40e3)]]
Ytrain = labels[idx[1:Int(40e3)]]
Xtest = rft_embeddings[:, idx[1+Int(40e3):end]]
Ytest = labels[idx[1+Int(40e3):end]]

svm_model_bert = svmtrain(Xtrain, Ytrain)
ŷ, decision_values = svmpredict(svm_model_bert, Xtest);

#84.9%
mean(Ytest .== ŷ)
#77.8
mean(Ytest .== ŷ)
