cd(@__DIR__)
using OhMyREPL
using LinearAlgebra
using Statistics
using StatsBase
using Base.Iterators
using Base.GC
using XLEs
using Knet
using Knet: update!
"""
using Flux
using Flux.Losses
using Flux.Data: DataLoader
using Flux.Optimise: update!, Descent, ADAM, RADAM
"""
using Printf
using CUDA
using MKL



function tokenizeFile(path::String="/home/PhD/Downloads/europarl-v7.es-en.en")
    mt = moses.MosesTokenizer("en")
    lines = readlines(path)
    sentences = String[]
    for (i, sentence) in collect(enumerate(lines))
        push!(sentences, mt.tokenize(sentence, return_str=true) |> lowercase)
    end
    return sentences
end


"""
convert corpus to integer valued sentences
"""
function crp2int(corpus::Array, s2i::Dict)
    intcrp = Array{Int64,1}[]
    for line in corpus
        aux = Int64[]
        for word in split(line)
            haskey(s2i, word) ? push!(aux, s2i[word]) : push!(aux, s2i["<UNK>"])
        end
        push!(intcrp, aux)
    end
    return intcrp
end


function padding(corpus::Array, s2i::Dict; maxSeqLen::Int64=70)
    maxlen = maximum(length.(corpus))
    idx = findall(i -> length(i) < maxSeqLen , corpus)
    for sent in idx
        diff = maxlen - length(corpus[sent])
        for i in 1:diff
            insert!(corpus[sent], 1, s2i["<PAD>"])
        end
    end
end


function createBatches(input::Array, output::Array; BSize::Int=64)
    # for input
    long = maximum(length.(input))
    input = collect(partition(partition(flatten(input), long), BSize))
    input = reduce.(hcat, input)

    # for output
    r, samples = size(output)
    output = collect(partition(partition(flatten(output), r), BSize))
    output = reduce.(hcat, output)
    output = map(permutedims, output)
    return (input[1:end-1],output[1:end-1])
end


"""
lookup table creation according to minimum frequency limit
"""
function createFreqs!(lines::Vector{String}, fDict::Dict{String, Int64}; limit::Int64=5,)
    words = lines .|> split |> flatten .|> String 
    for word in words
        fDict[word] = get(fDict, word, 0) + 1
    end
    return filter!(w -> w.second > limit, fDict)
end


initstate(bsize, hidden) = atype(zeros(Float32, bsize, hidden))
init(d...) = atype(xavier(Float32, d...))

function initModel2(hidden, vocsize)
    model = Dict{Symbol, Any}();
    params = (:hidden, :z, :r)
    for param in params
        model[Symbol(:W_, param)] = init(2hidden, hidden)
        model[Symbol(:b_, param)] = init(1, hidden)
    end
    model[:embeds] = init(vocsize, 300)
    model[:dense] = init(300, hidden)
    return model
end

function initModel(hidden; embedding::Matrix{Float32}=F)
    model = Dict{Symbol, Any}();
    params = (:hidden, :z, :r)
    for param in params
        model[Symbol(:W_, param)] = init(2hidden, hidden)
        model[Symbol(:b_, param)] = init(1, hidden)
    end
    model[:embeds] = F |> permutedims |> KnetArray
    model[:rotation] = init(size(F, 1), hidden)
    return model
end

function outputModel(hidden, classSize)
    model = Dict{Symbol, Any}();
    model[:W] = init(hidden, classSize)
    model[:b] = init(1, classSize)
    return model
end


function gru(model, hidden, input, mask)
    x = hcat(hidden, input)
    z = sigm.(x * model[:W_z] .+ model[:b_z])
    r = sigm.(x * model[:W_r] .+ model[:b_r])
    x = hcat((r .* hidden), input)
    h = tanh.(x * model[:W_hidden] .+ model[:b_hidden])
    hidden =  ((1 .- z) .* hidden) .+ (z .* h)
    M = atype{Float32,2}(fill(1.0, size(input)))
    M = M .* atype{Float32}(mask)
    return hidden .* M # hence we clear all paddings states
end


predict(output, hidden) = logp(output[:W] * hidden  .+ output[:b], dims=2)

loss(ŷ, y) = (sum(abs2, (ŷ - y)) / size(ŷ, 1))

function encode(model, x, y)
    sumloss = 0
    encoder = model[:encoder]
    output = model[:output]
    tsteps = size(x, 1)
    hState = initstate(bsize, hidden)
    for t in 1:tsteps
        input = encoder[:embeds][x[t, :], :] * encoder[:dense]
        mask = .!(x[t, :] .== vocsize) # vocsize length is the padding number in dictionary
        hState = gru(encoder, hState, input, mask)
    end
    # ŷ = predict(output, hState)
    sumloss += loss(hState, y |> permutedims) # ŷ = hState
    return sumloss
end

gradients = gradloss(encode)

function train(model, x, y)
    # global atype = KnetArray
    examples = 0; totloss = 0.;
    opts = optimizers(model, optim, lr=lrate)
    j = 1
    for (i, o) in zip(x, y)
        grads, loss = gradients(model, i, o)
        update!(model, grads, opts)
        @printf "j : %i \n" j
        totloss += loss
        examples += size(o, 2)
        j += 1
    end
    totloss/examples
end


function validate(model, x, y)
    examples = 0; totloss = 0
    # global atype = Array
    # model = convertArrayTypes(model, Array)
    for (i,o) in zip(x,y)
        loss = encode(model, i, o)
        totloss += loss
        examples += size(o, 1)
    end
    #global atype = KnetArray
    #model = convertArrayTypes(model, KnetArray)
    return totloss/examples
end


function getindices(V::Vector{String}, ftV::Vector{String}) 
    idx = Any[]
    for word in V
        push!(idx, findfirst(x -> x == word, ftV))
    end
    filter!(x -> x != nothing, idx)
    return idx
end


function createBatches2(input::Array, output::Array; BSize::Int=64)

    all_input = []
    split_size = BSize
    total = input |> length |> i -> div(i, split_size) - 1 
    # excluding the last batch !!
    init = 1
    @info "Creating Batches of size $(BSize)"
    for i in 1:total
        push!(all_input, input[init:split_size])
        init += BSize
        split_size = split_size + BSize
    end

    @info "Dynamic Padding"
    for i in 1:total
        padding(all_input[i], s2i, maxSeqLen=maximum(length.(all_input[i])))
    end

    input = reduce.(hcat, all_input)


    # creating batches for the output
    @info "Creating Batches for the Target side"
    r, samples = size(output)
    output = collect(partition(partition(flatten(output), r), BSize))
    output = reduce.(hcat, output)
    return input, output[1:end-1]
end


# Reading text file 

lines = readlines("/home/PhD/europarl-en.txt");
line_lens = lines .|> split .|> length # 1965734
# lets consider lines that are greater than 4 
line_lens = findall(x -> x > 4, line_lens)
lines = lines[line_lens]

_, sBertV = readBinaryEmbeddings("/home/PhD/github/XLEScripts/data/vBert.bin")
sBertV = sBertV |> permutedims
sBertV[:, line_lens]

# vocabulary = lines .|> split |> i -> reduce(vcat, i) |> unique;

fDict = Dict{String, Int64}();
createFreqs!(lines, fDict; limit=5)

freqVoc = fDict |> keys |> collect
# vocabulary = intersect(vocabulary, freqVoc)


# load FT embedding model
fastText = "/home/PhD/.julia/datadeps/FastText Common Crawl/crawl-300d-2M.vec"
ftV, F = readEmbeddings(fastText);

V = intersect(vocabulary, ftV) .|> String;
@time idx = getindices(V, ftV);


# subF = F[:, idx |> sort] 

# s2i = Dict(V .=> idx);
# s2i = sort(s2i, byvalue=true)
# i2f = Dict(idx .=> (1:length(idx)))

vocabulary = sort(freqVoc)
s2i = Dict(term => i for (i, term) in enumerate(vocabulary))
s2i["<UNK>"] = get(s2i, "<UNK>", 0) + length(s2i) + 1 # adding UNK to the dictionary
s2i["<PAD>"] = get(s2i, "<PAD>", 0) + length(s2i) + 1 # adding PAD to the dictionary

# convert corpus to integer format
@time trn  = crp2int(lines, s2i);


sidx_trn = trn .|> length |> i -> sortperm(i, rev=true);

trn = trn[sidx_trn]
Y   = sBertV[:, sidx_trn]


trn, sBertV = createBatches2(trn, Y; BSize=64);

# lets shuffle batches for training and validation

using Random
rng = MersenneTwister(1234);
ids = shuffle(rng, collect(1:length(trn)))


trainSize = ids[1:Int(30e3)]
Xtrn, Ytrn = trn[trainSize], sBertV[trainSize];

testSize = ids[length(trainSize)+1:length(trn)]
Xtst, Ytst = trn[testSize], sBertV[testSize];


global bsize  = 64
global hidden = 768
global optim  = Adam
global lrate  = 4e-4
global epochs = 10
global vocsize = length(s2i)

idx = collect(values(s2i)) |> sort

model = Dict{Any, Any}();
model[:encoder] = initModel2(hidden, vocsize)
model[:output]  = outputModel(hidden, size(Ytrn[1], 1))


xs = findall(x -> length(x) < 40, Xtrn)
xtrn = Xtrn[xs]
ytrn = Ytrn[xs]
for epoch in 1:epochs
   local loss2 =  train(model, Xtrn, Ytrn .|> KnetArray)
    @printf "Epoch: %i, loss: %.9f \n" epoch loss2
end
