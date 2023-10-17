
using LinearAlgebra
using Random
using Statistics
using Printf
using Base.Iterators


using XLEs
using CUDA
using Flux 


CUDA.allowscalar(false)
Random.seed!(1234)


using Flux: onehotbatch
using Flux.Losses
using Flux: params
using Flux.Data: DataLoader

using BSON: @save, @load




path = "/run/media/phd/PhD/vecmap/data/embeddings/"

# lets first adapt the domains on the same language pairs
glove = path * "Glove"
en    = path * "en"


E = map(readBinaryEmbeddings, [glove, en])
srcG, G = E[1]
srcE, E = E[2]

G, E = map(permutedims, [G, E])
G, E = map(XLEs.unit, [G, E])

# lets intersect the vocabularies
commonV = intersect(srcG, srcE)

w2iG = Dict(word => idx for (idx, word) in enumerate(srcG))
w2iE = Dict(word => idx for (idx, word) in enumerate(srcE)) 

# get indices for Glove and Vecmap 
idxG = collect(w2iG[word] for word in commonV)
idxE = collect(w2iE[word] for word in commonV)

# lets split words as train and test(valid) data
idx = randperm(length(commonV)) # this will shuffle idx
# split data as .85/.15 ratio
trnsize = Int(floor(length(idx) * .85))
valsize = trnsize + 1: length(idx)
trainIdx = idx[1:trnsize]
validIdx = idx[valsize]


train = DataLoader((G[:, idxG[trainIdx]], E[:, idxE[trainIdx]]), shuffle=true, batchsize=512)
valid = DataLoader((G[:, idxG[validIdx]], E[:, idxE[validIdx]]), shuffle=true, batchsize=512)

p_norm(M::T; dim=2) where {T} = sqrt.(sum(real(M .* conj(M)), dims=dim))
cosine(X::T, Y::T) where {T} = diag((X ./ p_norm(X)) * (Y ./ p_norm(Y))')


function MMD(x::T, y::T; kernel::String="rbf") where T
    atype = typeof(x)
    xx = x' * x
    yy = y' * y 
    zz = x' * y

    
    #rx = repeat(diag(x), 1, size(x, 2))
    #ry = repeat(diag(y), 1, size(y, 2))
    
    
    rx, ry = map(x -> repeat(diag(x), 1, size(x, 2)), [xx, yy])

    dxx = rx + rx' - 2xx
    dyy = ry + ry' - 2yy
    dxy = rx + ry' - 2zz
   

    if isequal(string(atype), "CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}")
        XX, YY, XY = map(xx -> cu(zeros(size(xx))), [xx, xx, xx])
    else 
        XX, YY, XY = map(xx -> zeros(size(xx)), [xx, xx, xx])
    end

    XX, YY, XY = map(cu, [XX, YY, XY])

    if isequal(kernel, "multiscale")
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for b in bandwidth_range
            XX += b^2 * (b^2 .+ dxx).^(-1)
            YY += b^2 * (b^2 .+ dyy).^(-1)
            XY += b^2 * (b^2 .+ dxy).^(-1)
        end
    end

    if isequal(kernel, "rbf")
        bandwidth_range = [10, 15, 20, 50]
        for b in bandwidth_range

            XX += exp.(-.5 * dxx / b)
            YY += exp.(-.5 * dyy / b)
            XY += exp.(-.5 * dxy / b)
        end
    end

    mmd = mean(XX + YY - 2 .* XY) # this is very similar to CSLS!!
    return mmd
end

in_dim = 300
out_dim = 300

model = Chain(Dense(in_dim, 300),
              Dense(300, 150),
              Dense(150, 300),
              Dense(300, out_dim)) |> gpu

optim = Flux.RMSProp(0.001)



cosine_loss(model) = (x, y) -> 1 .- mean(abs.(cosine(model(x) |> permutedims, y |> permutedims)))
mmd_loss(model) = (x, y) -> MMD(model(x), y, kernel="rbf")

combined_loss(model) = (x, y) -> mmd_loss(model)(x, y) + cosine_loss(model)(x, y)

epochs = 40

for epoch in 1:epochs
    train_loss = []; valid_loss = []
    train_cos  = []; valid_cos  = []
    Flux.trainmode!(model)
    for (n, t) in train
        
        Flux.train!(mmd_loss(model), Flux.params(model), [(n |> gpu, t |> gpu)], optim)
        # Flux.train!(mmd_loss(model), Flux.params(model), [(n |> gpu, t |> gpu)], optim)
        # Flux.train!(cosine_loss(model), Flux.params(model),[(n |> gpu, t |> gpu)], optim)
        push!(train_loss, mmd_loss(model)(n |> gpu, t |> gpu))
        push!(train_cos, cosine_loss(model)(n |> gpu, t |> gpu))
    end

    @printf "Epoch : %i, Train Loss: %.4f, Cos Loss %.4f : \t" epoch mean(train_loss) mean(train_cos)

    Flux.testmode!(model)
    for (n, t) in valid
        push!(valid_loss, mmd_loss(model)(n |> gpu, t |> gpu))
        push!(valid_cos,  cosine_loss(model)(n |> gpu , t |> gpu))
    end

    @printf "Test Loss: %.4f, Cosine Loss: %.4f \n" mean(valid_loss) mean(valid_cos)
end


newG = model(G |> gpu) |> Array

@save "./aligned_Glove.bson" newG 

trgfile = path * "it"
# @time srcV, X = readBinaryEmbeddings(glove)
# @load "./aligned_Glove.bson" newG
@time trgV, Y = readBinaryEmbeddings(trgfile);

Y = Y |> permutedims
Y = Y |> XLEs.unit
# X, Y = map(XLEs.unit, [newG, Y])


X, Y = map(cu, [newG, Y])

val = "/run/media/phd/PhD/vecmap/data/dictionaries/en-it.test.txt"

src_w2i, trg_w2i = map(word2idx, [srcG, trgV]);
validation = readValidation(val, src_w2i, trg_w2i);

rng = 1:Int(4e3)
x = X[:, rng]  #|> Matrix
y = Y[:, rng]  #|> Matrix


# @time src_idx, trg_idx = XLEs.buildSeedDictionary0(subx, suby)
# @time src_idx, trg_idx = XLEs.buildRBFSeedDictionary(x, y, threshold=0.6, k=40);
@time src_idx, trg_idx = XLEs.buildSeedDictionary(x, y, sim_size=length(rng));
# @time src_idx3, trg_idx3 = XLEs.buildCosineDictionary(x |> Matrix, y |> Matrix)
# @time src_idx3, trg_idx3 = XLEs.buildMahalanobisDictionary(x |> Matrix, y |> Matrix)
# @time src_idx, trg_dix = XLEs.mahalanobisGPU(subx, suby);
@time W, src_idx, trg_idx = XLEs.main(X, Y, src_idx, trg_idx, validation, src_size=Int(20e3), trg_size=Int(20e3));
