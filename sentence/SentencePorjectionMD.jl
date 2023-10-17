using Pkg
Pkg.activate("XLEScripts")

using OhMyREPL

using Random
using Base.Iterators
using LinearAlgebra
using Statistics: mean
using Printf
using CUDA

CUDA.allowscalar(false)
Random.seed!(1234)

using Flux
using Flux: onehotbatch
using Flux.Losses
using Flux: params
using Flux.Data: DataLoader

using BSON: @save, @load

using SentProjections
using XLEs: readBinaryEmbeddings, normalizeEmbedding
using XLEs




path = "./notebooks/models/WMT/lc-tok/" 
src = path * "FT_WMT_ALL"
trg = path * "sBERT_768_WMT_ALL"


S = map(readBinaryEmbeddings, [src, trg])

_, F = S[1]
_, B = S[2]

F, B = map(permutedims, [F, B])
F, B = map(XLEs.unit, [F, B])

lines = readlines("/home/phd/Documents/europarl-en.lower.txt");
line_lens = lines .|> split .|> length; # 1965734
linesGTWords = findall(x -> x > 10, line_lens);
lines = lines[linesGTWords];


idx = pickSentences(length(lines), k=Int(120e3));
rng = 1:Int(100e3)
rngval = length(rng)+1:Int(120e3)

FData = F[:, idx[rng]] |> gpu
BData = B[:, idx[rng]] |> gpu

FvalData = F[:, idx[rngval]] |> gpu
BvalData = B[:, idx[rngval]] |> gpu

train = DataLoader((FData, BData), batchsize=128, shuffle=true)
valid = DataLoader((FvalData, BvalData), batchsize=128, shuffle=true)

p_norm(M::T; dim=2) where {T} = sqrt.(sum(real(M .* conj(M)), dims=dim))
cosine(X::T, Y::T) where {T} = diag((X ./ p_norm(X)) * (Y ./ p_norm(Y))')


function MMD(x::T, y::T; kernel::String="multiscale") where T
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




in_dim = size(F, 1)
out_dim = size(B, 1)


model = Chain(Dense(in_dim, 768),
              Dense(768, 100),
              Dense(100, 768),
              Dense(768, out_dim)) |> gpu

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
        push!(train_loss, mmd_loss(model)(n, t))
        push!(train_cos, cosine_loss(model)(n, t))
    end

    @printf "Epoch : %i, Train Loss: %.4f, Cos Loss %.4f : \t" epoch mean(train_loss) mean(train_cos)

    Flux.testmode!(model)
    for (n, t) in valid
        push!(valid_loss, mmd_loss(model)(n, t))
        push!(valid_cos,  cosine_loss(model)(n , t))
    end

    @printf "Test Loss: %.4f, Cosine Loss: %.4f \n" mean(valid_loss) mean(valid_cos)
end



model = model |> cpu
@save "./SentencePorjectionMD_wo_normalization.bson" model


valid_cos = []
valid_loss = []
Flux.testmode!(model)
for (x, y) in valid
    push!(valid_loss, mmd_loss(model)(x, y))
    push!(valid_cos, cosine_loss(model)(x, y))
end

