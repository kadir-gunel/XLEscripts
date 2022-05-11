using MKL
using XLEs
using LinearAlgebra
using RandomizedLinAlg
using Statistics
using BSON: @load 
using CUDA
using Printf

@load "./W.bson" W src_idx trg_idx

src, trg, valfile = EmbeddingData() |> readData;

srcV, X = map(i -> src[i] , 1:2);
trgV, Y = map(i -> trg[i] , 1:2);


src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
validation = readValidation(valfile, src_w2i, trg_w2i);



# W = W |> cu
WX, X, Y = map(normalizeEmbedding, [W * X, X, Y]);

acc, sim = validate(WX, Y, validation)
@printf "k-nn: %.4f , Similarity: %.4f" acc sim

first20k = 1:Int(20e3);
last20k  = Int(20e3)+1:Int(40e3);

groundSRC, predictedTRG = src_idx[first20k], trg_idx[first20k]; 
predictedSRC, groundTRG = src_idx[last20k] , trg_idx[last20k];


n = Int(20e3)
x  = WX[:, groundSRC[1:n]];
y  = Y[:, predictedTRG[1:n]];


srcPairs = pairs(IndexStyle(1:n), groundSRC);
trgPairs = pairs(IndexStyle(1:n), predictedTRG);

cosX = x' * x;
cosY = y' * y;

@time F = rsvd(cosX * cosY, 40, 4);
@time R = permutedims(cu(F.U) * cu(F.Vt));
@time cosRX = R * cu(cosX) * R';
@time revcosRX = permutedims(cosRX);
# now we will calculate the new k neighbors!

@time knnsim = XLEs.topk_mean(cosRX, 10, inplace=true); # en iyi k komsunun ortalama similarity degerini donuyor. 
bestsim = maximum(revcosRX, dims=2)  # objective score degerini donuyor. 
revcosRX = revcosRX .- (knnsim / 2)
idx = getindex.(argmax(revcosRX, dims=2), 2) |> Array |> vec
               


### we will rotate Y according to X
# both cosx and cosy matrices are symmetric; no need to take the transpose of any!
F = rsvd(Array(cosy * cosx), 30, 4)
W = F.V * F.U' |> cu
cosynew = (W * cu(cosy) * W') |> Array

idx = Matrix(undef, 10, Int(20e3))

for i in 1:10
    top_idx = findmax(sim, dims=1)
    idx[i, :] = getindex.(top_idx[2], 1) 
    sim[top_idx[2]] .= -10000
end



# 2 
x_dynamic, y_fix = trg_idx[x2y], src_idx[x2y]

cosx = X[:, x_dynamic]' * X[:, x_dynamic]
cosy = Y[:, y_fix]' * Y[:, y_fix]

R = rsvd(Array(cosx * cosy), 30, 4)
W = R.V * R.U' |> cu
cosynew = W * cosy * W'

