cd(@__DIR__)

using OhMyREPL
using LinearAlgebra
using Statistics
using Printf
using XLEs
using Distances
using PGFPlotsX


dcov(sub::Matrix) = sub' * sub;

src, trg, valfile = EmbeddingData() |> readData;

srcV, X = map(i-> src[i], 1:2);

X = X |> normalizeEmbedding;

rng = 1:Int(1e2);
subx = X[:, rng];

dcovx = dcov(subx);
ccovx = pairwise(CosineDist(), subx);

# for all words !

distanceCov = collect(sortperm(dcovx[:, i]) for i in rng);
distanceCos = collect(sortperm(ccovx[:, i]) |> reverse for i in rng);

distanceCov = reduce(hcat, distanceCov);
distanceCos = reduce(hcat, distanceCos);


@printf "DistCovariance vs Cosine Distance: %.4f \n" mean(distanceCos .== distanceCov)



dist = Mahalanobis(subx * subx');
mahx = pairwise(dist, subx);
distanceMah = collect(sortperm(mahx[:, i]) |> reverse for i in rng);
distanceMah = reduce(hcat, distanceMah);





@printf "Mahalanobis vs Cosine Distance: %.4f \n" mean(distanceMah .== distanceCov)


wordidx = 2500;

@time distMah = XLEs.mahalanobisGPU(subx |> cu, subx |> cu) |> Array;
distMah = collect(sortperm(distMah[:, i]) |> reverse for i in rng);
distMah = reduce(hcat, distMah);

simMah = XLEs.mahalanobis(subx, subx, ssize=length(rng))
simMah = collect(sortperm(simMah[:, i]) |> reverse for i in rng);
simMah = reduce(hcat, simMah)

#word_gpu = sortperm(distmah[:, wordidx]) |> reverse |> Array;

#mean(srcV[word_gpu][end-10:end] .== srcV[distanceMah[:, wordidx]][end-10:end])

