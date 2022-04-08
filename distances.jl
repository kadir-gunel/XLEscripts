cd(@__DIR__)

using OhMyREPL
using LinearAlgebra
using Statistics
using Printf
using XLEs
using Distances
using PGFPlotsX
using CUDA
using TransferEntropy
using BSON: @save, @load

@printf "Setting GPU Device %s :" CUDA.device!(1)

dcov(sub::Matrix) = sub' * sub;

src, trg, valfile = EmbeddingData() |> readData;

srcV, X = map(i-> src[i], 1:2);

X = X |> normalizeEmbedding;

rng = 1:Int(4e3);
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



dist  = Mahalanobis(subx * subx');
mahxs = pairwise(dist, subx, subx)
distancePairMahX = collect(sortperm(mahxs[:, i]) |> reverse for i in rng);
distancePairMahX = reduce(hcat, distancePairMahX);

@printf "Mahalanobis vs Pair Mahalanobis: %.4f \n" mean(distanceMah .== distancePairMahX)



@time distMah = XLEs.mahalanobisGPU(subx |> cu, subx |> cu) |> Array;
distMah = collect(sortperm(distMah[:, i]) |> reverse for i in rng);
distMah = reduce(hcat, distMah);

@time simMah = XLEs.parallelMahalanobis1(subx, subx, sim_size=length(rng))
simMah = collect(sortperm(simMah[:, i]) |> reverse for i in rng);
simMah = reduce(hcat, simMah)

@printf "Mahalanobis vs GPU Mahalanobis Distance: %.4f \n" mean(distanceMah .== distMah)
@printf "Mahalanobis vs Parallel Mahalanobis Distance: %.4f \n" mean(distanceMah .== simMah)
@printf "GPU vs Parallel Mahalanobis Distance: %.4f \n" mean(simMah .== distMah)

# just to show how words are related with different metrics
# the chosen word is politician; idx= 3999

idx = rand(rng, 4)
for id in idx
    @printf "Chosen word: %s, index in Vocabulary: %i \n" uppercase(srcV[id]) id
    cos = srcV[distanceCov[:, id]]
    mah = srcV[distanceMah[:, id]]
    gpuMah = srcV[distMah[:, id]]
    parallelMah = srcV[simMah[:, id]]

    @printf "Words chosen by Cosine :%s \n" cos[end-10:end] |> reverse;
    @printf "Words chosen by Mahalanobis :%s \n" mah[end-10:end] |> reverse;
    @printf "Words chosen by GPU Mahalanobis :%s \n" gpuMah[end-10:end] |> reverse;
    @printf "Words chosen by Parallel Mahalanobis :%s \n" parallelMah[end-10:end] |> reverse;

    @printf "------------------------------------------------------------------------ \n"
end

# lets observe how mutual information changes between words
x = Dataset(subx |> permutedims);

cosMI = [];
mahMI = [];
for id in idx
    coslist = distanceCov[:, id][end-10:end]
    mahlist = distanceMah[:, id][end-10:end]
    for (cidx, midx) in zip(coslist, mahlist)
        push!(cosMI, mutualinfo(x[id, :], x[cidx, :], Kraskov1(3)))
        push!(mahMI, mutualinfo(x[id, :], x[midx, :], Kraskov1(3)))
    end
end

cosMI = reduce(hcat, collect(Iterators.partition(cosMI, 11)))
mahMI = reduce(hcat, collect(Iterators.partition(mahMI, 11)))

# how about sorting words according to their mutual information ?

mi = Matrix{Float32}(undef, 4000, 4000);
@time @threads for i in axes(subx, 2)
    for j in axes(subx, 2)
        mi[i, j] = @views mutualinfo(subx[:, i], subx[:, j], Kraskov2(20))
    end
end

distMI = collect(sortperm(mi[:, i]) for i in rng);
distMI = reduce(hcat, distMI)

idx = rand(rng, 4)
for id in idx
    @printf "Chosen word: %s, index in Vocabulary: %i \n" uppercase(srcV[id]) id
    cos = srcV[distanceCov[:, id]]
    mah = srcV[distanceMah[:, id]]
    mutual = srcV[distMI[:, id]]

    @printf "Words chosen by Cosine :%s \n" cos[end-10:end] |> reverse;
    @printf "Words chosen by Mahalanobis :%s \n" mah[end-10:end] |> reverse;
    @printf "Words chosen by MI :%s \n" mutual[end-10:end] |> reverse;


    @printf "------------------------------------------------------------------------ \n"
end

distMI_3 = deepcopy(distMI_old)
distMI_K1_20 = deepcopy(distMI)


@save "./MI_3.bson" distMI_old
@save "./MI_K1_20.bson" distMI_K1_20
@save "./MI_K2_20.bson" distMI

# the differences between different MI models
for id in idx
    @printf "Chosen word: %s, index in Vocabulary: %i \n" uppercase(srcV[id]) id
    cos = srcV[distMI_old[:, id]]
    mah = srcV[distMI_K1_20[:, id]]
    mutual = srcV[distMI[:, id]]

    @printf "Words chosen by Cosine :%s \n" cos[end-10:end] |> reverse;
    @printf "Words chosen by Mahalanobis :%s \n" mah[end-10:end] |> reverse;
    @printf "Words chosen by MI :%s \n" mutual[end-10:end] |> reverse;


    @printf "------------------------------------------------------------------------ \n"
end
