cd(@__DIR__)

using LinearAlgebra
using Statistics
using StatsBase
using XLEs
using CUDA
using BSON: @save, @load
using Printf

using ParallelKMeans

using Plots
pgfplotsx()


glove= "/run/media/phd/PhD/DATA/Glove/glove.6B.300d.txt"
muse = "/run/media/phd/PhD/MUSE/data/fasttext/"
vecmap="/run/media/phd/PhD/vecmap/data/embeddings/"
langs = ["en", "es", "de", "fi", "it"]



# embeddings = glove, muse, vecmap_es
X = map(readEmbeddings, [glove, muse]);

vecmaps = []
for lang in langs 
	_, V = vecmap * lang |> readBinaryEmbeddings
	V = V |> permutedims
	push!(vecmaps, V)
end

muses = []
for lang in langs
	_, M = muse * "wiki.$(lang).vec" |> readEmbeddings
	push!(muses, M)
end


_, G = readEmbeddings(glove)

X = getindex.(X, 2)

push!(X, V)


Xu = map(XLEs.unit, X)

# Xur = [Xu[i][:, 1:Int(20e3)] for i in 1:3]

logl(x) = log.(x)

function splitEmbedding(E::Matrix{Float32})
	freqs = 1:Int(15e3)
	ordinary = freqs[end]+1:Int(80e3)
	rare = ordinary[end]+1:size(E, 2) 

	F = E[:, freqs]
	O = E[:, ordinary]
	R = E[:, rare]
	
	FOR = [F, O, R]

	sfor  = map(logl, map(svdvals, FOR))

	return sfor
end

su = map(logl, map(svdvals, Xu))
# μ = map(mean, su)

# s = collect(su[i] .- μ[i] for i in 1:3)

clrs = [:red2, :blue3, :green3]
labs = ["Log Glove", "Log Muse", "Log Vecmap FI"]
logp = [histogram(x, bins=300, label=lab, color=clr) for (x, lab, clr) in zip(su, labs, clrs)];

plot(logp[1], logp[2], logp[3], layout=(3, 1))

savefig("./figs/glove_muse_vecmap_fi.svg")


Gu = G |> XLEs.unit
singularG = Gu |> splitEmbedding



clrs = [:red2, :orange2, :purple2]
labs = ["Freq", "Ordinary", "Rare"]
logp = [histogram(x, bins=300, label=lab, color=clr) for (x, lab, clr) in zip(singularG, labs, clrs)];
plot(logp[1], logp[2], logp[3], layout=(3, 1))
savefig("./figs/glove_for.svg")


Vus = map(XLEs.unit, vecmaps)
singulars = map(splitEmbedding, Vus)


Mus = map(XLEs.unit, muses)
singulars = map(splitEmbedding, Vus)


# usingulars = map(splitEmbedding, Xu)


# GMV = collect(singulars[i] for i in 1:length(singulars))
# uGMV = collect(getindex(singulars[i], 2) for i in 1:length(singulars))



for (lang, gmv) in zip(langs, singulars)
	clrs = [:red2, :orange2, :purple2]
	labs = ["$(lang) Freq", "$(lang) Ordinary", "$(lang) Rare"]
	logp = [histogram(x, bins=300, label=lab, color=clr) for (x, lab, clr) in zip(gmv, labs, clrs)];
	plot(logp[1], logp[2], logp[3], layout=(3, 1))
	savefig("./figs/vecmap_$(lang).svg")
end



# reading validation datasets for pairwise distance calculation
# first let's check the distance between vecmap word embeddings 
# ex. english - spanish , en-fi, en-etc.

using Distances

vecmap="/run/media/phd/PhD/vecmap/data/embeddings/"
langs = ["en", "es", "de", "fi", "it"]

vecmaps = []
for lang in langs 
	V = vecmap * lang |> readBinaryEmbeddings
	push!(vecmaps, V)
end




srcV, X = vecmaps[1]
X = permutedims(X) 
X = X |> XLEs.unit
for i in 2:length(langs)
	trgV, Y = vecmaps[i]
	Y = Y |> permutedims
	Y = Y |> XLEs.unit
	val = "/run/media/phd/PhD/vecmap/data/dictionaries/en-$(langs[i]).test.txt"
	src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
	validation = readValidation(val, src_w2i, trg_w2i);
	k = validation |> keys |> collect
	v = validation |> values |> collect .|> first
	
	Fx = svd(X[:, k])
	Fy = svd(Y[:, v])

	cosxx = 1 .- pairwise(CosineDist(), Fx.U * diagm(Fx.S), Fy.U * diagm(Fy.S))



end

