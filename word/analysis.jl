cd(@__DIR__)
using LinearAlgebra
using Statistics

using StatsBase
using Distances
using MKL


using XLEs

using PyCall


using Plots
gaston()

@pyimport fasttext as ft 
@pyimport fasttext.util as ftu


ft_en = ft.load_model("cc.en.300.bin")
ft_de = ft.load_model("cc.de.300.bin")
ft_es = ft.load_model("cc.es.300.bin")
ft_it = ft.load_model("cc.it.300.bin")

words, freqs = ft_en.get_labels(include_freq=true)
E = collect(ft_en.get_input_vector(i) for i in 1:length(words))
E = reduce(hcat, E)




# load Glove EN
glove = "/run/media/phd/PhD/DATA/Glove/glove.6B.300d.txt"
muse = "/run/media/phd/PhD/MUSE/data/wiki.en.vec"
vecmap = "/run/media/phd/PhD/vecmap/data/embeddings_txt/en.emb.txt"

	"""
	karsilastirma islemi 2 tuple olarak gerceklestirilecek : 
	i. <G,V>, 
	ii. <G, M>, 
	iii. <V,M>
"""

function load_tuple(embed1::String, embed2::String)
	src, S = readEmbeddings(embed1)
	trg, T = readEmbeddings(embed2) 

	s_w2i, t_w2i = map(word2idx, [src, trg])
	vocabulary = intersect(src, trg)

	s_idx = Int64[]
	t_idx = Int64[]

	for word in vocabulary
		push!(s_idx, s_w2i[word])
		push!(t_idx, t_w2i[word])
	end
	return s_idx, t_idx, S, T
end


gv_idx, vg_idx, G, V = load_tuple(glove, vecmap)

gm_idx, mg_idx, _, M = load_tuple(glove, muse)

vm_idx, mv_idx, _, _ = load_tuple(vecmap, muse)

@info  "" length(gv_idx) length(gm_idx) length(vm_idx)


function checkCosineSimilarity(embed1, embed2) 
	K = 1 .- pairwise(CosineDist(), embed1, embed2)
	L = getindex.(argmax(K, dims=1), 1)
	return L
end 









