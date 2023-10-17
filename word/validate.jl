cd(@__DIR__)

using XLEs
using CUDA
using BSON: @load 
using Printf
# validate vecmap 

using Plots
pgfplotsx()


function printformat(lang::String, knn, ksim, cnn, csim)
	@printf "Langauge: %s |Accuracy | Similarity\n"  lang
	@printf "===========================================\n"
	@printf "KNN          | %.4f  |  %.4f \n" knn ksim
	@printf "CSLS         | %.4f  |  %.4f \n" cnn csim
	@printf "------------------------------------------\n"
end


src = "en"
langs = ["es", "it", "fi", "de"]
we = "/run/media/phd/PhD/vecmap/data/embeddings/"

for lang in langs
	srcV, X = readBinaryEmbeddings(we * src)
	trgV, Y = readBinaryEmbeddings(we * lang)

	X, Y = map(permutedims, [X, Y]) 
	X, Y = map(normalizeEmbedding, [X, Y])

	@load "./models/W_$(lang).bson" W src_idx trg_idx

	W, X, Y = map(cu, [W, X, Y])


	val = "/run/media/phd/PhD/vecmap/data/dictionaries/en-$(lang).test.txt"

	src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
	validation = readValidation(val, src_w2i, trg_w2i);


	kacc1, sims1 = validate(W * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)
	kacc2, sims2 = validateCSLS(W * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)

	printstyled("Normal Results\n", color=:green)
	printformat(lang, kacc1, sims1, kacc2, sims2)


	XW, YW = advancedMapping(permutedims(W * X), permutedims(Y), src_idx, trg_idx);
	
	kacc1, sims1 = validate(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)
	kacc2, sims2 = validateCSLS(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)

	printstyled("advancedMapping Results\n", color=:blue)
	printformat(lang, kacc1, sims1, kacc2, sims2)

end



# validate ot


for lang in langs
	srcV, X = readBinaryEmbeddings(we * src)
	trgV, Y = readBinaryEmbeddings(we * lang)

	X, Y = map(permutedims, [X, Y]) 
	X, Y = map(normalizeEmbedding, [X, Y])

	@load "./models/R_$(lang).bson" R 

	R, X, Y = map(cu, [R, X, Y])


	val = "/run/media/phd/PhD/vecmap/data/dictionaries/en-$(lang).test.txt"

	src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
	validation = readValidation(val, src_w2i, trg_w2i);


	kacc1, sims1 = validate(R * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)
	kacc2, sims2 = validateCSLS(R * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)

	printstyled("Normal Results\n", color=:green)
	printformat(lang, kacc1, sims1, kacc2, sims2)

	"""
	WE CANNOT APPLY THIS POST PROCESSING STEP 
	XW, YW = advancedMapping(permutedims(W * X), permutedims(Y), src_idx, trg_idx);
	
	kacc1, sims1 = validate(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)
	kacc2, sims2 = validateCSLS(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)

	printstyled("advancedMapping Results\n", color=:blue)
	printformat(lang, kacc1, sims1, kacc2, sims2)
	"""

end

using BSON: @load

@load "./models/W_$(lang).bson" W src_idx trg_idx




kacc1, sims1 = validate(R * X |> cu |> normalizeEmbedding, Y |> cu |> normalizeEmbedding, validation)
kacc1, sims1 = validate(Xnew |> cu |> normalizeEmbedding, Y |> cu |> normalizeEmbedding, validation)


kacc1, sims1 = validateCSLS(R * X |> cu |> normalizeEmbedding, Y |> cu |> normalizeEmbedding, validation)
kacc1, sims1 = validateCSLS(Xnew |> cu |> normalizeEmbedding, Y |> cu |> normalizeEmbedding, validation)


Fx = svd(W * X);
BoxCoxTrans.lambda(log.(Fx.S)).value
newsx = BoxCoxTrans.transform(log.(Fx.S), 0.013)
Xnew = Fx.U * diagm(Float32.(newsx)) * Fx.Vt

Fy = svd(Y);
BoxCoxTrans.lambda(log.(Fy.S)).value
newsy = BoxCoxTrans.transform(log.(Fy.S), 0.013)
Ynew = Fy.U * diagm(Float32.(newsy)) * Fy.Vt

kacc1, sims1 = validate(newX |> cu |> normalizeEmbedding, newY |> cu |> normalizeEmbedding, validation)
kacc1, sims1 = validateCSLS(newX |> cu |> normalizeEmbedding, newY |> cu |> normalizeEmbedding, validation)


ENV["PYTHON"] = "/home/phd/miniconda3/envs/torch2.0/bin/python"
using PyCall
@pyimport sklearn.preprocessing as skp



ptx = skp.PowerTransformer(method="yeo-johnson", standardize=false)
ptx.fit(permutedims(XW))
newX = ptx.transform(permutedims(XW)) |> permutedims

pty = skp.PowerTransformer(method="yeo-johnson", standardize=false)
pty.fit(permutedims(YW))
newY = pty.transform(permutedims(YW)) |> permutedims




XW, YW, Wx1, Wy1 = whiten(cu(X), cu(Y), src_idx, trg_idx)
XO, YO, F = orthmapping(cu(XW), cu(YW), src_idx, trg_idx)
XR, YR = reweighting(cu(XO), cu(YO), F; atype=cu)
XD, YD = dewhiten(cu(XR), cu(YR), F, Wx1, Wy1)

XW, YW, XO, YO, XR, YR, XD, YD = map(permutedims, [XW, YW, XO, YO, XR, YR, XD, YD])



kacc1, sims1 = validate(W * X |> cu |> normalizeEmbedding, Y |> cu |> normalizeEmbedding, validation)
kacc1, sims1 = validate(W * XW |> cu |> normalizeEmbedding, YW |> cu |> normalizeEmbedding, validation)
kacc1, sims1 = validate(XO |> cu |> normalizeEmbedding, YO |> cu |> normalizeEmbedding, validation)
kacc1, sims1 = validate(XR |> cu |> normalizeEmbedding, YR |> cu |> normalizeEmbedding, validation)
kacc1, sims1 = validate(XD |> cu |> normalizeEmbedding, YD |> cu |> normalizeEmbedding, validation)


XW, YW, XO, YO, XR, YR, XD, YD = map(Array, [XW, YW, XO, YO, XR, YR, XD, YD])

@. broadcastlog(x) = log(x)
K = map(svdvals, [XW, YW, XO, YO, XR, YR, XD, YD])
L = map(broadcastlog, K)

P = []
[push!(P, histogram(l, bins=100, color=:green)) for l in L]



XA, YA = advancedMapping(permutedims(W * X) |> cu, permutedims(Y) |> cu, src_idx, trg_idx);
kacc1, sims1 = validate(XA |> cu |> normalizeEmbedding, YA |> cu |> normalizeEmbedding, validation)


kacc1, sims1 = validateCSLS(newX |> cu |> normalizeEmbedding, newY |> cu |> normalizeEmbedding, validation)


kacc1, sims1 = validate(XW |> cu |> normalizeEmbedding, YW |> cu |> normalizeEmbedding, validation)
kacc1, sims1 = validateCSLS(XW |> cu |> normalizeEmbedding, YW |> cu |> normalizeEmbedding, validation)


ptx = skp.PowerTransformer(method="yeo-johnson", standardize=false)
ptx.fit(permutedims(W * X[:, src_idx[Int(20e3)+1:end]]))
subX = ptx.transform(permutedims(W * X[:, src_idx[Int(20e3)+1:end]])) |> permutedims

pty = skp.PowerTransformer(method="yeo-johnson", standardize=false)
pty.fit(permutedims(Y[:, trg_idx]))
subY = pty.transform(permutedims(Y[:, trg_idx])) |> permutedims




