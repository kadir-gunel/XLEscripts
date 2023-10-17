cd(@__DIR__)
# using OhMyREPL
using LinearAlgebra
using Statistics
using XLEs
using CUDA
using Base.Iterators
using BSON: @save, @load
using Printf
using ParallelKMeans
using Distances

getSentences(file) = file |> readlines .|> l -> split(l) .|> i -> String(i)


langs = ["it", "fi"]

for lang in langs 
    val = "/run/media/phd/PhD/vecmap/data/dictionaries/en-$(lang).test.txt"

    @time srcV, X = readBinaryEmbeddings("/run/media/phd/PhD/vecmap/data/embeddings/en");
    @time trgV, Y = readBinaryEmbeddings("/run/media/phd/PhD/vecmap/data/embeddings/$(lang)");

    X, Y = map(permutedims, [X, Y])
    X, Y = map(XLEs.unit, [X, Y])


   
    X, Y = map(cu, [X, Y])
    rng = 1:Int(4e3)
    subx = X[:, rng];
    suby = Y[:, rng];

    @time src_idx, trg_idx = buildSeedDictionary(subx, suby, sim_size=Int(4e3))

    src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
    validation = readValidation(val, src_w2i, trg_w2i);

    X, Y = map(cu, [X, Y])
    W, src_idx, trg_idx = XLEs.main(X, Y, src_idx, trg_idx, validation; src_size=Int(20e3), trg_size=Int(20e3));

    W = W |> Array

    @save "./models/W_$(lang).bson" W src_idx trg_idx
end 


kacc1, sims1 = validate(W * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)
kacc2, sims2 = validateCSLS(W * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)

XW, YW = advancedMapping(permutedims(W * X), permutedims(Y), src_idx, trg_idx);
kacc1, sims1 = validate(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)
kacc2, sims2 = validateCSLS(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)


kacc1, sims1 = validate(XW |> unit, YW |> unit, validation)
kacc2, sims2 = validateCSLS(XW |> unit, YW |> unit, validation)



# analysis of replacement
using Plots
pgfplotsx()




function reweighting(XW, YW, F; atype=cu)
    src_reweight = 0.5
    trg_reweight = 0.5
    XW = XW * atype(diagm(F.S) .^ src_reweight);
    YW = YW * atype(diagm(F.S) .^ trg_reweight);
    return XW, YW
end


function dewhiten(XW, YW, F, Wx1, Wy1)
    src_dewhiten = "src"  # || global src_dewhiten = "trg"
    XW = isequal(src_dewhiten, "src") ? XW * F.U' * XLEs.cuinv(Wx1) * F.U : XW * F.Vt * XLEs.cuinv(Wy1) * F.V
    trg_dewhiten = "trg" # || global trg_dewhiten = "trg"
    YW = isequal(trg_dewhiten, "src") ? YW * F.U' * XLEs.cuinv(Wx1) * F.U : YW * F.Vt * XLEs.cuinv(Wy1) * F.V
    return XW, YW
end


# whitened X, Y
XW, YW, Wx1, Wy1 = whiten(X, Y, src_idx, trg_idx)
XO, YO, F = orthmapping(XW, YW, src_idx, trg_idx)
XR, YR = reweighting(XO, YO, F; atype=Array)
XD, YD = dewhiten(XR, YR, F, Wx1, Wy1)

logl(S) = log.(S)
Xs = [W * X, XW, XO, XR, XD]
XNs = map(normalizeEmbedding, Xs)

Xs = map(Array, Xs);
XNs = map(Array, XNs);

logSs = map(logl, map(svdvals, Xs))
logNSs= map(logl, map(svdvals, XNs))



# singulars = [logxts, logxtus, logxtrepls, logxturepls];
labels= ["log WX", "log WX Whiten", "log WX WhiteOrth", "log WX Reweight" , "log WX Dewhiten"]
colors= [:blue1, :purple1, :purple2, :purple3, :purple4]
plots = []

for (singular, color, label) in zip(logNSs, colors, labels)
    push!(plots, histogram(singular, bins=100, label=label, color=color);)
end

plot(plots[1], plots[3], plots[4], plots[5], layout=(4, 1))
savefig("./figs/mapping_steps_norm_en.svg")





Ys = [Y, YW, YO, YR, YD]
YNs = map(normalizeEmbedding, Ys)

Ys = map(Array, Ys);
YNs = map(Array, YNs);

logSs = map(logl, map(svdvals, Ys));
logNSs= map(logl, map(svdvals, YNs));



# singulars = [logxts, logxtus, logxtrepls, logxturepls];
labels= ["log WX", "log WX Whiten", "log WX WhiteOrth", "log WX Reweight" , "log WX Dewhiten"]
colors= [:blue1, :purple1, :purple2, :purple3, :purple4]
plots = []

for (singular, color, label) in zip(logNSs, colors, labels)
    push!(plots, histogram(singular, bins=100, label=label, color=color);)
end

plot(plots[1], plots[3], plots[4], plots[5], layout=(4, 1))
savefig("./figs/mapping_steps_norm_es.svg")



for (i, j) in Iterators.product(2:5, 2:5)
    @info validate(permutedims(Xs[i]) |> XLEs.unit, permutedims(Ys[j]) |> XLEs.unit, validation), (i , j)
end


for (i, j) in Iterators.product(2:5, 2:5)
    @info validate(permutedims(Xs[i]), permutedims(Ys[j]), validation), (i , j)
end
