cd(@__DIR__)
using OhMyREPL
using OptimalTransport
using LinearAlgebra
using Random
using Printf
using XLEs
using BSON: @save, @load
using CUDA

#@pyimport importlib.machinery as machinery
#loader = machinery.SourceFileLoader("utils","/home/PhD/github/fastText/alignment/utils.py")
#utils = loader[:load_module]("utils")

#@pyimport ot
#@pyimport numpy

relu(E::Matrix) = max.(0, E);

function gram_initialization(X, Y; sim_size::Int64=Int(2.5e3))
      src_idx, trg_idx = buildSeedDictionary(X |> cu , Y |> cu, sim_size=sim_size)
      R0, _ = mapOrthogonal(X[:, src_idx], Y[:, trg_idx])
      return R0 |> Array
end

function procrustes(XP, Y)
      F = svd(XP * Y');
      W = F.U * F.Vt
      return W'
end

function sqrt_eigen(subE)
      F = svd(subE)
      F.U * diagm(sqrt.(F.S)) * F.Vt
end


function object(X, Y, R; n=Int(5e3))
      subX = X[:, 1:n]
      subY = Y[:, 1:n]
      C = -(subY' *  (R * subX))
      P = sinkhorn(ones(n), ones(n), C, .025)
      return 1000 * norm((R * subX) - (subY * P)) / n
end



function convex_initialization(subX, subY; niter=100, λ=.05, apply_sqrt=false)
      c, r = size(subX)
      if apply_sqrt
            subX, subY = map(sqrt_eigen, [subX, subY])
      end
      K_X = subX' * subX
      K_Y = subY' * subY

      K_Y = K_Y * (norm(K_X) / norm(K_Y))

      K2_X = (K_X * K_X)
      K2_Y = (K_Y * K_Y)

      P = ones(r, r) ./ r
      μ, 𝒗  = ones(r), ones(r) # \biv for v symbol
      for it in 1:niter
            G = (K2_X * P) + (P * K2_Y) - 2 * (K_X * P * K_Y)
            q = sinkhorn(μ, 𝒗, G, λ)
            α = 2.0 / (2.0 + it)
            P = (α .* q) + ((1. - α) .* P)
      end

      obj = norm((K_X * P) - (P * K_Y))
      printstyled(obj, color=:green)
      println();
      return P # procrustes(subX * P, subY)
end


function align(X, Y, R, validation; α=10., bsz=200, nepoch=5, niter=Int(1e3), nmax=Int(10e3), λ=.05)
    seed = MersenneTwister(1234);
    Ps = []; src_idx = []; trg_idx = [];
    for epoch in 1:nepoch
        for it in 1:niter
            sidx = randperm(seed, nmax)[1:bsz];
            tidx = randperm(seed, nmax)[1:bsz];
            xt = X[:, sidx]
            yt = Y[:, tidx]
            C = -(yt' * R * xt) # actually this is kind of covariance matrix between y and ŷ !!!
                # this thing has a name it is called GRAM matrix !! Notice that it checks not the feature space but the words!!
            P = sinkhorn(ones(bsz), ones(bsz), C, λ)
                G = -((yt * P) * xt')
            R -= (α / bsz * G)
            F = svd(R)
            R = (F.U * F.Vt) # son islem olarak transpose alindi
            push!(Ps, P)
            push!(src_idx, sidx)
            push!(trg_idx, tidx)
        end
        bsz *= 2
        niter = Int(div(niter, 4))
        @printf "epoch: %i ,  objective: %.2f \n" epoch object(X, Y, R)
        kacc, ksim = validate(X |> cu, (R' * Y) |> cu, validation)
        @printf "knn: %.3f similarity: %.3f   \n" kacc ksim
        @printf "---------------------------- \n"
    end
    return R, Ps[end], src_idx[end], trg_idx[end]
end



#=
en = "./vecmap/data/embeddings/en.emb.txt";
es = "./vecmap/data/embeddings/es.emb.txt";
valfile= "./vecmap/data/dictionaries/en-es.test.txt";
=#
@info "Reading files"
en = "../XLEs/data/exp_raw/embeddings/en";
es = "../XLEs/data/exp_raw/embeddings/es";
valfile= "../vecmap/data/dictionaries/en-es.test.txt";

#=
maxload = Int(200e3)
w_src, X = utils.load_vectors(en, maxload, norm=true, center=true)
w_trg, Y = utils.load_vectors(es, maxload, norm=true, center=true)
src2trg, _ = utils.load_lexicon(valfile, w_src, w_trg)
=#
src_voc, X = readBinaryEmbeddings(en)
trg_voc, Y = readBinaryEmbeddings(es)

@info "Reading Validation Files"
src_w2i = word2idx(src_voc);
trg_w2i = word2idx(trg_voc);
validation = readValidation(valfile, src_w2i, trg_w2i);

@info "Normalization Process"
X = X |> normalizeEmbedding;
Y = Y |> normalizeEmbedding;

subX = X[:, 1:Int(2.5e3)];
subY = Y[:, 1:Int(2.5e3)];

# @info "Gram Initialization"
# R0 = gram_initialization(subX, subY)

@info "Convex Initialization"
@time P = convex_initialization(subX, subY, apply_sqrt=true);
R0 = procrustes(subX * P, subY);

@time R, Ps, sidx, tidx = align(X, Y, R0, validation, α=500, bsz=4, nepoch=1, niter=Int(1e1), nmax=Int(10e3));

@info "Training " #finding the first rotation matrix R0 as seed
@time R, Ps, sidx, tidx  = align(X, Y, R0, validation, α=500., bsz=500, niter= Int(2e3), nmax=Int(10e3));



R = R |> cu;
X = X |> cu;
Y = Y |> cu;

@info "Validation"
kacc, ksim = validate(R * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)
@printf "KNN Accuracy: %.3f Similarity: %.3f \n" kacc ksim

acc, sim = validateCSLS(R * X |> normalizeEmbedding,  Y |> normalizeEmbedding, validation)
@printf "CSLS Accuracy: %.3f Similarity: %.3f \n" acc sim

# XW, YW = advancedMapping(X |> permutedims, (R' * Y) |> permutedims, sidx, tidx)




R = R |> Array;
@save "./R.bson" R
@save "./Ps.bson" Ps
@save "./sidx.bson" sidx
@save "./tidx.bson" tidx


# loading the data
@load "./R.bson" R
@load "./Ps.bson" Ps
@load "./sidx.bson" sidx
@load "./tidx.bson" tidx

info  = SplitInfo(freqs=5e3, ordinary=25e3, rares=170e3)
info2 = SplitInfo(freqs=10e3, ordinary=50e3, rares=140e3)

newX = XLEs.replaceSingulars(X, info=info);
newY = XLEs.replaceSingulars(R' * Y, info=info2);

kacc, ksim = validate(newX , newY, validation)
kacc, ksim = validateCSLS(newX,  newY, validation)

