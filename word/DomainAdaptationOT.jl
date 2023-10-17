cd(@__DIR__)
using OhMyREPL
using PyCall
using MKL
using LinearAlgebra
using Random
using Printf
using XLEs
using Embeddings
using BSON: @save, @load


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


function objective(X, Y, R; n=Int(5e3))
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

function rankSimilarities(ŷ, y)
    cosŷ = ŷ' * ŷ |>  XLEs.dist2sim;
    cosy = y' * y|> XLEs.dist2sim;
    F = rsvd(cosŷ * cosy', 10, 3)
    W = F.V * F.U'
    return W
end

function align2(X, Y, R, src_idx, trg_idx, validation; α=10., bsz=200, nepoch=5, niter=Int(1e3), nmax=Int(10e3), λ=.05)
    seed = MersenneTwister(1234);
    for epoch in 1:nepoch
        for it in 1:niter
            # sidx = randperm(seed, nmax)[1:bsz];
            # tidx = randperm(seed, nmax)[1:bsz];
            xt = X[:, src_idx[1:bsz]];
            yt = Y[:, trg_idx[1:bsz]];
            C = -(yt' * R * xt) # actually this is kind of covariance matrix between y and ŷ !!!
            P = sinkhorn(ones(bsz), ones(bsz), C, λ)
            G = -((yt * P) * xt')
            R -= (α / bsz * G)
            F = svd(R)
            R = (F.U * F.Vt) # son islem olarak transpose alindi
        end
        bsz *= 2
        niter = Int(div(niter, 4))
        @printf "epoch: %i ,  objective: %.2f \n" epoch objective(X, Y, R)
    end
    return R #, Ps[end], src_idx[end], trg_idx[end]
end



# our aim is to rotate Glove Embeddings toward FastText Embeddings

G = load_embeddings(GloVe{:en}, 4; max_vocab_size=Int(200e3))
srcV, X = G.vocab, G.embeddings

src , _, valfile = EmbeddingData(trgLang="es") |> readData;
trgV, Y = map(i -> src[i], 1:2)

@info "Normalization Process"
X, Y = map(normalizeEmbedding, [X, Y]);

@info "Building indices for vocabularies"
src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
intersectedWords = intersect(srcV, trgV)
src_idx = [src_w2i[intersectedWords[i]] for i in 1:length(intersectedWords)]
trg_idx = [trg_w2i[intersectedWords[i]] for i in 1:length(intersectedWords)]




@info "Validation"
validation = readValidation(valfile, src_w2i, trg_w2i);


subx = X[:, 1:Int(2.5e3)];
suby = Y[:, 1:Int(2.5e3)];


@info "Convex Initialization"
@time P = convex_initialization(subx, suby, apply_sqrt=true);
R0 = procrustes(subx * P, suby);

@time R = align(X, Y, R0, validation, α=500, bsz=20, nepoch=1, niter=Int(1e1), nmax=Int(10e3));

@info "Training " #finding the first rotation matrix R0 as seed
@time R = align2(X, Y, R0, src_idx, trg_idx, validation, α=500., bsz=2000, niter= Int(2e3), nmax=Int(20e3));


kacc, ksim = validate(Float32.(R) * X |> normalizeEmbedding |> Array, Y |> normalizeEmbedding |> Array, validation)
@printf "KNN Accuracy: %.3f Similarity: %.3f \n" kacc ksim


XR = Float32.(R) * X;

_ , trg, valfile = EmbeddingData(trgLang="de") |> readData;
trgV, Y = map(i -> trg[i], 1:2)

@info "Normalization Process"
XR, Y = map(normalizeEmbedding, [XR, Y]);

@info "Building indices for vocabularies"
src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
validation = readValidation(valfile, src_w2i, trg_w2i);

subx = XR[:, 1:Int(2.5e3)];
suby = Y[:, 1:Int(2.5e3)];


@info "Convex Initialization"
@time P = convex_initialization(subx, suby, apply_sqrt=true);
R0 = procrustes(subx * P, suby);

# @time R = align(X, Y, R0, validation, α=500, bsz=20, nepoch=1, niter=Int(1e1), nmax=Int(10e3));

@info "Training " #finding the first rotation matrix R0 as seed
@time R = align(XR, Y, R0, validation, α=500., bsz=2000, niter= Int(2e3), nmax=Int(20e3));


kacc, ksim = validate(Float32.(R) * XR |> normalizeEmbedding |> Array, Y |> normalizeEmbedding |> Array, validation)
@printf "KNN Accuracy: %.3f Similarity: %.3f \n" kacc ksim
