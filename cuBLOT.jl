cd(@__DIR__)
using OhMyREPL
using OptimalTransport
using LinearAlgebra
using Random
using Printf
using XLEs
using CUDA


function procrustes(XP, Y)
      F = CUDA.CUBLAS.svd(XP * Y');
      W = F.U * F.Vt
      return W'
end

function sqrt_eigen(subE)
      F = CUDA.CUBLAS.svd(subE)
      F.U * cu(diagm(sqrt.(F.S))) * F.Vt
end


function objective(X, Y, R; n=Int(5e3))
      subX = X[:, 1:n]
      subY = Y[:, 1:n]
      C = -(subY' *  (R * subX))
      P = sinkhorn(CUDA.ones(n) |> cu, CUDA.ones(n) |> cu, C, .025) .|> Float32
      return 1000 * CUDA.norm((R * subX) - (subY * P)) / n
end

function relaxation(K_X, K_Y, K2_X, K2_Y, P, μ, 𝒗, λ, α)
      G = (K2_X * P) + (P * K2_Y) - 2 * (K_X * P * K_Y)
      q = sinkhorn(μ, 𝒗, G, λ)
      α = 2.0 / (2.0 + it)
      P = (α .* q) + ((1. - α) .* P)
end

function convex_initialization(subX, subY; niter=100, λ=Float32(.05), apply_sqrt=false)
      c, r = size(subX)
      if apply_sqrt
            subX, subY = map(sqrt_eigen, [subX, subY])
      end
      K_X = subX' * subX
      K_Y = subY' * subY

      K_Y = K_Y * (norm(K_X) / norm(K_Y))

      K2_X = K_X * K_X
      K2_Y = K_Y * K_Y

      P = (ones(r, r) ./ r) |> cu
      μ, 𝒗  = ones(r) |> cu , ones(r) |> cu # \biv for v symbol
      for it in 1:niter
            G = (K2_X * P) + (P * K2_Y) - 2 * (K_X * P * K_Y)
            q = sinkhorn(μ, 𝒗, G, λ) .|> Float32
            α = 2.0 / (2.0 + it) |> Float32
            P = (α .* q) + ((1. - α) .* P) .|> Float32
      end

      obj = norm((K_X * P) - (P * K_Y))
      printstyled(obj, color=:green)
      println();
      return P # procrustes(subX * P, subY)
end


function align(X, Y, R; α=10., bsz=200, nepoch=5, niter=Int(1e3), nmax=Int(10e3), λ=Float32(.05))
      seed = MersenneTwister(1234);
      for epoch in 1:nepoch
            for it in 1:niter
                  xt = @view(X[:, randperm(seed, nmax)[1:bsz]])
                  yt = @view(Y[:, randperm(seed, nmax)[1:bsz]])
                  C = -(yt' * R * xt) .|> Float32
                  P = sinkhorn(CUDA.ones(bsz), CUDA.ones(bsz), C, λ) .|> Float32
                  G = -((yt * P) * xt') .|> Float32
                  R -= α / bsz * G .|> Float32
                  F = CUDA.CUBLAS.svd(R)
                  R = (F.U * F.Vt) # son islem olarak transpose alindi
            end
            bsz *= 2
            niter = Int(div(niter, 4))
            @printf "epoch: %i ,  objective: %.2f \n" epoch objective(X, Y, R)
      end
      return R
end


@info "Reading files"
en = "../XLEs/data/exp_raw/embeddings/en";
es = "../XLEs/data/exp_raw/embeddings/es";
valfile= "../vecmap/data/dictionaries/en-es.test.txt";

src_voc, X = readBinaryEmbeddings(en)
trg_voc, Y = readBinaryEmbeddings(es)

@info "Reading Validation Files"
src_w2i = word2idx(src_voc);
trg_w2i = word2idx(trg_voc);
validation = readValidation(valfile, src_w2i, trg_w2i)

@info "Normalization Process"
X = X |> normalizeEmbedding |> cu;
Y = Y |> normalizeEmbedding |> cu;

subX = @view(X[:, 1:Int(2.5e3)])
subY = @view(Y[:, 1:Int(2.5e3)])


@info "Convex Initialization"
P = convex_initialization(subX, subY, apply_sqrt=true);
R0 = procrustes(subX * P, subY) .|> Float32
@info "Training " #finding the first rotation matrix R0 as seed
@time R, = align(X, Y, R0, α=500, bsz=4, nepoch=1, niter=Int(1e1), nmax=Int(10e3));
@time R  = align(X, Y, R0, α=500., bsz=500, niter= Int(5e3), nmax=Int(10e3))

R = R |> cu;
X = X |> cu;
Y = Y |> cu;

@info "Validation"
acc, sim = validate(X, R' * Y, validation)
acc, sim = validateCSLS(X, R' * Y, validation) 
@printf "Accuracy: %.2f Similarity: %.2f \n" acc sim
