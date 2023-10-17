cd(@__DIR__)
using Printf
using OptimalTransport
using Random
using XLEs
using BSON: @load, @save



function sqrt_eigen(subE)
      F = svd(subE)
      return F.U * diagm(sqrt.(F.S)) * F.Vt
end

function procrustes(XP, Y)
      F = svd(XP * Y');
      W = F.U * F.Vt
      return W'
end


function convex_initialization(subX, subY; niter=100, λ=Float32(.05), apply_sqrt=false)
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
      G = zeros(size(P))
      μ, 𝒗  = ones(r), ones(r) # \biv for v symbol
      for it in 1:niter
            G = (K2_X * P) + (P * K2_Y) - 2 * (K_X * P * K_Y)
            q = sinkhorn(μ, 𝒗, G, λ)
            α = (2.0 / (2.0 + it))
            P = (α .* q) + ((1. - α) .* P)
      end

      obj = norm((K_X * P) - (P * K_Y))
      printstyled(obj, color=:green)
      println();
      return P, G # procrustes(subX * P, subY)
end



function main(X, Y, src_idx, trg_idx, validation)
    src_size = Int(20e3);
    trg_size = Int(20e3);
    @info "Starting Training"
    stochastic_interval   = 50
    stochastic_multiplier = 2.0
    stochastic_initial    = .1
    threshold = Float64(1e-6) # original threshold = Float64(1e-6)
    best_objective = -100. # floating
    objective = 100.
    it = 1
    last_improvement = 0
    keep_prob = stochastic_initial
    stop = !true
    W = CUDA.zeros(size(X, 1), size(X,1))
    Wt_1 = CUDA.zeros(size(W))
    λ = Float32(1.)

    while true
        printstyled("Iteration : # ", it, "\n", color=:green)
        # increase the keep probability if we have not improved in stochastic_interval iterations
        if it - last_improvement > stochastic_interval
            if keep_prob >= 1.0
                stop = true
            end
            keep_prob = min(1., stochastic_multiplier * keep_prob)
            println("Drop probability : ", 100 - 100 * keep_prob)
            last_improvement = it
        end

        if stop
            break
        end

        # updating training dictionary
        src_idx, trg_idx, objective, W = train(X[:, 1:src_size], Y[:, 1:trg_size], Wt_1, src_idx, trg_idx, src_size, trg_size, keep_prob, objective; stop=stop, time=true, lambda=λ)

        if objective - best_objective >= threshold
            last_improvement = it
            best_objective = objective
        end

        # validating
        if mod(it, 10) == 0
            accuracy, similarity = validate((W' * X), Y, validation)
            @info "Accuracy on validation set :", accuracy
            @info "Validation Similarity = " , similarity
        end
        it += 1

    end
end


src, trg, valfile = EmbeddingData() |> readData;
srcV, X = src[1], src[2];
trgV, Y = trg[1], trg[2];

src_w2i = word2idx(srcV);
trg_w2i = word2idx(trgV);
validation = readValidation(valfile, src_w2i, trg_w2i);

X, Y = map(normalizeEmbedding, [X, Y]);

rng = 1:Int(10e3)
subX = @view(X[:, rng]);
subY = @view(Y[:, rng]);

@info "Building Seed Dictionary by Sinkhorn - convex initialization"
@time P, C = convex_initialization(subX, subY, apply_sqrt=true);
W0 = procrustes(subX * P , subY)

subXW = W0 * subX;
subXW, subY = map(cu, [subXW, -subY])

trg_idx_forward,  best_sim_forward  = XLEs.updateDictionary(subXW, subY, .1, direction=:forward)
src_idx_backward, best_sim_backward = XLEs.updateDictionary(subXW, subY, .1, direction=:backward)

src_idx = vcat(cu(collect(rng)), src_idx_backward)
trg_idx = vcat(trg_idx_forward, cu(collect(rng)))

main(X |> cu, Y |> cu, src_idx, trg_idx, validation)

distcov(D::Matrix) = D' * D;

distX = subX |> Matrix |> distcov;
distY = subY |> Matrix |> distcov;

C = fit(CCA, distX, distY, method=:svd)
