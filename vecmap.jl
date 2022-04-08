cd(@__DIR__)
using OhMyREPL
using LinearAlgebra
using Statistics
using XLEs
using CUDA
using Base.Iterators


getSentences(file) = file |> readlines .|> l -> split(l) .|> i -> String(i)

function main(X, Y, src_idx, trg_idx, validation; src_size=Int(20e3), trg_size=Int(20e3))
    @info "Starting Training"
    stochastic_interval   = 50
    stochastic_multiplier = 2.0
    stochastic_initial    = .1
    threshold = Float64(1e-6) # original threshold = Float64(1e-6)
    best_objective = objective = -100. # floating
    it = 1
    last_improvement = 0
    keep_prob = stochastic_initial
    stop = !true
    W = CUDA.zeros(size(X, 1), size(X,1))
    Wt_1 = CUDA.zeros(size(W))
    λ = Float32(1)

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

        src_idx, trg_idx, objective, W = XLEs.train2(X[:, 1:src_size], Y[:, 1:trg_size], Wt_1, src_idx, trg_idx, keep_prob, objective; stop=stop, lambda=λ)
        #src_idx, trg_idx, objective, W = XLEs.train2(X[:, 1:src_size], Y[:, 1:trg_size], Wt_1, src_idx, trg_idx, src_size, trg_size, keep_prob, objective; stop=stop, lambda=λ)

        # updating training dictionary
        # src_idx, trg_idx, objective, W = train(X[:, 1:src_size], Y[:, 1:trg_size], Wt_1, src_idx, trg_idx, src_size, trg_size, keep_prob, objective; stop=stop, time=true, lambda=λ)

        if objective - best_objective >= threshold
            last_improvement = it
            best_objective = objective
        end

        # validating
        if mod(it, 10) == 0
            accuracy, similarity = validate((W * X), Y, validation)
            @info "Accuracy on validation set :", accuracy
            @info "Validation Similarity = " , similarity
        end

        it += 1

    end
    return W, src_idx, trg_idx
end



src, trg, val = EmbeddingData() |> readData;

srcV, X = map(i -> src[i],  1:2)
trgV, Y = map(i -> trg[i],  1:2)

X, Y = map(normalizeEmbedding, [X, Y]);
X, Y = map(cu, [X, Y]);

src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
validation = readValidation(val, src_w2i, trg_w2i);


rng = 1:Int(4e3)
subx = X[:, rng];
suby = Y[:, rng];

# src_idx, trg_idx = buildSeedDictionary(subX, subY, sim_size=length(rng))
@time src_idx, trg_idx = XLEs.buildMahalanobisDictionary(subx |> Array, suby |> Array);
# @time src_idx, trg_dix = XLEs.mahalanobisGPU(subx, suby);

W, src_idx, trg_idx = main(X, Y, src_idx, trg_idx, validation)
acc, sims = validate(W * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)

XW, YW = advancedMapping(permutedims(W * X), permutedims(Y), src_idx, trg_idx);

acc, sims = validate(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)
acc, sims= validateCSLS(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)

