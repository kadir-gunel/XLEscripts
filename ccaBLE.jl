cd(@__DIR__)

using OhMyREPL
using LinearAlgebra
using XLEs
using MultivariateStats
using CUDA

en = "../XLEs/data/exp_raw/embeddings/en";
es = "../XLEs/data/exp_raw/embeddings/es";
valfile= "../vecmap/data/dictionaries/en-es.test.txt";


@info "Reading files"
src_voc, X = readBinaryEmbeddings(en);
trg_voc, Y = readBinaryEmbeddings(es);


src_w2i = word2idx(src_voc);
trg_w2i = word2idx(trg_voc);
validation = readValidation(valfile, src_w2i, trg_w2i);

X = X |> normalizeEmbedding |> cu;
Y = Y |> normalizeEmbedding |> cu;

rng = 1:Int(4e3)
subX = X[:, rng] |> cu;
subY = Y[:, rng] |> cu;

keep_prob = 0.1
src_idx, trg_idx = buildSeedDictionary(subX, subY)


stochastic_interval   = 50
stochastic_multiplier = 2.0
stochastic_initial    = .1
threshold = Float64(1e-6) # original threshold = Float64(1e-6)
best_objective = -100. # floating
objective = -100. # floating
it = 1
last_improvement = 0
keep_prob = stochastic_initial
stop = !true
W = CUDA.zeros(size(X, 1), size(X,1))
Wt_1 = CUDA.zeros(size(W))
λ = Float32(1.);
src_size = Int(20e3)
trg_size = Int(20e3);


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
            accuracy, similarity = validate((W * X), Y, validation)
            @info "Accuracy on validation set :", accuracy
            @info "Validation Similarity = " , similarity
        end
        it += 1

end


C = fit(CCA, (W * X[:, src_idx]) |> Array, (Y[:, trg_idx]) |> Array)

CX = (Matrix(X) .- C.xmean)
CY = (Matrix(Y) .- C.ymean)

XW = C.xproj' * Matrix(W) * Matrix(CX)
YW = C.yproj' * Matrix(CY)

accuracy, similarity = validate(XW |> normalizeEmbedding |> cu, YW |> normalizeEmbedding |> cu, validation)
accuracy, similarity = validateCSLS(XW |> normalizeEmbedding |> cu, YW |> normalizeEmbedding |> cu, validation)


XW2, YW2 = advancedMapping(permutedims(XW) |> cu, permutedims(YW) |> cu, src_idx, trg_idx);

accuracy, similarity = validate(XW2 |> normalizeEmbedding |> cu, YW2 |> normalizeEmbedding |> cu, validation)
accuracy, similarity = validateCSLS(XW2 |> normalizeEmbedding |> cu, YW2 |> normalizeEmbedding |> cu, validation)



XWOrig, YWOrig = advancedMapping(permutedims(W * X) |> cu, permutedims(Y) |> cu, src_idx, trg_idx);

accuracy, similarity = validate(XWOrig |> normalizeEmbedding |> cu, YWOrig |> normalizeEmbedding |> cu, validation)
accuracy, similarity = validateCSLS(XWOrig |> normalizeEmbedding |> cu, YWOrig |> normalizeEmbedding |> cu, validation)


cond((Y * Y') |> Array)
cond(YW * YW')
cond((YW2 * YW2') |> normalizeEmbedding |> Array)
cond((YWOrig * YWOrig') |> normalizeEmbedding |>  Array)

cond(X * X')
cond((XW * XW') |> normalizeEmbedding)
cond((XW2 * XW2') |> normalizeEmbedding)
cond((XWOrig * XWOrig') |> normalizeEmbedding)
