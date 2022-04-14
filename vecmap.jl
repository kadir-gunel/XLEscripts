cd(@__DIR__)
using OhMyREPL
using LinearAlgebra
using Statistics
using XLEs
using CUDA
using Base.Iterators
using BSON: @save, @load
using Printf

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

        # src_idx, trg_idx, objective, W = XLEs.train2(X[:, 1:src_size], Y[:, 1:trg_size], Wt_1, src_idx, trg_idx, src_size, trg_size, keep_prob, objective; stop=stop, time=true, lambda=λ)


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
    return W, src_idx, trg_idx
end

src, trg, val = EmbeddingData(trgLang="fi") |> readData;

srcV, X = map(i -> src[i],  1:2)
trgV, Y = map(i -> trg[i],  1:2)

X, Y = map(normalizeEmbedding, [X, Y]);
X, Y = map(cu, [X, Y]);

src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
validation = readValidation(val, src_w2i, trg_w2i);

rng = 1:Int(4e3)
subx = X[:, rng] |> Matrix
#|> XLEs.sqrt_eigen;
suby = Y[:, rng] |> Matrix
#|> XLEs.sqrt_eigen;

# @time src_idx, trg_idx = XLEs.buildSeedDictionary0(subx, suby)
# @time src_idx, trg_idx = buildSeedDictionary(subx |> cu, suby |> cu)
@time src_idx, trg_idx = XLEs.buildMahalanobisDictionary(subx |> Matrix, suby |> Matrix);
# @time src_idx, trg_dix = XLEs.mahalanobisGPU(subx, suby);

W, src_idx, trg_idx = main(X, Y, src_idx, trg_idx, validation);
kacc1, sims1 = validate(W * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)
kacc2, sims2 = validateCSLS(W * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)


XW, YW = advancedMapping(permutedims(W * X), permutedims(Y), src_idx, trg_idx);
kacc3, sims3 = validate(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)
kacc4, sims4= validateCSLS(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)


@printf "           |Accuracy | Similarity"
@printf "==========================================="
@printf "KNN        | %.4f  |  %.4f" kacc1 sims2
@printf "CSLS       | %.4f  |  %.4f" kacc2 sims2
@printf "------------------------------------------"
@printf "KNN_{adv}  | %.4f  |  %.4f" kacc3 sims3
@printf "CSLS_{adv} | %.4f  |  %.4f" kacc4 sims4








#=
list = [W, src_idx, trg_idx];
W, src_idx, trg_idx = map(Array, list)

@save "./W_vecmap.bson" W
@save "./src_vecmap.bson" src_idx
@save "./trg_vecmap.bson" trg_idx


info = SplitInfo();
Postprocessing(Matrix(X), Matrix(Y), Array(src_idx), Array(trg_idx), replaceSingulars, info, validation, srcV, trgV ) |> validateModel
Postprocessing(Matrix(X), Matrix(Y), Array(src_idx), Array(trg_idx), (x) -> (x), info, validation, srcV, trgV ) |> validateModel

info2 = SplitInfo(change=true)
Postprocessing(Matrix(X), Matrix(Y), Array(src_idx), Array(trg_idx), replaceSingulars, info2, validation, srcV, trgV ) |> validateModel
Postprocessing(Matrix(X), Matrix(Y), Array(src_idx), Array(trg_idx), (x) -> (x), info2, validation, srcV, trgV ) |> validateModel


=#
