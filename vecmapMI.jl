using OhMyREPL
using PyCall
using XLEs
using TransferEntropy
using CUDA
using Printf
using BSON: @save, @load

# importing MI estimators
@pyimport importlib.machinery as machinery
loader = machinery.SourceFileLoader("entropy_estimators","/home/PhD/github/NPEET/npeet/entropy_estimators.py")
entropy_estimators = loader[:load_module]("entropy_estimators")
mi = entropy_estimators.mi

convertAndPermute(E) = E |> Array |> permutedims

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

        # src_idx, trg_idx, objective, W = XLEs.train2(X[:, 1:src_size], Y[:, 1:trg_size], Wt_1, src_idx, trg_idx, src_size, trg_size, keep_prob, objective; stop=stop, lambda=λ)
         src_idx, trg_idx, objective, W = train(X[:, 1:src_size], Y[:, 1:trg_size], Wt_1, src_idx, trg_idx, src_size, trg_size, keep_prob, objective; stop=stop, time=true, lambda=λ)

        # updating training dictionary
        #src_idx, trg_idx, objective, W = train(X[:, 1:src_size], Y[:, 1:trg_size], Wt_1, src_idx, trg_idx, src_size, trg_size, keep_prob, objective; stop=stop, time=true, lambda=λ)

        if objective - best_objective >= threshold
            last_improvement = it
            best_objective = objective
        end

        # validating
        if mod(it, 10) == 0
            accuracy, similarity = validate((W * X), Y, validation)
            @info "Accuracy on validation set :", accuracy
            @info "Validation Similarity = " , similarity
#            @printf "Mutual Information: %.5f" mi( (W * X)[:, src_idx] |> convertAndPermute, Y[:, trg_idx] |> convertAndPermute)
        end

        it += 1

    end
    return W, src_idx, trg_idx
end






src, trg, valfile = EmbeddingData() |> readData;

srcV, X = map(i -> src[i], 1:2)
trgV, Y = map(i -> trg[i], 1:2)

src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
validation = readValidation(valfile, src_w2i, trg_w2i);



X, Y = map(normalizeEmbedding, [X, Y])
X, Y = map(cu, [X, Y]);

rng = 1:Int(4e3)
subx = X[:, rng];
suby = Y[:, rng];

x, y = map(d -> Dataset(permutedims(d)), [subx |> Array, suby |> Array])
# @printf "Mutual Information Before Alignement: %.5f" mutualinfo(x, y, Kraskov2(3), base=exp(1))


@time @printf "Mutual Information Before Alignement: %.5f" mi(subx |> convertAndPermute, suby |> convertAndPermute)

src_idx, trg_idx = buildSeedDictionary(subx, suby)

@printf "Mutual Information Before Alignement: %.5f" mi(X[:, src_idx] |> convertAndPermute, Y[:,trg_idx] |> convertAndPermute, base=exp(1))



W, src_idx, trg_idx = main(X, Y, src_idx, trg_idx, validation)
@printf "Mutual Information After Alignement: %.5f" mi(X[:, src_idx] |> convertAndPermute, Y[:,trg_idx] |> convertAndPermute, base=exp(1))
