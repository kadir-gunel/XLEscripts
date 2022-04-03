cd(@__DIR__)

using LinearAlgebra
using XLEs
using CUDA
using BSON
using OhMyREPL

relu(E) = max.(0, E |> Array) |> cu;


function mybuildSeedDictionary(X::T, Y::T; sim_size::Int64=4000) where {T}
   # sims = map(cudaCorrelationMatrix, [X, Y])
    xsim = XLEs.cudaCorrelationMatrix(X, sim_size=sim_size) |> relu;
    ysim = XLEs.cudaCorrelationMatrix(Y, sim_size=sim_size) |> relu;
    sort!(ysim, dims=1)
    sort!(xsim, dims=1)
    # map(sim -> sort!(sim, dims=1), sims);
    xsim, ysim = map(normalizeEmbedding, [xsim, ysim])
    sim = (xsim' * ysim); # actually this is still the cosine similarity from X -> Z.
    # csls_neighborhood = 10
    sim = XLEs.csls(sim)

    src_idx = vec(vcat(collect(1:sim_size), permutedims(getindex.(argmax(sim, dims=1), 1))|> Array))
    trg_idx = vec(vcat((getindex.(argmax(sim, dims=2),2))|> Array, collect(1:sim_size)))

    return src_idx, trg_idx
end;



datadir = "../XLEs/data/exp_raw/";
root, folders, files = first(walkdir(datadir * "embeddings/"));

srcfile = root * "en"
trgfile = root * "es"
valfile = datadir * "dictionaries/en-es.test.txt";


srcV, X = readBinaryEmbeddings(srcfile);
trgV, Y = readBinaryEmbeddings(trgfile);


src_w2i = word2idx(srcV);
trg_w2i = word2idx(trgV);
validation = readValidation(valfile, src_w2i, trg_w2i);


X, Y = map(normalizeEmbedding, [X, Y]);

X, Y = map(cu, [X, Y]);
rng = 1:Int(4e3);
subX = X[:, rng];
subY = Y[:, rng];

src_idx, trg_idx = buildSeedDictionary(subX, subY);


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

XW, YW = advancedMapping(permutedims(W * X), permutedims(Y), src_idx, trg_idx);

acc, sims = validate(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)
acc, sims= validateCSLS(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)


condX = cond(XW * permutedims(XW))

condY = cond(YW * permutedims(YW))

condY0 = cond(Y * permutedims(Y))

end
