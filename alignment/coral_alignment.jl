cd(@__DIR__)

using XLEs
using LinearAlgebra
using Statistics
using StatsBase
using CUDA

function coral(X::AbstractArray, Y::AbstractArray)
    covX = cov(X, dims=2) + I(size(X, 1))
    covY = cov(Y, dims=2) + I(size(Y, 1))

    A_coral = covX^(-0.5) * covY^(0.5)

    X̂ = Float32.(A_coral' * X)
    return X̂
end


function whitenedSimilarity(x::AbstractArray)
    F = svd(x)
    return F.v * diagm(sqrt.(F.s)) * F.vt
end

function buildSeedDictionary3(subx::CuMatrix, suby::CuMatrix, subx̂::CuMatrix; sim_size::Int64=4000, k::Int64=10)

    xsim = subx' * subx̂
    ysim = suby' * subx̂

    sort!(ysim, dims=1)
    sort!(xsim, dims=1)
    
    xsim, ysim = map(normalizeEmbedding, [xsim, ysim])

    sim = CuMatrix{Float32}(undef, sim_size, sim_size);
    CUBLAS.gemm!('T', 'N', cu(1.), xsim, ysim, cu(0.), sim);

    sim = XLEs.csls(sim, k=k)

    src_idx = vec(vcat(collect(1:sim_size), permutedims(getindex.(argmax(sim, dims=1), 1))|> Array))
    trg_idx = vec(vcat((getindex.(argmax(sim, dims=2), 2))|> Array, collect(1:sim_size)))

    return src_idx, trg_idx

end

function buildSeedDictionary2(X::T, Y::T; sim_size::Int64=4000, k::Int64=10) where {T}
    # sims = map(cudaCorrelationMatrix, [X, Y])
    xsim = XLEs.cudaCorrelationMatrix(X, sim_size=sim_size)
    # xsim = X' * X
    ysim = Y' * Y 
    sort!(ysim, dims=1)
    sort!(xsim, dims=1)
    # map(sim -> sort!(sim, dims=1), sims);
    xsim, ysim = map(normalizeEmbedding, [xsim, ysim])

    sim = CuMatrix{Float32}(undef, sim_size, sim_size);
    CUBLAS.gemm!('T', 'N', cu(1.), xsim, ysim, cu(0.), sim);

    # sim = xsim' * ysim; # actually this is still the cosine similarity from X -> Z.
    # csls_neighborhood = 10

    sim = XLEs.csls(sim, k=k)

    src_idx = vec(vcat(collect(1:sim_size), permutedims(getindex.(argmax(sim, dims=1), 1))|> Array))
    trg_idx = vec(vcat((getindex.(argmax(sim, dims=2), 2))|> Array, collect(1:sim_size)))

    return src_idx, trg_idx
end

langs = ["it", "es", "fi", "de"]
printstyled("MMD X and X̂: ", MMD(X, WY), color=:green )
src = "en"
# trg = "fi"

# muse = "/run/media/phd/PhD/MUSE/data/fasttextBins/"
vecmap = "/run/media/phd/PhD/vecmap/data/embeddings/" 


src = vecmap * "Glove"
trg = vecmap * "it"


Es = map(readBinaryEmbeddings, [src, trg])

#     printstyled("MMD for en - $trg pair :", MMD(subx, suby), color=:blue, "\n")])

srcV, X = Es[1]
trgV, Y = Es[2]

X, Y = map(permutedims, [X, Y])
# X, Y = map(normalizeEmbedding, [X, Y])
X, Y = map(XLEs.unit, [X, Y])


X = coral(X, Y)


X, Y = map(cu, [X, Y]);

val = "/run/media/phd/PhD/vecmap/data/dictionaries/en-it.test.txt"

src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
validation = readValidation(val, src_w2i, trg_w2i);


rng = Int(4e3)
x = X[:, rng]  #|> Matrix
y = Y[:, rng]  #|> Matrix


# @time src_idx, trg_idx = XLEs.buildSeedDictionary0(subx, suby)
# @time src_idx, trg_idx = XLEs.buildRBFSeedDictionary(x, y, threshold=0.6, k=40);
# @time src_idx, trg_idx = XLEs.buildSeedDictionary(x, y, sim_size=length(rng));
@time src_idx, trg_idx = buildSeedDictionary2(x, y, sim_size=length(rng));
# @time src_idx3, trg_idx3 = XLEs.buildCosineDictionary(x |> Matrix, y |> Matrix)
# @time src_idx3, trg_idx3 = XLEs.buildMahalanobisDictionary(x |> Matrix, y |> Matrix)
# @time src_idx, trg_dix = XLEs.mahalanobisGPU(subx, suby);
@time W, src_idx, trg_idx = XLEs.main(X, Y, src_idx, trg_idx, validation, src_size=Int(30e3), trg_size=Int(30e3));

kacc1, sims1 = validate(W * X |> XLEs.unit , Y |> XLEs.unit, validation)
kacc2, sims2 = validateCSLS(W * X |> XLEs.unit, Y |> XLEs.unit, validation)





using MATLAB

x = rand(40, 10)
y = rand(40, 10)