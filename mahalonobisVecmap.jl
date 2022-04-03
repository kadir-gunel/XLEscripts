using LinearAlgebra
using Statistics
using OhMyREPL
using XLEs
using Distances





S, T, valfile = EmbeddingData() |> readData;

srcV, X = S[1], S[2];
trgV, Y = T[1], T[2];

src_w2i, trg_w2i = map(word2idx, [srcV, trgV]);
validation = readValidation(valfile, src_w2i, trg_w2i);


rng = 1:Int(4e3);
subX = X[:, rng];
subY = Y[:, rng];



# positive point wise mutual information
