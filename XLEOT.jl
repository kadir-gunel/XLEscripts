cd(@__DIR__)
using OhMyREPL
using LinearAlgebra
using OptimalTransport
using XLEs

"""
Gromov wasserstein Matrix
P stands for Pivot Embedding Space (EN)
"""
function GWmatrix(P)
    n, d = size(P)
    N2 = .5 * norm(emb, dims=1)

end
