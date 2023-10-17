cd(@__DIR__)

using OhMyREPL
using LinearAlgebra
using Statistics

ENV["PYTHON"] = "/home/phd/anaconda3/envs/torch2.0/bin/python"
using PyCall

using CUDA


x = rand(300, 1000) 
y = rand(300, 1000)
kernel = "multiscale"

function MMD(x::T, y::T; kernel::String="multiscale") where T
    atype = typeof(x)
    xx = x' * x
    yy = y' * y 
    zz = x' * y

    rx, ry = map(x -> repeat(diag(x), outer=(1,size(x, 2))), [xx, yy])

    dxx = rx + rx' - 2xx
    dyy = ry + ry' - 2yy
    dxy = rx + ry' - 2zz

    if isequal(string(atype), "CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}")
        XX, YY, XY = map(xx -> cu(zeros(size(xx))), [xx, xx, xx])
    else 
        XX, YY, XY = map(xx -> zeros(size(xx)), [xx, xx, xx])
    end

    if isequal(kernel, "multiscale")
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for b in bandwidth_range
            XX += b^2 * (b^2 .+ dxx).^(-1)
            YY += b^2 * (b^2 .+ dyy).^(-1)
            XY += b^2 * (b^2 .+ dxy).^(-1)
        end
    end

    if isequal(kernel, "rbf")
        bandwidth_range = [10, 15, 20, 50]
        for b in bandwidth_range
            XX += exp.(-.5 * dxx / b)
            YY += exp.(-.5 * dyy / b)
            XY += exp.(-.5 * dxy / b)
        end
    end

    return mean(XX + YY - 2 .* XY) # this is very similar to CSLS!!
end


MMD(x, y)

