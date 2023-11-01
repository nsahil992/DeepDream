using Metalhead: VGG19, preprocess, load
using Flux: @epochs
using Statistics, PyCall, Flux.Tracker
using Images: RGB

np = pyimport("numpy")
Image = pyimport("PIL.image.jpeg")

function deprocess_and_pillow(img)
    μ, σ = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    rbg = cat(collect(map(x -> (img[:, :, x, 1] .* σ[x]) .+  μ[x])))