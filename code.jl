using Metalhead: VGG19, preprocess, load
using Flux: @epochs
using Statistics, PyCall, Flux.Tracker
using Images: RGB

np = pyimport("numpy")
Image = pyimport("PIL.image.jpeg")

function deprocess_and_pillow(img)
    μ, σ = ([0.485])