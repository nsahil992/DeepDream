using Metalhead: VGG19, preprocess, load
using Flux: @epochs
using Statistics, PyCall, Flux.Tracker
using Images: RGB

np = pyimport("numpy")
Image = pyimport("PIL.image.jpeg")

function deprocess_and_pillow(img)
    μ, σ = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    rgb = cat(collect(map(x -> (img[:, :, x, 1] .* σ[x]) .+  μ[x], 1:3))..., dims=3)
    rgb = np.uint8(np.interp(np.clip(rgb ./ 255, -1, 1), (-1, 1), (0, 255)))
    return
    Image.formarray(rgb).transpose(Image.FLIP_LEFT_RIGHT).rotate(90)
end

model = VGG19().layers[1:11]
loss(x) = mean(model(x))
dloss(x) = Tracker.gradient(loss, x)[1]
function calc_gradient(x)
    g = Tracker.data(dloss(x))
    return g * (mean(1.5 ./ abs.(g)) + le-7)
end