using Reactant, PythonCall, Enzyme

ns = pydict()
pyexec("""
import jax.numpy as jnp

def forward(params, x):
    W1, b1, W2, b2 = params
    h = jnp.tanh(x @ W1 + b1)
    return h @ W2 + b2

def loss(params, x, y):
    pred = forward(params, x)
    return jnp.mean((pred - y) ** 2)
""", ns)

forward = ns["forward"]
loss_fn = ns["loss"]

W1 = Reactant.to_rarray(Float32.(0.1 * randn(4, 8)))
b1 = Reactant.to_rarray(zeros(Float32, 8))
W2 = Reactant.to_rarray(Float32.(0.1 * randn(8, 2)))
b2 = Reactant.to_rarray(zeros(Float32, 2))
params = (W1, b1, W2, b2)

x = Reactant.to_rarray(Float32.(randn(4)))
y = Reactant.to_rarray(Float32[1.0, 0.0])

lr = 0.01f0
n_steps = 20

println("Training 2-layer JAX MLP from Julia via Reactant + Enzyme")
println("=" ^ 55)

for step in 1:n_steps
    grads = Reactant.@jit Enzyme.gradient(Reverse, loss_fn, params, x, y)
    param_grads = grads[1]

    global params = map(params, param_grads) do p, g
        Reactant.to_rarray(Array(p) .- lr .* Array(g))
    end

    current_loss = Reactant.@jit loss_fn(params, x, y)
    # ConcretePJRTNumber needs convert, not Array
    println("Step $step  loss = $(convert(Float32, current_loss))")
end

println("=" ^ 55)
println("Done. Loss should be decreasing.")
