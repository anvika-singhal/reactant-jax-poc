using Reactant, PythonCall, Enzyme

# Define a 2-layer MLP in JAX
# Architecture: input(4) -> hidden(8, tanh) -> output(2)
# forward(params, x) is stateless -- weights passed explicitly
# so Enzyme can differentiate w.r.t. params from Julia
ns = pydict()
pyexec("""
import jax.numpy as jnp
import jax

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

# Initialize weights as Reactant arrays
# W1: (4, 8), b1: (8,), W2: (8, 2), b2: (2,)
W1 = Reactant.to_rarray(Float32.(0.1 * randn(4, 8)))
b1 = Reactant.to_rarray(zeros(Float32, 8))
W2 = Reactant.to_rarray(Float32.(0.1 * randn(8, 2)))
b2 = Reactant.to_rarray(zeros(Float32, 2))
params = (W1, b1, W2, b2)

# Dummy input and target
x = Reactant.to_rarray(Float32.(randn(4)))
y = Reactant.to_rarray(Float32[1.0, 0.0])

# Forward pass
println("Forward pass...")
out = Reactant.@jit forward(params, x)
println("Output: ", out)

# Gradient w.r.t. params
println("\nComputing gradients w.r.t. weights...")
grads = Reactant.@jit Enzyme.gradient(Reverse, loss_fn, params, x, y)
println("dL/dW1 shape: ", size(grads[1][1]))
println("dL/db1 shape: ", size(grads[1][2]))
println("dL/dW2 shape: ", size(grads[1][3]))
println("dL/db2 shape: ", size(grads[1][4]))
println("Gradient norms:")
println("  W1: ", sum(grads[1][1].^2))
println("  W2: ", sum(grads[1][3].^2))
