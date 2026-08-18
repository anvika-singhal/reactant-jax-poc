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

loss_fn = ns["loss"]

# Initialize weights as ConcreteRArrays -- stay on device throughout
W1 = Reactant.to_rarray(Float32.(0.1 * randn(4, 8)))
b1 = Reactant.to_rarray(zeros(Float32, 8))
W2 = Reactant.to_rarray(Float32.(0.1 * randn(8, 2)))
b2 = Reactant.to_rarray(zeros(Float32, 2))
params = (W1, b1, W2, b2)

x = Reactant.to_rarray(Float32.(randn(4)))
y = Reactant.to_rarray(Float32[1.0, 0.0])

lr = 0.01f0

# Single compiled function: gradient + update in one step
# No Array conversions, no boundary crossing, compiled once
function train_step(params, x, y)
    grads = Enzyme.gradient(Reverse, loss_fn, params, x, y)
    new_params = map((p, g) -> p .- lr .* g, params, grads[1])
    return new_params
end

println("Compiling train_step...")
compiled_step = @compile train_step(params, x, y)
println("Compiled. Training...")
println("=" ^ 55)

for step in 1:20
    global params = compiled_step(params, x, y)
    current_loss = @jit loss_fn(params, x, y)
    println("Step $step  loss = $(convert(Float32, current_loss))")
end

println("=" ^ 55)
println("Done.")
