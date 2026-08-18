using Reactant
using PythonCall

jax = pyimport("jax")

# Define JAX functions in Python via pyexec
# pyexec runs Python code in a given namespace
ns = pydict()
pyexec("""
import jax.numpy as jnp

def square(x):
    return jnp.multiply(x, x)

def square_sum(x):
    return jnp.sum(jnp.multiply(x, x))
""", ns)

square = ns["square"]
square_sum = ns["square_sum"]

x = Reactant.to_rarray(Float32[1.0, 2.0, 3.0, 4.0])

# Forward pass
println("Testing forward pass...")
result = Reactant.@jit square(x)
println("Forward result: ", result)
println("Expected:       [1. 4. 9. 16.]")

# Gradient via Enzyme through Reactant
println("\nTesting gradient...")
using Enzyme
grad = Reactant.@jit Enzyme.gradient(Reverse, square_sum, x)
println("Gradient: ", grad)
println("Expected: [2. 4. 6. 8.]")
