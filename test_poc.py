import jax
import jax.numpy as jnp
from enzyme_ad.jax import hlo_call

with open("my_square_0.mlir", "r") as f:
    hlo_code = f.read()

hlo_code = hlo_code.replace(
    'func.func @main(%arg0: tensor<4xf32> {tf.aliasing_output = 1 : i32}) -> (tensor<4xf32>, tensor<4xf32>)',
    'func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32>'
)
hlo_code = hlo_code.replace(
    'return %0, %arg0 : tensor<4xf32>, tensor<4xf32>',
    'return %0 : tensor<4xf32>'
)

@jax.jit
def run_square(x):
    return hlo_call(x, source=hlo_code)[0]  # index into list

x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)

print("Forward result:", run_square(x))
print("Expected:      [1. 4. 9. 16.]")

grad_fn = jax.grad(lambda x: run_square(x).sum())
grads = grad_fn(x)
print("Gradients:", grads)
print("Expected:  [2. 4. 6. 8.]")
