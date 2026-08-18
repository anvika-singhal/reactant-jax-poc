
# reactant-jax-poc

Proof-of-concept showing bidirectional interoperability between JAX (Python) and Julia via Reactant.jl, with automatic differentiation in both directions.

## What this demonstrates

```
Julia function → Reactant.jl → StableHLO → hlo_call → jax.grad
JAX function   → PythonCall  → StableHLO → Reactant.@jit → Enzyme.gradient
JAX MLP        → Reactant.@jit → Enzyme.gradient → Julia training loop
```

All three verified working with correct gradients. See results below.


##  Environment
| Component | Version |
|---|---|
| Julia | 1.12.7 (juliaup) |
| Reactant.jl | latest |
| PythonCall.jl | 0.9.35 |
| Enzyme.jl | latest |
| NPZ.jl | latest |
| Python | 3.11.15 (conda) |
| enzyme-ad | 0.0.15 |
| JAX / jaxlib | 0.10.2 |
| OS | Ubuntu 24.04 x86\_64 |

## Setup

### 1. Julia

Install Julia via juliaup (no root needed):

```bash
curl -fsSL https://install.julialang.org | sh
source ~/.bashrc
```

Install Julia dependencies:

```bash
julia --project=. -e 'using Pkg; Pkg.add(["Reactant", "PythonCall", "Enzyme", "NPZ"])'
```

### 2. Python

enzyme-ad requires Python 3.11 and Linux x86_64 or macOS ARM64. System Python is often 3.12, so use conda:

```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
conda create -y -n reactant-poc python=3.11
conda activate reactant-poc
pip install "enzyme-ad==0.0.15" jax
```

### 3. Critical environment variable

PythonCall creates its own Python environment (CondaPkg) and ignores the conda env above unless told otherwise. Set this before every Julia session:

```bash
export JULIA_PYTHONCALL_EXE=/home/anvika/miniconda3/envs/reactant-poc/bin/python
```

## Scripts

### Direction 1: Julia → JAX (`export_fn.jl` + `test_poc.py`)

Compile a Julia function to StableHLO via Reactant, execute and differentiate it from Python via EnzymeJAX.

```bash
julia --project=. export_fn.jl   # exports my_square_0.mlir, my_square.py
python test_poc.py
```

Output:

```
Forward result: [ 1.  4.  9. 16.]
Gradients:      [2. 4. 6. 8.]
```

**Known issue:** Reactant emits `tf.aliasing_output` in the StableHLO, which breaks EnzymeJAX's VJP. `test_poc.py` patches the MLIR string before calling `hlo_call`. This is fragile for complex functions and needs a proper fix upstream in Reactant.jl or Enzyme-JAX or something similar.

### Direction 2: JAX → Julia (`jax_to_julia.jl`)

Define a JAX function in Python, call and differentiate it from Julia via `Reactant.@jit` + `Enzyme.gradient`.

```bash
julia --project=. jax_to_julia.jl
```

Output:

```
Forward result: ConcretePJRTArray Float32[1.0, 4.0, 9.0, 16.0]
Gradient:       (Float32[2.0, 4.0, 6.0, 8.0],)
```

### Direction 3: JAX MLP training loop (`mlp_train.jl`)

2-layer MLP defined in JAX (input:4 → hidden:8 tanh → output:2), weights held as Reactant arrays in Julia, trained via Enzyme gradients from Julia.

```bash
julia --project=. mlp_train.jl
```

Output:

```
Step 1   loss = 0.514007
Step 5   loss = 0.456826
Step 10  loss = 0.394305
Step 15  loss = 0.340103
Step 20  loss = 0.292897
```

## Platform constraints

Fully supported on Linux x86_64, macOS ARM64 with enzyme-ad wheel available. No enzyme-ad Windows wheels, WSL2 required. No enzyme-ad wheel for macOS x86-64.

## Open issues

1.  **Aliasing output (Julia -> JAX):** Reactant's `tf.aliasing_output` convention breaks EnzymeJAX VJP. Patched with string replacement in `test_poc.py`. Needs upstream fix.
2.  **Weight update boundary (MLP training):** Weights are pulled to Julia, updated, and pushed back to Reactant every step. Probably triggers recompilation. Should keep weights on-device with an in-place update compiled inside `@jit`.
3.  **enzyme-ad Python 3.11 pin:** Every release pins to a specific Python version. No conda package, no Windows wheels.

