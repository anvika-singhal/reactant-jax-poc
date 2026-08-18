using Reactant, NPZ

# Simple function: elementwise square
# Derivative of x^2 is 2x, so we can verify gradients easily
function my_square(x)
    return x .* x
end

# Create a concrete Reactant array as example input
x = Reactant.to_rarray(Float32[1.0, 2.0, 3.0, 4.0])

# Compile to StableHLO and export to EnzymeJAX format
# Writes: my_square.mlir, my_square_0_inputs.npz, my_square.py
python_file_path = Reactant.Serialization.export_to_enzymejax(
    my_square, x;
    output_dir=".",
    function_name="my_square"
)

println("Exported to: ", python_file_path)
