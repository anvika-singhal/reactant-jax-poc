module @reactant_my_square attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<4xf32> {tf.aliasing_output = 1 : i32}) -> (tensor<4xf32>, tensor<4xf32>) {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
    return %0, %arg0 : tensor<4xf32>, tensor<4xf32>
  }
}