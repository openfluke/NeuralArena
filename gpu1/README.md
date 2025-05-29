# MNIST Activation Function Performance

This document summarizes the performance of various activation functions tested on the MNIST dataset with 10,000 test samples. The tests compare CPU and GPU execution times across different data types (float32, int32, uint32) over 1,000 iterations. The results include execution times, speedup (CPU time / GPU time), and verification of output matching between CPU and GPU.

> **Note**: A warning was observed during testing: `[wgpu] [Warn] Detected skylake derivative running on mesa i915. Clears to srgb textures will use manual shader clears.`

## Performance Results

### Float32

Testing activation functions with float32 data type.

| Activation | CPU Time     | GPU Time     | Speedup | Match |
| ---------- | ------------ | ------------ | ------- | ----- |
| linear     | 5.212794406s | 910.226618ms | 5.73x   | ✅    |
| relu       | 5.534079058s | 836.45741ms  | 6.62x   | ✅    |
| leaky_relu | 5.361402801s | 820.887026ms | 6.53x   | ✅    |
| elu        | 5.533879491s | 843.0522ms   | 6.56x   | ✅    |
| swish      | 5.401623632s | 865.415472ms | 6.24x   | ✅    |
| gelu       | 5.375120811s | 859.863697ms | 6.25x   | ✅    |
| tanh       | 5.293378546s | 797.211397ms | 6.64x   | ✅    |
| softmax    | 5.425525131s | 1.081928745s | 5.01x   | ✅    |

### Int32

Testing activation functions with int32 data type.

| Activation | CPU Time     | GPU Time     | Speedup | Match |
| ---------- | ------------ | ------------ | ------- | ----- |
| linear     | 5.154817162s | 820.523572ms | 6.28x   | ✅    |
| relu       | 5.180602845s | 815.494818ms | 6.35x   | ✅    |
| leaky_relu | 5.038878807s | 838.280008ms | 6.01x   | ✅    |
| elu        | 5.150231543s | 796.708062ms | 6.46x   | ✅    |
| swish      | 5.124261151s | 961.447786ms | 5.33x   | ✅    |
| gelu       | 7.825140508s | 1.20540534s  | 6.49x   | ✅    |
| tanh       | 7.917425032s | 1.253730796s | 6.32x   | ✅    |
| softmax    | 6.008736428s | 1.170632946s | 5.13x   | ✅    |

### Uint32

Testing activation functions with uint32 data type.

| Activation | CPU Time     | GPU Time     | Speedup | Match |
| ---------- | ------------ | ------------ | ------- | ----- |
| linear     | 5.834971077s | 918.112722ms | 6.36x   | ✅    |
| relu       | 5.184863425s | 913.515267ms | 5.68x   | ✅    |
| leaky_relu | 5.148608208s | 809.517272ms | 6.36x   | ✅    |
| elu        | 5.348548594s | 827.624704ms | 6.46x   | ✅    |
| swish      | 5.267901806s | 821.308109ms | 6.41x   | ✅    |
| gelu       | 5.552626835s | 915.125859ms | 6.07x   | ✅    |
| tanh       | 5.342581747s | 807.581801ms | 6.62x   | ✅    |
| softmax    | 5.382638925s | 865.591132ms | 6.22x   | ✅    |

## Activation Function Compatibility

All tested activation functions are compatible with float32, int32, and uint32 data types.

| Activation | float32 | int32 | uint32 |
| ---------- | ------- | ----- | ------ |
| linear     | ✅      | ✅    | ✅     |
| relu       | ✅      | ✅    | ✅     |
| leaky_relu | ✅      | ✅    | ✅     |
| elu        | ✅      | ✅    | ✅     |
| swish      | ✅      | ✅    | ✅     |
| gelu       | ✅      | ✅    | ✅     |
| tanh       | ✅      | ✅    | ✅     |
| softmax    | ✅      | ✅    | ✅     |

## Summary

- **Dataset**: MNIST (10,000 test samples)
- **Iterations**: 1,000
- **Data Types**: float32, int32, uint32
- **Observations**:
  - GPU consistently outperforms CPU, with speedups ranging from 5.01x to 6.64x.
  - All activation functions produce matching outputs between CPU and GPU (`Match=true`).
  - The `tanh` and `softmax` functions show slightly higher GPU times for int32 compared to other data types.
  - The warning about skylake derivative and mesa i915 may indicate potential optimization opportunities for specific hardware configurations.
