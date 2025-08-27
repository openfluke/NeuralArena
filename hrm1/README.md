# Activation Function Benchmark

This Go program benchmarks the performance of common neural network activation functions and their derivatives across various data types, comparing pre-conversion (casting values before benchmarking) and inline conversion (casting during inference). The functions are tested on 10 million input values ranging from -10 to 10.

## Purpose

The program evaluates the computational efficiency of activation functions (ReLU, Sigmoid, Tanh, LeakyReLU, ELU, Linear, Softmax Derivative) and their derivatives across different numeric data types (int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64). It compares two approaches: pre-converting values to the target type versus casting inline during inference. The results help identify optimal functions and data types for machine learning applications.

## How to Run

1. Ensure you have Go installed (version 1.16 or later recommended).
2. Save the code in a file named `main.go`.
3. Run the program:
   ```bash
   go run main.go
   ```
4. The program will output two tables of execution times (in milliseconds) for each activation function and derivative across all data types: one for pre-conversion and one for inline conversion during inference.

## Benchmark Results

### Pre-Conversion

Values are cast to the target data type before benchmarking.

| Type    | ReLU        | Sigmoid     | Tanh        | LeakyReLU   | ELU         | Linear      | SoftmaxDer  | dReLU       | dSigmoid    | dTanh       | dLeakyReLU  | dELU        | dLinear     |
| ------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| int8    | 25.412091ms | 60.658226ms | 67.711636ms | 19.793782ms | 35.079385ms | 14.272313ms | 14.391836ms | 14.94349ms  | 67.991452ms | 71.664869ms | 14.607044ms | 35.8806ms   | 13.858769ms |
| int16   | 23.838663ms | 60.387045ms | 65.752542ms | 14.899805ms | 31.928167ms | 14.73232ms  | 16.374855ms | 14.709659ms | 63.370456ms | 70.897772ms | 13.976076ms | 33.49964ms  | 14.55491ms  |
| int32   | 24.976769ms | 61.87885ms  | 67.482383ms | 17.062296ms | 36.095567ms | 15.503171ms | 15.123958ms | 14.885633ms | 63.964129ms | 73.908681ms | 17.393869ms | 32.323749ms | 13.803301ms |
| int64   | 25.478355ms | 61.413283ms | 68.084912ms | 13.016745ms | 34.114867ms | 13.593875ms | 12.442221ms | 12.92534ms  | 71.769088ms | 71.119903ms | 16.599328ms | 32.536527ms | 14.597496ms |
| uint8   | 24.546673ms | 59.858549ms | 42.259926ms | 15.728503ms | 16.397763ms | 16.343353ms | 13.467769ms | 13.6675ms   | 63.190731ms | 46.3579ms   | 12.679318ms | 14.31379ms  | 14.354192ms |
| uint16  | 25.375449ms | 57.858816ms | 42.754356ms | 14.69304ms  | 14.768587ms | 17.397893ms | 13.959288ms | 14.994249ms | 59.266639ms | 46.92734ms  | 14.149299ms | 19.314308ms | 13.890204ms |
| uint32  | 25.449954ms | 57.284973ms | 42.181679ms | 14.955632ms | 19.340583ms | 16.450015ms | 18.087036ms | 15.301083ms | 58.741563ms | 46.982989ms | 14.630804ms | 18.617918ms | 13.92229ms  |
| uint64  | 25.110697ms | 56.632177ms | 41.450698ms | 14.193335ms | 14.736416ms | 12.845329ms | 14.212904ms | 14.101583ms | 58.244014ms | 45.679707ms | 15.678335ms | 21.498615ms | 13.016348ms |
| float32 | 25.156132ms | 63.05245ms  | 70.774683ms | 14.307241ms | 34.312946ms | 10.73345ms  | 12.125348ms | 12.687031ms | 65.720745ms | 76.789584ms | 17.306593ms | 34.786818ms | 13.418259ms |
| float64 | 25.864306ms | 62.748425ms | 70.586033ms | 16.09706ms  | 35.28562ms  | 11.370099ms | 12.0622ms   | 14.614079ms | 66.345589ms | 76.597935ms | 14.057118ms | 34.495111ms | 13.678738ms |

### Conversion During Inference

Values are cast to the target data type inline during each function call.

| Type    | ReLU        | Sigmoid      | Tanh         | LeakyReLU   | ELU         | Linear      | SoftmaxDer  | dReLU       | dSigmoid     | dTanh        | dLeakyReLU  | dELU        | dLinear     |
| ------- | ----------- | ------------ | ------------ | ----------- | ----------- | ----------- | ----------- | ----------- | ------------ | ------------ | ----------- | ----------- | ----------- |
| int8    | 30.977567ms | 69.55644ms   | 82.01879ms   | 20.881059ms | 41.126618ms | 20.436228ms | 20.317502ms | 22.37918ms  | 78.265646ms  | 84.996813ms  | 22.346079ms | 40.001386ms | 19.390992ms |
| int16   | 31.045556ms | 70.031811ms  | 81.647059ms  | 22.415169ms | 41.446072ms | 19.364444ms | 20.071321ms | 22.786777ms | 77.686173ms  | 85.689607ms  | 24.114857ms | 39.956894ms | 21.113119ms |
| int32   | 30.947296ms | 68.873168ms  | 82.009432ms  | 22.912074ms | 40.930893ms | 19.09654ms  | 19.165896ms | 24.579338ms | 77.382538ms  | 84.390401ms  | 23.172766ms | 39.771593ms | 20.217815ms |
| int64   | 30.786923ms | 68.789111ms  | 80.709501ms  | 22.035244ms | 41.410049ms | 19.528464ms | 20.399024ms | 24.813099ms | 77.483296ms  | 84.709282ms  | 21.831962ms | 40.942888ms | 20.925509ms |
| uint8   | 31.467018ms | 69.056277ms  | 54.281804ms  | 22.689661ms | 24.784141ms | 20.060463ms | 21.104873ms | 23.737696ms | 77.955289ms  | 55.703816ms  | 24.509797ms | 25.088259ms | 20.759441ms |
| uint16  | 30.807271ms | 63.823936ms  | 56.973936ms  | 22.838697ms | 24.470969ms | 20.404347ms | 21.422694ms | 25.572144ms | 72.519972ms  | 56.626735ms  | 24.323658ms | 23.855528ms | 20.181817ms |
| uint32  | 32.017664ms | 64.633152ms  | 54.125214ms  | 23.398084ms | 22.919541ms | 19.287181ms | 19.284295ms | 22.891681ms | 72.403439ms  | 55.62676ms   | 23.117146ms | 24.275645ms | 20.945912ms |
| uint64  | 33.265013ms | 66.374524ms  | 56.241839ms  | 28.859551ms | 27.87883ms  | 23.540704ms | 24.997503ms | 27.634973ms | 76.274953ms  | 58.288166ms  | 28.129569ms | 28.176527ms | 26.473873ms |
| float32 | 33.628693ms | 240.860095ms | 283.361882ms | 21.098141ms | 42.325258ms | 19.063036ms | 22.816036ms | 23.660226ms | 280.994282ms | 299.675453ms | 22.808835ms | 44.830966ms | 21.675256ms |
| float64 | 29.266017ms | 64.308086ms  | 74.978551ms  | 22.387369ms | 40.535291ms | 20.477095ms | 19.021347ms | 22.640015ms | 74.162705ms  | 79.660136ms  | 23.996459ms | 40.582659ms | 19.535162ms |

## Performance Observations

- **Pre-Conversion vs. Inline Conversion**:
  - Inline conversion increases execution times by ~20-30% for most functions and data types (e.g., ReLU on int8: 25.41ms pre-conversion vs. 30.98ms inline) due to casting overhead in the benchmark loop.
  - **float32** inline conversion shows extreme slowdowns for **Sigmoid** (63.05ms to 240.86ms), **Tanh** (70.77ms to 283.36ms), and **dTanh** (76.79ms to 299.68ms), possibly due to floating-point precision issues or compiler inefficiencies.
- **Fastest Functions**:
  - **Linear** (10.73-17.40ms pre-conversion, 19.06-26.47ms inline) and **dLinear** (13.02-14.60ms pre-conversion, 19.39-26.47ms inline) are fastest due to trivial operations.
  - **ReLU** and **dReLU** are also fast (12.69-25.86ms pre-conversion, 22.38-33.63ms inline) with simple conditionals.
- **Slowest Functions**:
  - **Tanh** (41.45-70.77ms pre-conversion, 54.13-283.36ms inline) and **dTanh** (45.68-76.79ms pre-conversion, 55.63-299.68ms inline) are slowest due to trigonometric and exponential operations.
  - **Sigmoid** and **dSigmoid** are slow (56.63-66.35ms pre-conversion, 63.82-280.99ms inline) due to exponentiation and division.
  - **ELU** and **dELU** are slower than ReLU (14.74-36.10ms pre-conversion, 22.92-44.83ms inline) due to exponential calculations for negative inputs.
- **Data Type Effects**:
  - **Unsigned integers** (uint8, uint16, uint32, uint64) excel for **Tanh** (41.45-42.75ms pre-conversion vs. 65.75-70.77ms for signed/floating) and **ELU** (14.74-19.34ms pre-conversion vs. 31.93-36.10ms), likely due to simpler bit operations.
  - **float32/float64** have higher latency for complex functions (e.g., Tanh at 70.59-70.77ms pre-conversion, up to 283.36ms inline) due to floating-point overhead.
  - **int8/int16/int32/int64** are consistent, with **LeakyReLU** fastest on int64 (13.02ms pre-conversion).
- **Derivatives**:
  - **dReLU**, **dLeakyReLU**, and **dLinear** are faster than their base functions, returning constants (0, 1, or 0.01).
  - **dSigmoid** and **dTanh** are slower, requiring base function computation.
- **Anomalies**:
  - **uint8/uint16** have low **ELU** times (14.74-16.40ms pre-conversion, 23.86-25.09ms inline) vs. others (31.93-36.10ms pre-conversion), possibly due to casting optimizations.
  - **float32** inline conversion for **Sigmoid**, **Tanh**, and **dTanh** shows outliers (240-299ms), suggesting issues with Go's float32 handling.
  - **uint64** inline conversion shows variance (e.g., LeakyReLU at 28.86ms, dLinear at 26.47ms), possibly due to 64-bit unsigned integer handling.
- **Key Insights**:
  - Pre-conversion is significantly faster, ideal for performance-critical neural network inference.
  - Simple functions (Linear, ReLU) scale well across data types and conversion methods.
  - Complex operations (exponentiation, division) in Sigmoid, Tanh, and ELU cause slowdowns, especially with inline conversion.
  - Unsigned integers (uint8/uint16) are efficient for Tanh and ELU, but floating-point types ensure precision in neural networks.
  - Extreme float32 inline conversion times warrant further investigation into Go's compiler or runtime.

## Code Overview

The Go program defines activation functions and their derivatives, generates 10 million test values in [-10, 10], and benchmarks them using two methods: pre-conversion (`castAndTime`) and inline conversion (`castAndTimeInline`). Results are printed in formatted tables.

### Key Components

- **generateValues**: Creates 10 million evenly spaced values.
- **Activation Functions**: Use Go's `math` package for operations like `Exp` and `Tanh`.
- **Derivatives**: Compute gradients of activation functions.
- **benchmark**: Measures execution time over all values.
- **castAndTime**: Pre-converts values to the target type and benchmarks.
- **castAndTimeInline**: Casts values inline during benchmarking.
- **main**: Runs both benchmark methods and formats output.

## Limitations

- Results depend on hardware and Go runtime, varying across systems.
- Inline conversion introduces significant overhead, which may not reflect optimized production code.
- Only single-precision (float32) and double-precision (float64) floating-point types are tested; formats like bfloat16 are not included.
- The input range [-10, 10] may miss edge cases for some functions.

## Future Improvements

- Test additional data types (e.g., bfloat16, half-precision).
- Parallelize benchmarks using Go routines.
- Include more activation functions (e.g., Swish, GELU).
- Investigate float32 inline conversion outliers.
- Analyze cache effects and memory usage for large inputs.
