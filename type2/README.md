# Activation Function Performance

This table compares the performance (in milliseconds) of various activation functions and their derivatives across different data types.

| Type    | ReLU    | Sigmoid | Tanh    | LeakyReLU | ELU     | Linear  | dReLU   | dSigmoid | dTanh   | dLeakyReLU | dELU    | dLinear |
| ------- | ------- | ------- | ------- | --------- | ------- | ------- | ------- | -------- | ------- | ---------- | ------- | ------- |
| int     | 13.86ms | 26.28ms | 26.45ms | 15.40ms   | 23.11ms | 12.17ms | 12.85ms | 27.63ms  | 21.70ms | 12.79ms    | 21.10ms | 12.34ms |
| int8    | 12.48ms | 25.45ms | 27.81ms | 15.03ms   | 22.18ms | 10.55ms | 11.90ms | 26.69ms  | 26.62ms | 11.77ms    | 21.52ms | 11.71ms |
| int16   | 12.98ms | 25.47ms | 24.35ms | 15.14ms   | 23.17ms | 10.34ms | 11.90ms | 29.72ms  | 23.40ms | 11.36ms    | 22.86ms | 11.16ms |
| int32   | 13.07ms | 27.60ms | 24.22ms | 14.95ms   | 22.79ms | 10.82ms | 11.91ms | 28.49ms  | 26.25ms | 12.00ms    | 22.72ms | 11.35ms |
| int64   | 14.97ms | 26.96ms | 25.09ms | 17.35ms   | 21.05ms | 12.59ms | 13.35ms | 27.81ms  | 24.70ms | 14.66ms    | 21.06ms | 12.84ms |
| uint    | 16.12ms | 24.04ms | 23.75ms | 14.43ms   | 19.35ms | 12.74ms | 14.52ms | 29.27ms  | 23.33ms | 14.56ms    | 20.01ms | 14.25ms |
| uint8   | 14.72ms | 23.18ms | 22.10ms | 12.61ms   | 17.15ms | 10.85ms | 11.48ms | 24.73ms  | 21.46ms | 12.33ms    | 17.31ms | 10.97ms |
| uint16  | 13.53ms | 23.12ms | 22.27ms | 13.57ms   | 19.37ms | 10.72ms | 11.75ms | 26.27ms  | 22.39ms | 12.20ms    | 20.46ms | 11.41ms |
| uint32  | 12.98ms | 25.69ms | 27.47ms | 14.38ms   | 18.38ms | 12.28ms | 13.22ms | 26.25ms  | 23.53ms | 13.97ms    | 18.55ms | 12.21ms |
| uint64  | 14.80ms | 27.52ms | 25.43ms | 15.61ms   | 17.23ms | 13.68ms | 14.51ms | 28.01ms  | 27.44ms | 15.05ms    | 17.93ms | 14.36ms |
| float32 | 13.12ms | 46.95ms | 34.20ms | 16.15ms   | 27.37ms | 18.72ms | 13.98ms | 38.25ms  | 27.12ms | 14.43ms    | 28.23ms | 12.13ms |
| float64 | 14.11ms | 75.90ms | 89.73ms | 16.26ms   | 44.58ms | 12.82ms | 14.52ms | 79.05ms  | 93.66ms | 15.54ms    | 43.61ms | 14.01ms |

# Observations on Activation Function Performance

Below are key observations derived from the performance data of various activation functions and their derivatives across different data types:

## General Trends

- **Linear** is consistently the fastest activation function across most data types, with times ranging from 10.34ms (int16) to 18.72ms (float32). Its derivative (dLinear) also performs well, often close to Linear's performance.
- **Sigmoid** and **Tanh** are among the slowest activation functions, particularly for floating-point types (float32 and float64), with Sigmoid reaching 75.90ms and Tanh 89.73ms for float64. Their derivatives (dSigmoid, dTanh) are similarly slow.
- **ReLU** and **LeakyReLU** offer strong performance, with ReLU being slightly faster in most cases. Both are significantly faster than Sigmoid and Tanh, especially for floating-point types.
- **ELU** tends to be slower than ReLU and LeakyReLU but faster than Sigmoid and Tanh for most data types.

## Data Type Impact

- **Integer types (int, int8, int16, int32, int64)** generally perform better than floating-point types (float32, float64) for most activation functions. For example, ReLU takes 13.86ms for int but 14.11ms for float64.
- **Unsigned integer types (uint, uint8, uint16, uint32, uint64)** show mixed results. For instance, uint8 is notably fast for ELU (17.15ms) and LeakyReLU (12.61ms), but uint types are slower for ReLU compared to signed integers.
- **float64** is consistently the slowest data type across most functions, with Tanh (89.73ms) and dTanh (93.66ms) showing the highest latency. This suggests floating-point precision comes at a significant computational cost.
- **int16** and **int8** often yield the best performance for integer types, with Linear achieving 10.34ms for int16 and 10.55ms for int8.

## Derivative Performance

- Derivatives (prefixed with 'd') generally have performance close to their base functions, but there are exceptions:
  - **dSigmoid** and **dTanh** are slower than their base functions for float64 (79.05ms vs. 75.90ms for Sigmoid, 93.66ms vs. 89.73ms for Tanh).
  - **dReLU** and **dLeakyReLU** are slightly faster or comparable to their base functions across most data types, making them efficient choices for backpropagation.
- **dLinear** is nearly as fast as Linear, making it a low-overhead choice for gradient computations.

## Notable Patterns

- **ReLU** and **LeakyReLU** are robust across all data types, with minimal performance degradation even for float64 (14.11ms for ReLU, 16.26ms for LeakyReLU).
- **Sigmoid** and **Tanh** show significant performance drops for floating-point types, likely due to their reliance on exponential and hyperbolic computations, which are computationally expensive.
- **ELU** performs better with unsigned integers (e.g., 17.15ms for uint8) than signed integers or floating-point types, suggesting potential optimization opportunities for specific use cases.
- The performance gap between integer and floating-point types is most pronounced for Sigmoid and Tanh, indicating these functions are less suited for high-precision floating-point applications.

## Recommendations

- For applications prioritizing speed, use **Linear** or **ReLU** with integer data types (int8 or int16) to minimize latency.
- **LeakyReLU** is a good alternative to ReLU when negative gradients are needed, as it maintains comparable performance.
- Avoid **Sigmoid** and **Tanh** for floating-point types, especially float64, due to their high computational cost.
- Consider **uint8** or **int16** for memory-constrained environments, as they offer excellent performance across most functions.

These observations can guide the selection of activation functions and data types based on performance requirements and hardware constraints.
