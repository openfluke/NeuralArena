package main

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// Number of samples
const count = 10000000

// Generate test values [-10, 10]
func generateValues() []float64 {
	values := make([]float64, count)
	step := 20.0 / float64(count)
	for i := 0; i < count; i++ {
		values[i] = -10 + float64(i)*step
	}
	return values
}

// Activation functions
func relu(x float64) float64    { return math.Max(0, x) }
func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }
func tanh(x float64) float64    { return math.Tanh(x) }
func leakyRelu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0.01 * x
}
func elu(x float64) float64 {
	if x >= 0 {
		return x
	}
	return math.Exp(x) - 1
}
func linear(x float64) float64       { return x }
func softmaxDeriv(x float64) float64 { return x * (1 - x) }

// Derivatives
func dRelu(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}
func dSigmoid(x float64) float64 { s := sigmoid(x); return s * (1 - s) }
func dTanh(x float64) float64    { t := tanh(x); return 1 - t*t }
func dLeakyRelu(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0.01
}
func dElu(x float64) float64 {
	if x >= 0 {
		return 1
	}
	return math.Exp(x)
}
func dLinear(x float64) float64 { return 1 }

// Benchmark utility
func benchmark(fn func(float64) float64, values []float64) time.Duration {
	start := time.Now()
	for _, v := range values {
		_ = fn(v)
	}
	return time.Since(start)
}

func castAndTime(name string, cast func(float64) float64, values []float64) map[string]time.Duration {
	converted := make([]float64, len(values))
	for i, v := range values {
		converted[i] = cast(v)
	}
	return map[string]time.Duration{
		"relu":        benchmark(relu, converted),
		"sigmoid":     benchmark(sigmoid, converted),
		"tanh":        benchmark(tanh, converted),
		"leaky_relu":  benchmark(leakyRelu, converted),
		"elu":         benchmark(elu, converted),
		"linear":      benchmark(linear, converted),
		"softmax_der": benchmark(softmaxDeriv, converted),

		"d_relu":      benchmark(dRelu, converted),
		"d_sigmoid":   benchmark(dSigmoid, converted),
		"d_tanh":      benchmark(dTanh, converted),
		"d_leakyrelu": benchmark(dLeakyRelu, converted),
		"d_elu":       benchmark(dElu, converted),
		"d_linear":    benchmark(dLinear, converted),
	}
}

func main() {
	values := generateValues()

	types := []struct {
		name string
		cast func(float64) float64
	}{
		{"int8", func(x float64) float64 { return float64(int8(x)) }},
		{"int16", func(x float64) float64 { return float64(int16(x)) }},
		{"int32", func(x float64) float64 { return float64(int32(x)) }},
		{"int64", func(x float64) float64 { return float64(int64(x)) }},
		{"uint8", func(x float64) float64 { return float64(uint8(x)) }},
		{"uint16", func(x float64) float64 { return float64(uint16(x)) }},
		{"uint32", func(x float64) float64 { return float64(uint32(x)) }},
		{"uint64", func(x float64) float64 { return float64(uint64(x)) }},
		{"float32", func(x float64) float64 { return float64(float32(x)) }},
		{"float64", func(x float64) float64 { return x }},
	}

	// Clean header
	fmt.Printf("%-10s | %-10s %-10s %-10s %-10s %-10s %-10s %-12s %-10s %-10s %-10s %-12s %-10s %-10s\n",
		"Type", "ReLU", "Sigmoid", "Tanh", "LeakyReLU", "ELU", "Linear", "SoftmaxDer", "dReLU", "dSigmoid", "dTanh", "dLeakyReLU", "dELU", "dLinear")
	fmt.Println(strings.Repeat("-", 145))

	// Output each timing row
	for _, t := range types {
		timings := castAndTime(t.name, t.cast, values)
		fmt.Printf("%-10s | %-10v %-10v %-10v %-10v %-10v %-10v %-12v %-10v %-10v %-10v %-12v %-10v %-10v\n",
			t.name,
			timings["relu"],
			timings["sigmoid"],
			timings["tanh"],
			timings["leaky_relu"],
			timings["elu"],
			timings["linear"],
			timings["softmax_der"],
			timings["d_relu"],
			timings["d_sigmoid"],
			timings["d_tanh"],
			timings["d_leakyrelu"],
			timings["d_elu"],
			timings["d_linear"],
		)
	}

	fmt.Println("\n=== Conversion During Inference ===")
	fmt.Printf("%-10s | %-10s %-10s %-10s %-10s %-10s %-10s %-12s %-10s %-10s %-10s %-12s %-10s %-10s\n",
		"Type", "ReLU", "Sigmoid", "Tanh", "LeakyReLU", "ELU", "Linear", "SoftmaxDer", "dReLU", "dSigmoid", "dTanh", "dLeakyReLU", "dELU", "dLinear")
	fmt.Println(strings.Repeat("-", 145))

	for _, t := range types {
		timings := castAndTimeInline(t.name, t.cast, values)
		fmt.Printf("%-10s | %-10v %-10v %-10v %-10v %-10v %-10v %-12v %-10v %-10v %-10v %-12v %-10v %-10v\n",
			t.name,
			timings["relu"],
			timings["sigmoid"],
			timings["tanh"],
			timings["leaky_relu"],
			timings["elu"],
			timings["linear"],
			timings["softmax_der"],
			timings["d_relu"],
			timings["d_sigmoid"],
			timings["d_tanh"],
			timings["d_leakyrelu"],
			timings["d_elu"],
			timings["d_linear"],
		)
	}

}

func castAndTimeInline(name string, cast func(float64) float64, values []float64) map[string]time.Duration {
	return map[string]time.Duration{
		"relu":        benchmark(func(x float64) float64 { return relu(cast(x)) }, values),
		"sigmoid":     benchmark(func(x float64) float64 { return sigmoid(cast(x)) }, values),
		"tanh":        benchmark(func(x float64) float64 { return tanh(cast(x)) }, values),
		"leaky_relu":  benchmark(func(x float64) float64 { return leakyRelu(cast(x)) }, values),
		"elu":         benchmark(func(x float64) float64 { return elu(cast(x)) }, values),
		"linear":      benchmark(func(x float64) float64 { return linear(cast(x)) }, values),
		"softmax_der": benchmark(func(x float64) float64 { return softmaxDeriv(cast(x)) }, values),

		"d_relu":      benchmark(func(x float64) float64 { return dRelu(cast(x)) }, values),
		"d_sigmoid":   benchmark(func(x float64) float64 { return dSigmoid(cast(x)) }, values),
		"d_tanh":      benchmark(func(x float64) float64 { return dTanh(cast(x)) }, values),
		"d_leakyrelu": benchmark(func(x float64) float64 { return dLeakyRelu(cast(x)) }, values),
		"d_elu":       benchmark(func(x float64) float64 { return dElu(cast(x)) }, values),
		"d_linear":    benchmark(func(x float64) float64 { return dLinear(cast(x)) }, values),
	}
}
