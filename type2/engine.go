package main

import (
	"fmt"
	"math"
	"strings"
	"time"
)

const count = 10_000_000

// Numeric is “any signed or unsigned integer or float32/float64”
type Numeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64
}

// Float32 constraint for type-specific handling
type Float32 interface {
	~float32
}

// Float64 constraint for type-specific handling
type Float64 interface {
	~float64
}

// Integer constraint for type-specific handling
type Integer interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64
}

// ActivationBench holds a suite of activation + derivative funcs
type ActivationBench[T Numeric] struct {
	Name       string
	Values     []T
	ReLU       func(T) T
	Sigmoid    func(T) T
	Tanh       func(T) T
	LeakyReLU  func(T) T
	ELU        func(T) T
	Linear     func(T) T
	DReLU      func(T) T
	DSigmoid   func(T) T
	DTanh      func(T) T
	DLeakyReLU func(T) T
	DELU       func(T) T
	DLinear    func(T) T
}

// NewActivationBench wires up all of the generic funcs
func NewActivationBench[T Numeric](name string, values []T) ActivationBench[T] {
	return ActivationBench[T]{
		Name:       name,
		Values:     values,
		ReLU:       ReLU[T],
		Sigmoid:    Sigmoid[T],
		Tanh:       Tanh[T],
		LeakyReLU:  LeakyReLU[T],
		ELU:        ELU[T],
		Linear:     Linear[T],
		DReLU:      DReLU[T],
		DSigmoid:   DSigmoid[T],
		DTanh:      DTanh[T],
		DLeakyReLU: DLeakyReLU[T],
		DELU:       DELU[T],
		DLinear:    DLinear[T],
	}
}

// --- Generic Activation Functions ---

func ReLU[T Numeric](x T) T {
	var zero T
	if x > zero {
		return x
	}
	return zero
}

func Sigmoid[T Numeric](x T) T {
	switch any(x).(type) {
	case float32:
		// Approximation for float32 exp: e^x ≈ (1 + x/16)^16 (simple, not highly accurate)
		xf := float32(x)
		approxExp := func(z float32) float32 {
			t := 1 + z/16
			return t * t * t * t * t * t * t * t * t * t * t * t * t * t * t * t
		}
		s := 1 / (1 + approxExp(-xf))
		return T(s)
	case float64:
		s := 1 / (1 + math.Exp(-float64(x)))
		return T(s)
	default: // Integers
		// Scale to 0-100 range for integers (simplified sigmoid-like behavior)
		xi := int64(x)
		if xi > 10 {
			return T(100)
		} else if xi < -10 {
			return T(0)
		}
		return T(50 + xi*5) // Linear approximation between 0 and 100
	}
}

func Tanh[T Numeric](x T) T {
	switch any(x).(type) {
	case float32:
		// Simplified tanh approximation for float32: tanh(x) ≈ x / (1 + |x|)
		xf := float32(x)
		absX := xf
		if xf < 0 {
			absX = -xf
		}
		t := xf / (1 + absX)
		return T(t)
	case float64:
		t := math.Tanh(float64(x))
		return T(t)
	default: // Integers
		xi := int64(x)
		if xi > 5 {
			return T(50)
		} else if xi < -5 {
			return T(0) // instead of T(-50), for uints
		}
		return T(xi * 10)

	}
}

func LeakyReLU[T Numeric](x T) T {
	var zero T
	if x > zero {
		return x
	}
	switch any(x).(type) {
	case float32:
		return T(float32(x) * 0.01)
	case float64:
		return T(float64(x) * 0.01)
	default: // Integers
		return T(x / 100) // Integer division for 0.01 equivalent
	}
}

func ELU[T Numeric](x T) T {
	var zero T
	if x >= zero {
		return x
	}
	switch any(x).(type) {
	case float32:
		// Approximation for float32 exp
		xf := float32(x)
		approxExp := func(z float32) float32 {
			t := 1 + z/16
			return t * t * t * t * t * t * t * t * t * t * t * t * t * t * t * t
		}
		return T(approxExp(xf) - 1)
	case float64:
		return T(math.Exp(float64(x)) - 1)
	default: // Integers
		xi := int64(x)
		if xi > 5 {
			return T(50)
		} else if xi < -5 {
			return T(0) // instead of T(-50), for uints
		}
		return T(xi * 10)

	}
}

func Linear[T Numeric](x T) T {
	return x
}

// --- Generic Derivatives ---

func DReLU[T Numeric](x T) T {
	var one, zero T = 1, 0
	if x > zero {
		return one
	}
	return zero
}

func DSigmoid[T Numeric](x T) T {
	switch any(x).(type) {
	case float32:
		xf := float32(x)
		approxExp := func(z float32) float32 {
			t := 1 + z/16
			return t * t * t * t * t * t * t * t * t * t * t * t * t * t * t * t
		}
		s := 1 / (1 + approxExp(-xf))
		return T(s * (1 - s))
	case float64:
		s := 1 / (1 + math.Exp(-float64(x)))
		return T(s * (1 - s))
	default: // Integers
		xi := int64(x)
		if xi > 10 || xi < -10 {
			return T(0)
		}
		s := 50 + xi*5
		return T((100 - s) * s / 10000) // Derivative approximation scaled to 0-100
	}
}

func DTanh[T Numeric](x T) T {
	switch any(x).(type) {
	case float32:
		xf := float32(x)
		absX := xf
		if xf < 0 {
			absX = -xf
		}
		t := xf / (1 + absX)
		return T(1 - t*t)
	case float64:
		t := math.Tanh(float64(x))
		return T(1 - t*t)
	default: // Integers
		xi := int64(x)
		if xi > 5 || xi < -5 {
			return T(0)
		}
		return T(1) // Flat approximation
	}
}

func DLeakyReLU[T Numeric](x T) T {
	var zero, one T = 0, 1
	if x > zero {
		return one
	}
	return one / T(100) // 0.01 approximated as 1/100
}

func DELU[T Numeric](x T) T {
	var zero, one T = 0, 1
	if x >= zero {
		return one
	}
	switch any(x).(type) {
	case float32:
		xf := float32(x)
		approxExp := func(z float32) float32 {
			t := 1 + z/16
			return t * t * t * t * t * t * t * t * t * t * t * t * t * t * t * t
		}
		return T(approxExp(xf))
	case float64:
		return T(math.Exp(float64(x)))
	default: // Integers
		return T(0) // Simplified
	}
}

func DLinear[T Numeric](x T) T {
	return T(1)
}

// --- Benchmarking Harness ---

func runAndPrint[T Numeric](b ActivationBench[T]) {
	timings := map[string]time.Duration{
		"ReLU":       bench(b.Values, b.ReLU),
		"Sigmoid":    bench(b.Values, b.Sigmoid),
		"Tanh":       bench(b.Values, b.Tanh),
		"LeakyReLU":  bench(b.Values, b.LeakyReLU),
		"ELU":        bench(b.Values, b.ELU),
		"Linear":     bench(b.Values, b.Linear),
		"dReLU":      bench(b.Values, b.DReLU),
		"dSigmoid":   bench(b.Values, b.DSigmoid),
		"dTanh":      bench(b.Values, b.DTanh),
		"dLeakyReLU": bench(b.Values, b.DLeakyReLU),
		"dELU":       bench(b.Values, b.DELU),
		"dLinear":    bench(b.Values, b.DLinear),
	}
	fmt.Printf(
		"%-8s | %-10v %-10v %-10v %-10v %-10v %-10v %-10v %-10v %-10v %-10v %-10v %-10v\n",
		b.Name,
		timings["ReLU"],
		timings["Sigmoid"],
		timings["Tanh"],
		timings["LeakyReLU"],
		timings["ELU"],
		timings["Linear"],
		timings["dReLU"],
		timings["dSigmoid"],
		timings["dTanh"],
		timings["dLeakyReLU"],
		timings["dELU"],
		timings["dLinear"],
	)
}

func bench[T any](vals []T, fn func(T) T) time.Duration {
	start := time.Now()
	for _, v := range vals {
		_ = fn(v)
	}
	return time.Since(start)
}

func printHeader() {
	fmt.Printf("%-8s | %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n",
		"Type",
		"ReLU", "Sigmoid", "Tanh", "LeakyReLU", "ELU", "Linear",
		"dReLU", "dSigmoid", "dTanh", "dLeakyReLU", "dELU", "dLinear",
	)
	fmt.Println(strings.Repeat("-", 130))
}

// generateValues produces a float64 slice from –10 to +10
func generateValues() []float64 {
	out := make([]float64, count)
	step := 20.0 / float64(count)
	for i := range out {
		out[i] = -10 + float64(i)*step
	}
	return out
}

// CastSlice converts []float64 → []T by simple T(v) conversion
func CastSlice[T Numeric](in []float64) []T {
	out := make([]T, len(in))
	for i, v := range in {
		out[i] = T(v)
	}
	return out
}

func main() {
	base := generateValues()

	// instantiate for every type you care about
	benches := []any{
		NewActivationBench[int]("int", CastSlice[int](base)),
		NewActivationBench[int8]("int8", CastSlice[int8](base)),
		NewActivationBench[int16]("int16", CastSlice[int16](base)),
		NewActivationBench[int32]("int32", CastSlice[int32](base)),
		NewActivationBench[int64]("int64", CastSlice[int64](base)),
		NewActivationBench[uint]("uint", CastSlice[uint](base)),
		NewActivationBench[uint8]("uint8", CastSlice[uint8](base)),
		NewActivationBench[uint16]("uint16", CastSlice[uint16](base)),
		NewActivationBench[uint32]("uint32", CastSlice[uint32](base)),
		NewActivationBench[uint64]("uint64", CastSlice[uint64](base)),
		NewActivationBench[float32]("float32", CastSlice[float32](base)),
		NewActivationBench[float64]("float64", CastSlice[float64](base)),
	}

	printHeader()
	for _, anyBench := range benches {
		switch b := anyBench.(type) {
		case ActivationBench[int]:
			runAndPrint(b)
		case ActivationBench[int8]:
			runAndPrint(b)
		case ActivationBench[int16]:
			runAndPrint(b)
		case ActivationBench[int32]:
			runAndPrint(b)
		case ActivationBench[int64]:
			runAndPrint(b)
		case ActivationBench[uint]:
			runAndPrint(b)
		case ActivationBench[uint8]:
			runAndPrint(b)
		case ActivationBench[uint16]:
			runAndPrint(b)
		case ActivationBench[uint32]:
			runAndPrint(b)
		case ActivationBench[uint64]:
			runAndPrint(b)
		case ActivationBench[float32]:
			runAndPrint(b)
		case ActivationBench[float64]:
			runAndPrint(b)
		}
	}
}
