// main.go
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"time"

	"paragon"

	"gonum.org/v1/gonum/blas"        // BLAS constants
	"gonum.org/v1/gonum/blas/blas64" // High‑level GEMM wrapper
)

// plain triple‑nested loops: C = A · B
func mmNaive(a, b, c []float64, n int) {
	for i := 0; i < n; i++ {
		for k := 0; k < n; k++ {
			aik := a[i*n+k]
			for j := 0; j < n; j++ {
				c[i*n+j] += aik * b[k*n+j]
			}
		}
	}
}

func main() {
	// ----- CLI flags -----
	n := flag.Int("n", 512, "matrix dimension (nxn)")
	reps := flag.Int("reps", 10, "number of multiplications to run per method")
	flag.Parse()

	// ----- Allocate & fill matrices -----
	rand.Seed(42)
	elem := *n * *n
	a := make([]float64, elem)
	b := make([]float64, elem)
	for i := 0; i < elem; i++ {
		a[i] = rand.Float64()
		b[i] = rand.Float64()
	}

	// ----- Naïve benchmark -----
	startNaive := time.Now()
	for r := 0; r < *reps; r++ {
		c := make([]float64, elem)
		mmNaive(a, b, c, *n)
	}
	naiveDur := time.Since(startNaive)

	// ----- BLAS benchmark -----
	gA := blas64.General{Rows: *n, Cols: *n, Stride: *n, Data: a}
	gB := blas64.General{Rows: *n, Cols: *n, Stride: *n, Data: b}

	startBLAS := time.Now()
	for r := 0; r < *reps; r++ {
		c := make([]float64, elem)
		gC := blas64.General{Rows: *n, Cols: *n, Stride: *n, Data: c}
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1.0, gA, gB, 0.0, gC)
	}
	blasDur := time.Since(startBLAS)

	// ----- Results -----
	fmt.Printf("\nMatrix: %dx%d   reps: %d\n", *n, *n, *reps)
	fmt.Printf("Naïve total : %v  (%.3f ms/op)\n",
		naiveDur, float64(naiveDur.Milliseconds())/float64(*reps))
	fmt.Printf("BLAS  total : %v  (%.3f ms/op)\n",
		blasDur, float64(blasDur.Milliseconds())/float64(*reps))

	if blasDur > 0 {
		speedup := float64(naiveDur.Nanoseconds()) / float64(blasDur.Nanoseconds())
		fmt.Printf("Speed‑up    : %.1fx\n\n", speedup)
	}

	RunParagonBLASBenchmark()
}

func RunParagonBLASBenchmark() {
	fmt.Println("\n🚀 Running Enhanced PARAGON Scalar vs BLAS benchmark")

	const (
		inputDim   = 1024 // 32×32 flattened input
		hidden1    = 512
		hidden2    = 256
		outputDim  = 10
		epochs     = 10
		lr         = 0.001
		numSamples = 2000
	)

	// --- Generate synthetic classification data ---
	rand.Seed(42)
	ins := make([][][]float64, numSamples)
	tgts := make([][][]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		ins[i] = [][]float64{make([]float64, inputDim)}
		for j := 0; j < inputDim; j++ {
			ins[i][0][j] = rand.Float64()
		}
		label := rand.Intn(outputDim)
		tgts[i] = [][]float64{make([]float64, outputDim)}
		tgts[i][0][label] = 1.0
	}

	// --- Define architecture ---
	layers := []struct{ Width, Height int }{
		{inputDim, 1},
		{hidden1, 1},
		{hidden2, 1},
		{outputDim, 1},
	}
	acts := []string{"leaky_relu", "leaky_relu", "leaky_relu", "softmax"}
	fc := []bool{true, true, true, true}

	// --- Builder ---
	makeNet := func(useBLAS bool) *paragon.Network {
		net := paragon.NewNetwork(layers, acts, fc)
		net.ConnectLayers(fc)
		net.UseBLAS = useBLAS
		net.BakeDenseMatrices(fc)
		return net
	}

	// --- SCALAR ---
	netScalar := makeNet(false)
	startScalar := time.Now()
	for epoch := 0; epoch < epochs; epoch++ {
		for i := range ins {
			netScalar.Forward(ins[i])
			netScalar.Backward(tgts[i], lr)
		}
	}
	durScalar := time.Since(startScalar)

	// --- BLAS ---
	netBLAS := makeNet(true)
	startBLAS := time.Now()
	for epoch := 0; epoch < epochs; epoch++ {
		for i := range ins {
			netBLAS.Forward(ins[i])
			netBLAS.Backward(tgts[i], lr)
		}
	}
	durBLAS := time.Since(startBLAS)

	// --- Report ---
	fmt.Println("──────────────────────────────────────────────")
	fmt.Printf("Scalar → Total: %v | per epoch: %.2fs | samples/sec: %.1f\n",
		durScalar, durScalar.Seconds()/float64(epochs),
		float64(numSamples*epochs)/durScalar.Seconds())
	fmt.Printf("BLAS   → Total: %v | per epoch: %.2fs | samples/sec: %.1f\n",
		durBLAS, durBLAS.Seconds()/float64(epochs),
		float64(numSamples*epochs)/durBLAS.Seconds())

	if durBLAS > 0 {
		speed := float64(durScalar.Nanoseconds()) / float64(durBLAS.Nanoseconds())
		fmt.Printf("💡 Speedup from BLAS: %.1fx\n", speed)
	}
	fmt.Println("══════════════════════════════════════════════")
}
