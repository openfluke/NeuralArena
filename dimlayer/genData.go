// experiment.go
package main

import (
	"fmt"
	"math/rand"
	"os"
)

// GenerateXORGridDataset creates a synthetic 4x4 grid dataset with XOR-like patterns
func GenerateXORGridDataset(filename string, numSamples int) ([][][]float64, [][][]float64) {
	f, err := os.Create(filename)
	if err != nil {
		panic(fmt.Sprintf("failed to create dataset file: %v", err))
	}
	defer f.Close()

	inputs := make([][][]float64, numSamples)
	targets := make([][][]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		inputs[i] = make([][]float64, 4)
		targets[i] = make([][]float64, 1)
		targets[i][0] = make([]float64, 10)

		// Generate 4x4 grid with 0s and 1s
		for y := 0; y < 4; y++ {
			inputs[i][y] = make([]float64, 4)
			for x := 0; x < 4; x++ {
				inputs[i][y][x] = float64(rand.Intn(2))
			}
		}

		// Complex XOR-like pattern for class (e.g., XOR of corners and center)
		c1 := int(inputs[i][0][0]) // Top-left
		c2 := int(inputs[i][0][3]) // Top-right
		c3 := int(inputs[i][3][0]) // Bottom-left
		c4 := int(inputs[i][3][3]) // Bottom-right
		center := int(inputs[i][2][2])
		label := (c1 ^ c2 ^ c3 ^ c4 ^ center) + rand.Intn(8) // 0-9 range
		label %= 10                                          // Ensure 0-9
		targets[i][0][label] = 1.0

		// Write to file
		_, err := fmt.Fprintf(f, "Sample %d: Input=%v, Label=%d\n", i, inputs[i], label)
		if err != nil {
			panic(fmt.Sprintf("failed to write dataset: %v", err))
		}
	}
	return inputs, targets
}

// GenerateXORGridDataset creates a synthetic 4x4 grid dataset with a harder XOR pattern
func HarderGenerateXORGridDataset(filename string, numSamples int) ([][][]float64, [][][]float64) {
	f, err := os.Create(filename)
	if err != nil {
		panic(fmt.Sprintf("failed to create dataset file: %v", err))
	}
	defer f.Close()

	inputs := make([][][]float64, numSamples)
	targets := make([][][]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		inputs[i] = make([][]float64, 4)
		targets[i] = make([][]float64, 1)
		targets[i][0] = make([]float64, 10)

		for y := 0; y < 4; y++ {
			inputs[i][y] = make([]float64, 4)
			for x := 0; x < 4; x++ {
				inputs[i][y][x] = float64(rand.Intn(2))
			}
		}

		// Harder XOR pattern: Nested XORs across quadrants
		q1 := int(inputs[i][0][0]) ^ int(inputs[i][0][1]) ^ int(inputs[i][1][0]) ^ int(inputs[i][1][1])
		q2 := int(inputs[i][0][2]) ^ int(inputs[i][0][3]) ^ int(inputs[i][1][2]) ^ int(inputs[i][1][3])
		q3 := int(inputs[i][2][0]) ^ int(inputs[i][2][1]) ^ int(inputs[i][3][0]) ^ int(inputs[i][3][1])
		q4 := int(inputs[i][2][2]) ^ int(inputs[i][2][3]) ^ int(inputs[i][3][2]) ^ int(inputs[i][3][3])
		label := (q1 ^ q2 ^ q3 ^ q4) % 10
		targets[i][0][label] = 1.0

		_, err := fmt.Fprintf(f, "Sample %d: Input=%v, Label=%d\n", i, inputs[i], label)
		if err != nil {
			panic(fmt.Sprintf("failed to write dataset: %v", err))
		}
	}
	return inputs, targets
}
