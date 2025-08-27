package main

import (
	"fmt"
	"math"
	"math/rand"
	"paragon"
	"time"
)

// 4-bit parity with noise dataset for testing
func generateParityDataset(numSamples int) ([][][]float64, [][][]float64) {
	rand.Seed(time.Now().UnixNano())
	inputs := make([][][]float64, numSamples)
	targets := make([][][]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		input := make([]float64, 4)
		for j := 0; j < 4; j++ {
			input[j] = rand.Float64() // Random bit (0 or 1)
			if input[j] > 0.5 {
				input[j] = 1.0 + rand.NormFloat64()*0.1 // Noise ±0.1
			} else {
				input[j] = 0.0 + rand.NormFloat64()*0.1 // Noise ±0.1
			}
		}
		parity := math.Mod(math.Round(input[0])+math.Round(input[1])+math.Round(input[2])+math.Round(input[3]), 2)
		inputs[i] = [][]float64{input}
		targets[i] = [][]float64{{parity}}
	}
	return inputs, targets
}

func main() {
	// Generate dataset
	numSamples := 1000
	trainInputs, trainTargets := generateParityDataset(numSamples)
	// Split dataset (80% train, 20% test)
	trainX, trainY, testX, testY := paragon.SplitDataset(trainInputs, trainTargets, 0.8)
	fmt.Println("Training set size:", len(trainX), "Test set size:", len(testX))
	// Initialize HRM with larger hidden size
	hrm := paragon.NewHRM(4, 128, 1) // Increased HiddenSize to 128
	// Adjust architecture for deeper reasoning
	hrm.SetArchitecture(17, 15) // Increased LowSteps to 17, HighCycles to 15
	// Set higher learning rate
	hrm.SetLearningRate(0.01)
	// Training loop
	epochs := 2000 // Increased epochs for larger model
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		for i := 0; i < len(trainX); i++ {
			// Train with current sample
			loss := hrm.Train(trainX[i], trainY[i]) // Returns float64
			totalLoss += loss
		}
		avgLoss := totalLoss / float64(len(trainX))
		if epoch%200 == 0 {
			fmt.Printf("Epoch %d, Average Loss: %.4f\n", epoch, avgLoss)
		}
	}
	// Test the model
	correct := 0
	for i := 0; i < len(testX); i++ {
		output := hrm.Forward(testX[i])[0] // Get first output sample
		prediction := 0.0
		if output[0] > 0.5 {
			prediction = 1.0
		}
		if math.Abs(prediction-testY[i][0][0]) < 0.5 { // Index inner slice
			correct++
		}
	}
	accuracy := float64(correct) / float64(len(testX)) * 100
	fmt.Printf("Test Accuracy: %.2f%%\n", accuracy)
}
