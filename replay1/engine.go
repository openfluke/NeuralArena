package main

import (
	"fmt"
	"log"

	"paragon"
)

const (
	baseURL   = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir  = "mnist_data"
	modelDir  = "models"
	modelFile = "mnist_model.json"
)

func main() {
	// --- Prepare MNIST ---
	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatalf("MNIST download error: %v", err)
	}
	trainInputs, trainTargets, err := loadMNISTData(mnistDir, true)
	if err != nil {
		log.Fatalf("Training load failed: %v", err)
	}
	testInputs, testTargets, err := loadMNISTData(mnistDir, false)
	if err != nil {
		log.Fatalf("Test load failed: %v", err)
	}
	trainSetInputs, trainSetTargets, _, _ := paragon.SplitDataset(trainInputs, trainTargets, 0.8)

	// --- Build Model ---
	layerSizes := []struct{ Width, Height int }{
		{28, 28}, // input layer
		{16, 16}, // hidden layer
		{10, 1},  // output layer
	}
	activations := []string{"leaky_relu", "leaky_relu", "softmax"}
	fullyConnected := []bool{true, false, true}

	var nn *paragon.Network
	fmt.Println("🧠 No pre-trained model found. Starting training...")
	nn = paragon.NewNetwork(layerSizes, activations, fullyConnected)

	// ✅ ✅ Replay only on hidden layer (Layer 1), NOT on output!
	nn.Layers[1].ReplayOffset = -1     // Replay back to input layer
	nn.Layers[1].ReplayPhase = "after" // Trigger replay after normal layer execution
	nn.Layers[1].MaxReplay = 1         // Only one replay cycle

	// --- Train Model ---
	nn.Train(trainSetInputs, trainSetTargets, 10, 0.01, true)
	fmt.Println("✅ Training complete.")

	// --- ADHD Evaluation ---
	var expected, predicted []float64
	for i, input := range testInputs {
		nn.Forward(input)
		out := extractOutput(nn)
		pred := paragon.ArgMax(out)
		trueLabel := paragon.ArgMax(testTargets[i][0])
		expected = append(expected, float64(trueLabel))
		predicted = append(predicted, float64(pred))
	}
	nn.EvaluateModel(expected, predicted)

	// --- Unified ADHD Diagnostics ---
	fmt.Println("\n---------SimplePRINT----------")
	fmt.Printf("🧠 ADHD Score: %.2f\n", nn.Performance.Score)
	fmt.Println("📊 Deviation Buckets:")
	for bucket, st := range nn.Performance.Buckets {
		fmt.Printf(" - %-7s → %d samples\n", bucket, st.Count)
	}

	fmt.Println("\n---------PrintFullDiagnostics----------")
	nn.EvaluateFull(expected, predicted)
	nn.PrintFullDiagnostics()

	fmt.Println("\n---------PrintSAMPLEDiagnostics----------")
	expectedVectors := make([][]float64, len(testInputs))
	actualVectors := make([][]float64, len(testInputs))
	for i := range testInputs {
		nn.Forward(testInputs[i])
		actualVectors[i] = nn.ExtractOutput()
		expectedVectors[i] = testTargets[i][0]
	}
}
