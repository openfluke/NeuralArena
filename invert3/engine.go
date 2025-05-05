package main

import (
	"fmt"
	"log"
	"math"

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
	layerSizes := []struct{ Width, Height int }{{28, 28}, {16, 16}, {10, 1}}
	activations := []string{"leaky_relu", "leaky_relu", "softmax"}
	fullyConnected := []bool{true, false, true}
	//modelPath := filepath.Join(modelDir, modelFile)

	var nn *paragon.Network
	fmt.Println("🧠 No pre-trained model found. Starting training...")
	nn = paragon.NewNetwork(layerSizes, activations, fullyConnected)
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
	for bucket, stats := range nn.Performance.Buckets {
		fmt.Printf(" - %-7s → %d samples\n", bucket, stats.Count)
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

	perSample := paragon.ComputePerSamplePerformance(expectedVectors, actualVectors, 0.01, nn)
	paragon.PrintSampleDiagnostics(perSample, 0.01)

	runStudentDistillation(trainSetInputs, trainSetTargets, nn)

	distillRandomMapping()
	distillXORSynthetic()
	distillSineMimicry()

}

func createStudentNet() *paragon.Network {

	layerSizes := []struct{ Width, Height int }{{28, 28}, {16, 16}, {10, 1}}
	activations := []string{"leaky_relu", "leaky_relu", "softmax"}
	fullyConnected := []bool{true, false, true}

	nn := paragon.NewNetwork(layerSizes, activations, fullyConnected)

	return nn
}

func runStudentDistillation(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64,
	nn *paragon.Network,
) {
	fmt.Println("\n---------🧠 Student Distillation With ProxyError (No Teacher Output)----------")

	params := []struct {
		MaxUpdate float64
		Damping   float64
	}{
		{0.5, 0.3},
		{0.5, 0.7},
		{0.5, 0.2},
		{0.5, 0.1},
		{0.5, 0.01},
		{0.1, 0.01},
		{5.0, 0.01},
	}

	// Evaluate teacher once
	var teachExpected, teachPredicted []float64
	for i := range trainSetInputs {
		nn.Forward(trainSetInputs[i])
		pred := float64(paragon.ArgMax(nn.ExtractOutput()))
		trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
		teachExpected = append(teachExpected, trueLabel)
		teachPredicted = append(teachPredicted, pred)
	}
	nn.EvaluateModel(teachExpected, teachPredicted)
	teacherScore := nn.Performance.Score

	fmt.Printf("%-20s %-20s %-20s\n", "Params", "Teacher ADHD", "Student ADHD")

	// Train and evaluate each student
	for _, p := range params {
		student := createStudentNet()
		lr := 0.01

		for i := range trainSetInputs {
			input := trainSetInputs[i]
			targetVec := trainSetTargets[i][0] // assumes [1][10] shape

			student.Forward(input)
			predVec := student.ExtractOutput()

			// Compute average absolute error
			var err float64
			for j := range predVec {
				err += math.Abs(predVec[j] - targetVec[j])
			}
			err /= float64(len(predVec))

			// Apply proxy error adjustment
			student.PropagateProxyError(input, err, lr, p.MaxUpdate, p.Damping, 0.9)
		}

		// Evaluate student
		var studExpected, studPredicted []float64
		for i := range trainSetInputs {
			student.Forward(trainSetInputs[i])
			pred := float64(paragon.ArgMax(student.ExtractOutput()))
			trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
			studExpected = append(studExpected, trueLabel)
			studPredicted = append(studPredicted, pred)
		}
		student.EvaluateModel(studExpected, studPredicted)

		fmt.Printf("max=%.2f damp=%.2f   %-20.2f %-20.2f\n",
			p.MaxUpdate, p.Damping, teacherScore, student.Performance.Score)
	}
}
