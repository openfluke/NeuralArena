package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"paragon"

	"github.com/openfluke/pilot"
	"github.com/openfluke/pilot/experiments"
)

const (
	epochs       = 10
	learningRate = 0.05
	modelsDir    = "./models"
)

type NetworkResult struct {
	TypeName   string
	TrainScore float64
	TestScore  float64
	EvalTime   time.Duration
	ModelSize  string
}

func main() {
	startTotal := time.Now()
	fmt.Println("🚀 Running Multi-Type Network Experiment: MNIST")

	// Create models directory
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		fmt.Printf("❌ Failed to create models directory: %v\n", err)
		return
	}

	// Load MNIST data
	fmt.Println("⚙ Stage: MNIST Dataset Prep")
	startData := time.Now()
	mnist := experiments.NewMNISTDatasetStage("./data/mnist")
	exp := pilot.NewExperiment("MNIST", mnist)
	if err := exp.RunAll(); err != nil {
		fmt.Println("❌ Experiment failed:", err)
		os.Exit(1)
	}
	allInputs, allTargets, err := loadMNISTData("./data/mnist")
	if err != nil {
		fmt.Println("❌ Failed to load MNIST:", err)
		return
	}
	fmt.Printf("📊 Dataset sizes: Train=%d, Test=%d\n", len(allInputs)*8/10, len(allInputs)*2/10)
	fmt.Printf("⏱ Data Prep Time: %v\n", time.Since(startData))

	// Split into 80% training and 20% testing
	trainInputs, trainTargets, testInputs, testTargets := paragon.SplitDataset(allInputs, allTargets, 0.8)

	// Build and train the original float32 network
	fmt.Println("🧠 Building and Training Original Float32 Network...")
	startInit := time.Now()
	nnFloat32 := paragon.NewNetwork[float32](
		[]struct{ Width, Height int }{{28, 28}, {32, 32}, {10, 1}},
		[]string{"linear", "relu", "softmax"},
		[]bool{true, true, true},
	)
	fmt.Printf("⏱ Network Init Time: %v\n", time.Since(startInit))

	// Enable WebGPU for training
	nnFloat32.WebGPUNative = true
	nnFloat32.Debug = false
	startGPU := time.Now()
	if err := nnFloat32.InitializeOptimizedGPU(); err != nil {
		fmt.Printf("⚠️ Failed to initialize WebGPU: %v\n", err)
		fmt.Println("   Continuing with CPU-only processing...")
		nnFloat32.WebGPUNative = false
	} else {
		fmt.Println("✅ WebGPU initialized successfully")
		defer nnFloat32.CleanupOptimizedGPU()
	}
	fmt.Printf("⏱ WebGPU Init Time: %v\n", time.Since(startGPU))

	// Train the float32 network
	fmt.Println("🏋️ Training Float32 Network...")
	startTrain := time.Now()
	nnFloat32.TrainWithGPUSync(trainInputs, trainTargets, epochs, learningRate, false, float32(2), float32(-2))
	fmt.Printf("⏱ Float32 Training Time: %v\n", time.Since(startTrain))

	// Store results for all network types
	var results []NetworkResult

	// Evaluate original float32 network
	fmt.Println("\n📊 Evaluating Float32 Network...")
	float32Result := evaluateNetwork(nnFloat32, trainInputs, trainTargets, testInputs, testTargets, "float32")
	results = append(results, float32Result)

	// Save original float32 model
	modelPath := filepath.Join(modelsDir, "mnist_float32_model.json")
	if err := nnFloat32.SaveJSON(modelPath); err != nil {
		fmt.Printf("❌ Failed to save float32 model: %v\n", err)
	} else {
		fmt.Printf("💾 Saved float32 model to %s\n", modelPath)
	}

	// Convert to other numeric types and evaluate (CPU only)
	fmt.Println("\n🔄 Converting to Other Numeric Types...")

	conversionTypes := []string{"float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32"}

	for _, typeName := range conversionTypes {
		fmt.Printf("\n🔄 Converting to %s...\n", typeName)
		startConvert := time.Now()

		var result NetworkResult

		switch typeName {
		case "float64":
			if convertedNN, err := paragon.ConvertNetwork[float32, float64](nnFloat32); err != nil {
				fmt.Printf("❌ Failed to convert to %s: %v\n", typeName, err)
				continue
			} else {
				// Ensure converted network uses CPU only
				convertedNN.WebGPUNative = false
				fmt.Printf("✅ Conversion to %s completed in %v\n", typeName, time.Since(startConvert))
				result = evaluateNetworkCPUOnly(convertedNN, testInputs, testTargets, typeName)
				// Save converted model
				convertedPath := filepath.Join(modelsDir, fmt.Sprintf("mnist_%s_model.json", typeName))
				if err := convertedNN.SaveJSON(convertedPath); err == nil {
					fmt.Printf("💾 Saved %s model to %s\n", typeName, convertedPath)
				}
			}
		case "int8":
			if convertedNN, err := paragon.ConvertNetwork[float32, int8](nnFloat32); err != nil {
				fmt.Printf("❌ Failed to convert to %s: %v\n", typeName, err)
				continue
			} else {
				convertedNN.WebGPUNative = false
				fmt.Printf("✅ Conversion to %s completed in %v\n", typeName, time.Since(startConvert))
				result = evaluateNetworkCPUOnly(convertedNN, testInputs, testTargets, typeName)
				convertedPath := filepath.Join(modelsDir, fmt.Sprintf("mnist_%s_model.json", typeName))
				if err := convertedNN.SaveJSON(convertedPath); err == nil {
					fmt.Printf("💾 Saved %s model to %s\n", typeName, convertedPath)
				}
			}
		case "int16":
			if convertedNN, err := paragon.ConvertNetwork[float32, int16](nnFloat32); err != nil {
				fmt.Printf("❌ Failed to convert to %s: %v\n", typeName, err)
				continue
			} else {
				convertedNN.WebGPUNative = false
				fmt.Printf("✅ Conversion to %s completed in %v\n", typeName, time.Since(startConvert))
				result = evaluateNetworkCPUOnly(convertedNN, testInputs, testTargets, typeName)
				convertedPath := filepath.Join(modelsDir, fmt.Sprintf("mnist_%s_model.json", typeName))
				if err := convertedNN.SaveJSON(convertedPath); err == nil {
					fmt.Printf("💾 Saved %s model to %s\n", typeName, convertedPath)
				}
			}
		case "int32":
			if convertedNN, err := paragon.ConvertNetwork[float32, int32](nnFloat32); err != nil {
				fmt.Printf("❌ Failed to convert to %s: %v\n", typeName, err)
				continue
			} else {
				convertedNN.WebGPUNative = false
				fmt.Printf("✅ Conversion to %s completed in %v\n", typeName, time.Since(startConvert))
				result = evaluateNetworkCPUOnly(convertedNN, testInputs, testTargets, typeName)
				convertedPath := filepath.Join(modelsDir, fmt.Sprintf("mnist_%s_model.json", typeName))
				if err := convertedNN.SaveJSON(convertedPath); err == nil {
					fmt.Printf("💾 Saved %s model to %s\n", typeName, convertedPath)
				}
			}
		case "int64":
			if convertedNN, err := paragon.ConvertNetwork[float32, int64](nnFloat32); err != nil {
				fmt.Printf("❌ Failed to convert to %s: %v\n", typeName, err)
				continue
			} else {
				convertedNN.WebGPUNative = false
				fmt.Printf("✅ Conversion to %s completed in %v\n", typeName, time.Since(startConvert))
				result = evaluateNetworkCPUOnly(convertedNN, testInputs, testTargets, typeName)
				convertedPath := filepath.Join(modelsDir, fmt.Sprintf("mnist_%s_model.json", typeName))
				if err := convertedNN.SaveJSON(convertedPath); err == nil {
					fmt.Printf("💾 Saved %s model to %s\n", typeName, convertedPath)
				}
			}
		case "uint8":
			if convertedNN, err := paragon.ConvertNetwork[float32, uint8](nnFloat32); err != nil {
				fmt.Printf("❌ Failed to convert to %s: %v\n", typeName, err)
				continue
			} else {
				convertedNN.WebGPUNative = false
				fmt.Printf("✅ Conversion to %s completed in %v\n", typeName, time.Since(startConvert))
				result = evaluateNetworkCPUOnly(convertedNN, testInputs, testTargets, typeName)
				convertedPath := filepath.Join(modelsDir, fmt.Sprintf("mnist_%s_model.json", typeName))
				if err := convertedNN.SaveJSON(convertedPath); err == nil {
					fmt.Printf("💾 Saved %s model to %s\n", typeName, convertedPath)
				}
			}
		case "uint16":
			if convertedNN, err := paragon.ConvertNetwork[float32, uint16](nnFloat32); err != nil {
				fmt.Printf("❌ Failed to convert to %s: %v\n", typeName, err)
				continue
			} else {
				convertedNN.WebGPUNative = false
				fmt.Printf("✅ Conversion to %s completed in %v\n", typeName, time.Since(startConvert))
				result = evaluateNetworkCPUOnly(convertedNN, testInputs, testTargets, typeName)
				convertedPath := filepath.Join(modelsDir, fmt.Sprintf("mnist_%s_model.json", typeName))
				if err := convertedNN.SaveJSON(convertedPath); err == nil {
					fmt.Printf("💾 Saved %s model to %s\n", typeName, convertedPath)
				}
			}
		case "uint32":
			if convertedNN, err := paragon.ConvertNetwork[float32, uint32](nnFloat32); err != nil {
				fmt.Printf("❌ Failed to convert to %s: %v\n", typeName, err)
				continue
			} else {
				convertedNN.WebGPUNative = false
				fmt.Printf("✅ Conversion to %s completed in %v\n", typeName, time.Since(startConvert))
				result = evaluateNetworkCPUOnly(convertedNN, testInputs, testTargets, typeName)
				convertedPath := filepath.Join(modelsDir, fmt.Sprintf("mnist_%s_model.json", typeName))
				if err := convertedNN.SaveJSON(convertedPath); err == nil {
					fmt.Printf("💾 Saved %s model to %s\n", typeName, convertedPath)
				}
			}
		}

		if result.TypeName != "" {
			results = append(results, result)
		}
	}

	// Print comprehensive comparison
	printComparisonTable(results)
	fmt.Printf("\n⏱ Total Experiment Time: %v\n", time.Since(startTotal))
}

func evaluateNetwork[T paragon.Numeric](nn *paragon.Network[T], trainInputs, trainTargets, testInputs, testTargets [][][]float64, typeName string) NetworkResult {
	startEval := time.Now()

	// Evaluate training set
	expectedTrain := make([]float64, len(trainInputs))
	actualTrain := make([]float64, len(trainInputs))
	for i := range trainInputs {
		nn.Forward(trainInputs[i])
		out := nn.ExtractOutput()
		expectedTrain[i] = float64(paragon.ArgMax(trainTargets[i][0]))
		actualTrain[i] = float64(paragon.ArgMax(out))
	}
	nn.EvaluateModel(expectedTrain, actualTrain)
	trainScore := nn.Performance.Score

	// Evaluate test set
	expectedTest := make([]float64, len(testInputs))
	actualTest := make([]float64, len(testInputs))
	for i := range testInputs {
		nn.Forward(testInputs[i])
		out := nn.ExtractOutput()
		expectedTest[i] = float64(paragon.ArgMax(testTargets[i][0]))
		actualTest[i] = float64(paragon.ArgMax(out))
	}
	nn.EvaluateModel(expectedTest, actualTest)
	testScore := nn.Performance.Score

	evalTime := time.Since(startEval)

	// Print detailed ADHD assessment
	fmt.Printf("\n📈 ADHD Performance (%s Network):\n", typeName)
	fmt.Printf("- Train Score: %.4f%%\n", trainScore)
	fmt.Printf("- Test Score: %.4f%%\n", testScore)
	fmt.Printf("- Test Set Breakdown:\n")
	for name, bucket := range nn.Performance.Buckets {
		if bucket.Count > 0 {
			fmt.Printf("  - %s: %d samples (%.2f%%)\n", name, bucket.Count, float64(bucket.Count)/float64(nn.Performance.Total)*100)
		}
	}
	fmt.Printf("- Total Test Samples: %d\n", nn.Performance.Total)
	fmt.Printf("- Failures (100%%+): %d (%.2f%%)\n", nn.Performance.Failures, float64(nn.Performance.Failures)/float64(nn.Performance.Total)*100)
	fmt.Printf("⏱ Evaluation Time: %v\n", evalTime)

	return NetworkResult{
		TypeName:   typeName,
		TrainScore: trainScore,
		TestScore:  testScore,
		EvalTime:   evalTime,
		ModelSize:  estimateModelSize(typeName),
	}
}

func evaluateNetworkCPUOnly[T paragon.Numeric](nn *paragon.Network[T], testInputs, testTargets [][][]float64, typeName string) NetworkResult {
	startEval := time.Now()

	// Only evaluate test set for converted networks (CPU only)
	expectedTest := make([]float64, len(testInputs))
	actualTest := make([]float64, len(testInputs))
	for i := range testInputs {
		nn.Forward(testInputs[i]) // This will use CPU since WebGPUNative is false
		out := nn.ExtractOutput()
		expectedTest[i] = float64(paragon.ArgMax(testTargets[i][0]))
		actualTest[i] = float64(paragon.ArgMax(out))
	}
	nn.EvaluateModel(expectedTest, actualTest)
	testScore := nn.Performance.Score

	evalTime := time.Since(startEval)

	// Print detailed ADHD assessment
	fmt.Printf("\n📈 ADHD Performance (%s Network - CPU Only):\n", typeName)
	fmt.Printf("- Test Score: %.4f%%\n", testScore)
	fmt.Printf("- Test Set Breakdown:\n")
	for name, bucket := range nn.Performance.Buckets {
		if bucket.Count > 0 {
			fmt.Printf("  - %s: %d samples (%.2f%%)\n", name, bucket.Count, float64(bucket.Count)/float64(nn.Performance.Total)*100)
		}
	}
	fmt.Printf("- Total Test Samples: %d\n", nn.Performance.Total)
	fmt.Printf("- Failures (100%%+): %d (%.2f%%)\n", nn.Performance.Failures, float64(nn.Performance.Failures)/float64(nn.Performance.Total)*100)
	fmt.Printf("⏱ CPU Evaluation Time: %v\n", evalTime)

	return NetworkResult{
		TypeName:   typeName,
		TrainScore: 0.0, // Not evaluated for converted networks
		TestScore:  testScore,
		EvalTime:   evalTime,
		ModelSize:  estimateModelSize(typeName),
	}
}

func printComparisonTable(results []NetworkResult) {
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("📊 COMPREHENSIVE NETWORK TYPE COMPARISON")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Printf("%-10s %-12s %-12s %-12s %-15s\n", "Type", "Train Score", "Test Score", "Eval Time", "Est. Size")
	fmt.Println(strings.Repeat("-", 80))

	for _, result := range results {
		if result.TrainScore > 0 {
			// For float32 original network (has both train and test scores)
			fmt.Printf("%-10s %-12.4f %-12.4f %-12v %-15s\n",
				result.TypeName,
				result.TrainScore,
				result.TestScore,
				result.EvalTime,
				result.ModelSize)
		} else {
			// For converted networks (test score only)
			fmt.Printf("%-10s %-12s %-12.4f %-12v %-15s\n",
				result.TypeName,
				"N/A",
				result.TestScore,
				result.EvalTime,
				result.ModelSize)
		}
	}

	fmt.Println(strings.Repeat("-", 80))

	// Find best and worst performers
	if len(results) > 0 {
		best := results[0]
		worst := results[0]

		for _, result := range results[1:] {
			if result.TestScore > best.TestScore {
				best = result
			}
			if result.TestScore < worst.TestScore {
				worst = result
			}
		}

		fmt.Printf("🏆 Best Performer: %s (%.4f%% test score)\n", best.TypeName, best.TestScore)
		fmt.Printf("🔻 Worst Performer: %s (%.4f%% test score)\n", worst.TypeName, worst.TestScore)
		fmt.Printf("📈 Performance Range: %.4f%% difference\n", best.TestScore-worst.TestScore)
	}

	// Performance insights
	fmt.Println("\n🔍 INSIGHTS:")
	for _, result := range results {
		if result.TypeName == "float32" {
			fmt.Printf("• %s: Original precision baseline (GPU trained)\n", result.TypeName)
		} else if result.TypeName == "float64" {
			fmt.Printf("• %s: Higher precision (%.4f%% vs float32) - CPU evaluated\n", result.TypeName,
				result.TestScore-getScoreByType(results, "float32"))
		} else if result.TypeName == "int8" {
			fmt.Printf("• %s: Extremely quantized (%.4f%% vs float32) - %s - CPU evaluated\n", result.TypeName,
				result.TestScore-getScoreByType(results, "float32"), result.ModelSize)
		} else if result.TypeName == "uint8" {
			fmt.Printf("• %s: Unsigned quantized (%.4f%% vs float32) - %s - CPU evaluated\n", result.TypeName,
				result.TestScore-getScoreByType(results, "float32"), result.ModelSize)
		} else {
			fmt.Printf("• %s: Converted model (%.4f%% vs float32) - %s - CPU evaluated\n", result.TypeName,
				result.TestScore-getScoreByType(results, "float32"), result.ModelSize)
		}
	}
}

func getScoreByType(results []NetworkResult, typeName string) float64 {
	for _, result := range results {
		if result.TypeName == typeName {
			return result.TestScore
		}
	}
	return 0.0
}

func estimateModelSize(typeName string) string {
	// Rough estimation based on type sizes
	// Assuming typical network with ~50K parameters
	baseParams := 50000

	switch typeName {
	case "int8", "uint8":
		return fmt.Sprintf("~%dKB", baseParams/1024)
	case "int16", "uint16":
		return fmt.Sprintf("~%dKB", (baseParams*2)/1024)
	case "int32", "uint32", "float32":
		return fmt.Sprintf("~%dKB", (baseParams*4)/1024)
	case "int64", "float64":
		return fmt.Sprintf("~%dKB", (baseParams*8)/1024)
	default:
		return "Unknown"
	}
}
