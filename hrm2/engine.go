package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"paragon"

	"github.com/openfluke/pilot"
	"github.com/openfluke/pilot/experiments"
)

const (
	epochs       = 10
	learningRate = 0.01
	batchSize    = 64
	modelsDir    = "./models"
)

func main() {
	startTotal := time.Now()
	fmt.Println("🚀 Running experiment: MNIST with HRM")
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
	// Build the HRM
	startInit := time.Now()
	hrm := paragon.NewHRM(784, 512, 10) // InputSize=784 (28x28), HiddenSize=512, OutputSize=10
	hrm.SetArchitecture(17, 15)         // Increased LowSteps and HighCycles
	hrm.SetLearningRate(learningRate)
	fmt.Printf("⏱ HRM Init Time: %v\n", time.Since(startInit))
	// Note: HRM doesn't use WebGPU directly yet; we'll simulate training
	fmt.Println("⚠️ WebGPU not yet integrated for HRM - using CPU")
	// Training loop with batches
	fmt.Println("🧠 Training the HRM...")
	startTrain := time.Now()
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		for i := 0; i < len(trainInputs); i += batchSize {
			end := i + batchSize
			if end > len(trainInputs) {
				end = len(trainInputs)
			}
			batchX := trainInputs[i:end]
			batchY := trainTargets[i:end]
			for j := 0; j < len(batchX); j++ {
				loss := hrm.Train(batchX[j], batchY[j])
				totalLoss += loss
			}
		}
		avgLoss := totalLoss / float64(len(trainInputs))
		if epoch%1 == 0 {
			fmt.Printf("Epoch %d, Average Loss: %.4f\n", epoch, avgLoss)
		}
	}
	fmt.Printf("⏱ Total Training Time: %v\n", time.Since(startTrain))
	// Evaluate with custom logic (since no ADHD for HRM yet)
	startEval := time.Now()
	trainScore := evaluateHRM(hrm, trainInputs, trainTargets, "Train")
	testScore := evaluateHRM(hrm, testInputs, testTargets, "Test")
	fmt.Printf("📊 Train Score: %.4f%%\n", trainScore)
	fmt.Printf("📊 Test Score: %.4f%%\n", testScore)
	fmt.Printf("⏱ Evaluation Time: %v\n", time.Since(startEval))
	// Save the model (custom for HRM)
	startSave := time.Now()
	modelPath := filepath.Join(modelsDir, "mnist_hrm_model.json")
	if err := saveHRMModel(hrm, modelPath); err != nil {
		fmt.Printf("❌ Failed to save HRM model: %v\n", err)
	} else {
		fmt.Printf("💾 Saved HRM model to %s\n", modelPath)
	}
	fmt.Printf("⏱ Model Save Time: %v\n", time.Since(startSave))
	fmt.Printf("⏱ Total Experiment Time: %v\n", time.Since(startTotal))
}

// evaluateHRM evaluates HRM performance
func evaluateHRM(hrm *paragon.HRM, inputs, targets [][][]float64, dataset string) float64 {
	start := time.Now()
	correct := 0
	for i := range inputs {
		output := hrm.Forward(inputs[i])[0]
		prediction := 0.0
		if output[0] > 0.5 { // Assuming single output for now
			prediction = 1.0
		}
		target := targets[i][0][paragon.ArgMax(targets[i][0])] // Get the true class
		if math.Abs(prediction-target) < 0.5 {
			correct++
		}
	}
	score := float64(correct) / float64(len(inputs)) * 100
	// Print basic assessment
	fmt.Printf("\n📈 HRM Performance (%s Set):\n", dataset)
	fmt.Printf("- Correct: %d\n", correct)
	fmt.Printf("- Total Samples: %d\n", len(inputs))
	fmt.Printf("- Score: %.4f%%\n", score)
	fmt.Printf("⏱ Evaluate Time (%s): %v\n", dataset, time.Since(start))
	return score
}

// saveHRMModel saves the HRM model (placeholder - adapt to your format)
func saveHRMModel(hrm *paragon.HRM, path string) error {
	// This is a placeholder - you'll need to implement serialization for HRM
	// For now, just create a dummy file
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString("HRM Model Placeholder")
	return err
}

// Include your existing loadMNISTData functions here (already provided)
func loadMNISTData(dir string) ([][][]float64, [][][]float64, error) {
	images := make([][][]float64, 0)
	labels := make([][][]float64, 0)

	for _, set := range []string{"train", "t10k"} {
		imgPath := filepath.Join(dir, set+"-images-idx3-ubyte")
		lblPath := filepath.Join(dir, set+"-labels-idx1-ubyte")

		imgs, err := loadMNISTImages(imgPath)
		if err != nil {
			return nil, nil, err
		}

		lbls, err := loadMNISTLabels(lblPath)
		if err != nil {
			return nil, nil, err
		}

		images = append(images, imgs...)
		labels = append(labels, lbls...)
	}

	return images, labels, nil
}

func loadMNISTImages(path string) ([][][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var header [16]byte
	if _, err := f.Read(header[:]); err != nil {
		return nil, err
	}
	num := int(binary.BigEndian.Uint32(header[4:8]))
	rows := int(binary.BigEndian.Uint32(header[8:12]))
	cols := int(binary.BigEndian.Uint32(header[12:16]))

	images := make([][][]float64, num)
	buf := make([]byte, rows*cols)
	for i := 0; i < num; i++ {
		if _, err := f.Read(buf); err != nil {
			return nil, err
		}
		img := make([][]float64, rows)
		for r := 0; r < rows; r++ {
			img[r] = make([]float64, cols)
			for c := 0; c < cols; c++ {
				img[r][c] = float64(buf[r*cols+c]) / 255.0
			}
		}
		images[i] = img
	}
	return images, nil
}

func loadMNISTLabels(path string) ([][][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var header [8]byte
	if _, err := f.Read(header[:]); err != nil {
		return nil, err
	}
	num := int(binary.BigEndian.Uint32(header[4:8]))

	labels := make([][][]float64, num)
	for i := 0; i < num; i++ {
		var b [1]byte
		if _, err := f.Read(b[:]); err != nil {
			return nil, err
		}
		labels[i] = labelToOneHot(int(b[0]))
	}
	return labels, nil
}

func labelToOneHot(label int) [][]float64 {
	t := make([][]float64, 1)
	t[0] = make([]float64, 10)
	t[0][label] = 1.0
	return t
}
