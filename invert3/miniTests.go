package main

import (
	"fmt"
	"math"
	"math/rand/v2"
	"paragon"
)

// --- Mini Experiment 1: Random Mapping ---

func distillRandomMapping() {
	fmt.Println("\n---------🎲 Random Mapping Test ----------")

	teacher := createStudentNet()
	student := createStudentNet()

	inputs := generateRandomInputs(100, 28, 28)
	outputs := generateRandomOutputs(100, 10)

	// Train teacher by nudging bias
	for i := 0; i < 100; i++ {
		teacher.Forward(inputs[i])
		teacherOut := teacher.ExtractOutput()
		for j := range teacherOut {
			diff := outputs[i][j] - teacherOut[j]
			layer := &teacher.Layers[teacher.OutputLayer]
			y := j / layer.Width
			x := j % layer.Width
			layer.Neurons[y][x].Bias += diff * 0.1
		}
	}

	// Student learns by proxy
	for i := 0; i < 100; i++ {
		teacher.Forward(inputs[i])
		target := teacher.ExtractOutput()

		student.Forward(inputs[i])
		output := student.ExtractOutput()

		var err float64
		for j := range target {
			err += math.Abs(output[j] - target[j])
		}
		err /= float64(len(target))

		student.PropagateProxyError(inputs[i], err, 0.01, 0.5, 0.3, 0.9)
	}

	// ADHD scoring
	var expected, predicted []float64
	for i := range inputs {
		teacher.Forward(inputs[i])
		student.Forward(inputs[i])

		exp := float64(paragon.ArgMax(teacher.ExtractOutput()))
		pred := float64(paragon.ArgMax(student.ExtractOutput()))
		expected = append(expected, exp)
		predicted = append(predicted, pred)
	}

	teacher.EvaluateModel(expected, expected)
	student.EvaluateModel(expected, predicted)

	fmt.Printf("🧠 Teacher ADHD Score: %.2f\n", teacher.Performance.Score)
	fmt.Printf("🧠 Student ADHD Score: %.2f\n", student.Performance.Score)
}

// --- Mini Experiment 2: XOR Logic Test ---

func distillXORSynthetic() {
	fmt.Println("\n---------⚡ XOR Behavior Mimicry ----------")

	teacher := createXORNet()
	student := createXORNet()

	inputs := [][][]float64{
		{{0, 0}}, {{0, 1}}, {{1, 0}}, {{1, 1}},
	}
	targets := [][]float64{
		{0}, {1}, {1}, {0},
	}

	// Teacher learns XOR by bias tweak
	for epoch := 0; epoch < 10; epoch++ {
		for i, in := range inputs {
			teacher.Forward(in)
			out := teacher.ExtractOutput()[0]
			err := targets[i][0] - out
			teacher.Layers[teacher.OutputLayer].Neurons[0][0].Bias += err * 0.5
		}
	}

	// Student mimics
	for _, in := range inputs {
		teacher.Forward(in)
		target := teacher.ExtractOutput()
		student.Forward(in)
		output := student.ExtractOutput()

		var err float64
		for j := range target {
			err += math.Abs(target[j] - output[j])
		}
		err /= float64(len(target))
		student.PropagateProxyError(in, err, 0.1, 0.5, 0.2, 0.9)
	}

	// Evaluate mimicry using thresholding
	var expected, predicted []float64
	for _, in := range inputs {
		teacher.Forward(in)
		student.Forward(in)

		t := teacher.ExtractOutput()[0]
		s := student.ExtractOutput()[0]

		expected = append(expected, math.Round(t))
		predicted = append(predicted, math.Round(s))

		fmt.Printf("Input %v → Teacher: %.2f | Student: %.2f\n", in, t, s)
	}

	teacher.EvaluateModel(expected, expected)
	student.EvaluateModel(expected, predicted)
	fmt.Printf("🧠 Teacher ADHD Score: %.2f\n", teacher.Performance.Score)
	fmt.Printf("🧠 Student ADHD Score: %.2f\n", student.Performance.Score)
}

// --- Mini Experiment 3: Sine Function ---

func distillSineMimicry() {
	fmt.Println("\n---------🌊 Sine Function Mimicry ----------")

	layerSizes := []struct{ Width, Height int }{
		{1, 1},
		{4, 1},
		{1, 1},
	}
	activations := []string{"leaky_relu", "leaky_relu", "tanh"}
	fullyConnected := []bool{true, true, true}

	teacher := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	student := paragon.NewNetwork(layerSizes, activations, fullyConnected)

	var inputs [][][]float64
	var targets [][]float64
	for i := 0; i < 100; i++ {
		x := float64(i) * 2 * math.Pi / 100.0
		inputs = append(inputs, [][]float64{{x}})
		targets = append(targets, []float64{math.Sin(x)})
	}

	for epoch := 0; epoch < 10; epoch++ {
		for i := range inputs {
			teacher.Forward(inputs[i])
			out := teacher.ExtractOutput()[0]
			err := targets[i][0] - out
			teacher.Layers[teacher.OutputLayer].Neurons[0][0].Bias += err * 0.1
		}
	}

	for i := range inputs {
		teacher.Forward(inputs[i])
		target := teacher.ExtractOutput()
		student.Forward(inputs[i])
		output := student.ExtractOutput()

		var err float64
		for j := range target {
			err += math.Abs(target[j] - output[j])
		}
		err /= float64(len(target))
		student.PropagateProxyError(inputs[i], err, 0.01, 0.5, 0.3, 0.9)
	}

	// Evaluate mimicry using rounding for classification
	var expected, predicted []float64
	for i := 0; i < 100; i++ {
		teacher.Forward(inputs[i])
		student.Forward(inputs[i])

		t := teacher.ExtractOutput()[0]
		s := student.ExtractOutput()[0]

		expected = append(expected, math.Round(t*10)/10)
		predicted = append(predicted, math.Round(s*10)/10)
	}

	teacher.EvaluateModel(expected, expected)
	student.EvaluateModel(expected, predicted)

	fmt.Printf("🧠 Teacher ADHD Score: %.2f\n", teacher.Performance.Score)
	fmt.Printf("🧠 Student ADHD Score: %.2f\n", student.Performance.Score)

	fmt.Println("x\tTeacher\tStudent")
	for i := 0; i < 10; i++ {
		x := inputs[i][0][0]
		teacher.Forward(inputs[i])
		student.Forward(inputs[i])
		t := teacher.ExtractOutput()[0]
		s := student.ExtractOutput()[0]
		fmt.Printf("%.2f\t%.2f\t%.2f\n", x, t, s)
	}
}

// --- Helpers ---

func createXORNet() *paragon.Network {
	layerSizes := []struct{ Width, Height int }{
		{2, 1},
		{4, 1},
		{1, 1},
	}
	activations := []string{"leaky_relu", "leaky_relu", "sigmoid"}
	fullyConnected := []bool{true, true, true}

	return paragon.NewNetwork(layerSizes, activations, fullyConnected)
}

func generateRandomInputs(n, w, h int) [][][]float64 {
	inputs := make([][][]float64, n)
	for i := range inputs {
		inputs[i] = make([][]float64, h)
		for y := 0; y < h; y++ {
			row := make([]float64, w)
			for x := 0; x < w; x++ {
				row[x] = randFloat(0, 1)
			}
			inputs[i][y] = row
		}
	}
	return inputs
}

func generateRandomOutputs(n, dim int) [][]float64 {
	outputs := make([][]float64, n)
	for i := range outputs {
		row := make([]float64, dim)
		for j := range row {
			row[j] = randFloat(0, 1)
		}
		outputs[i] = row
	}
	return outputs
}

func randFloat(min, max float64) float64 {
	return min + (max-min)*rand.Float64()
}
