package main

import (
	"fmt"
	"paragon"
)

func main() {

	//test1()
	behavioralMimicFromTrainedModel()
}

func test1() {
	fmt.Println("🧠 Starting Behavior-Based Mimicry Test")
	// --- Step 1: Oracle function (y = 2x + 3)
	oracle := func(x float64) float64 {
		return 2*x + 3
	}

	// --- Step 2: Create mimic (student) network
	net := createStudentNet()

	// --- Step 3: Define inputs to test
	inputs := []float64{1, 2, 3, 4, 5}
	lr := 0.05 // adjustment rate

	// --- Step 4: Run mimicry loop
	for epoch := 0; epoch < 100; epoch++ {
		fmt.Printf("\n🔁 Epoch %d\n", epoch)
		for _, x := range inputs {
			input := [][]float64{{x}}
			target := oracle(x)

			// Forward pass
			net.Forward(input)
			pred := net.ExtractOutput()[0]
			error := target - pred

			// Adjust output layer behaviorally
			adjustOutputLayer(net, x, error, lr)

			fmt.Printf("x = %.2f | target = %.2f | pred = %.4f | Δ = %.4f\n",
				x, target, pred, error)
		}
	}
}

func createStudentNet() *paragon.Network {
	layers := []struct{ Width, Height int }{
		{1, 1}, // input
		{1, 1}, // hidden
		{1, 1}, // output
	}
	activations := []string{"linear", "linear", "linear"}
	fullyConnected := []bool{true, true, true}
	return paragon.NewNetwork(layers, activations, fullyConnected)
}

// Adjust output-layer weights and bias based on behavioral delta
func adjustOutputLayer(net *paragon.Network, input float64, error float64, lr float64) {
	const maxUpdate = 1.0 // clip big updates
	const damping = 0.5   // reduce scale of updates

	layer := &net.Layers[net.OutputLayer]

	for y := 0; y < layer.Height; y++ {
		for x := 0; x < layer.Width; x++ {
			neuron := layer.Neurons[y][x]

			// Dampened error
			adj := lr * error * damping

			if adj > maxUpdate {
				adj = maxUpdate
			} else if adj < -maxUpdate {
				adj = -maxUpdate
			}

			neuron.Bias += adj

			for i := range neuron.Inputs {
				neuron.Inputs[i].Weight += adj * input
			}
		}
	}
}

func adjustNetworkUpstream(net *paragon.Network, input float64, error float64, lr float64) {
	const maxUpdate = 1.0
	const damping = 0.5

	// We'll propagate a proxy input value forward manually
	proxySignal := input

	// Work backwards from output to input
	for layerIndex := net.OutputLayer; layerIndex > 0; layerIndex-- {
		layer := &net.Layers[layerIndex]

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]

				// Dampened error correction
				adj := lr * error * damping
				if adj > maxUpdate {
					adj = maxUpdate
				} else if adj < -maxUpdate {
					adj = -maxUpdate
				}

				// Update bias
				neuron.Bias += adj

				// Update weights with proxy signal
				for i := range neuron.Inputs {
					neuron.Inputs[i].Weight += adj * proxySignal
				}
			}
		}

		// Very crude proxy: dampen signal for prior layer (could be improved)
		proxySignal *= 0.9
	}
}

///------------------

func behavioralMimicFromTrainedModel() {
	fmt.Println("\n🎯 Starting Behavioral Mimicry from Trained Paragon Model")

	// Teacher: trained model that learned y = 3x - 2
	teacher := trainTeacherModel()

	// Student: same architecture, random weights
	student := createStudentNet()

	// Inputs to test
	inputs := []float64{1, 2, 3, 4, 5}
	lr := 0.1 // learning rate for mimicry

	// Run 5 mimicry iterations
	for iter := 0; iter < 5; iter++ {
		fmt.Printf("\n🌀 Iteration %d\n", iter)

		for _, x := range inputs {
			in := [][]float64{{x}}

			teacher.Forward(in)
			target := teacher.ExtractOutput()[0]

			student.Forward(in)
			pred := student.ExtractOutput()[0]

			err := target - pred
			adjustNetworkUpstream(student, x, err, lr)

			fmt.Printf("x = %.2f | target = %.4f | pred = %.4f | Δ = %.4f\n", x, target, pred, err)
		}
	}
}

func trainTeacherModel() *paragon.Network {
	inputs := [][][]float64{}
	targets := [][][]float64{}
	for i := 1.0; i <= 5.0; i++ {
		inputs = append(inputs, [][]float64{{i}})
		targets = append(targets, [][]float64{{3*i - 2}})
	}

	layers := []struct{ Width, Height int }{
		{1, 1}, {1, 1},
	}
	activations := []string{"linear", "linear"}
	fullyConnected := []bool{true, true}

	net := paragon.NewNetwork(layers, activations, fullyConnected)
	net.Train(inputs, targets, 500, 0.01, true)

	fmt.Println("✅ Teacher Model Trained:")
	for _, inp := range inputs {
		net.Forward(inp)
		fmt.Printf("x = %.2f → y = %.4f\n", inp[0][0], net.ExtractOutput()[0])
	}
	return net
}
