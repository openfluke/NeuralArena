package main

import (
	"fmt"
	"paragon"
)

func runStudentDistillation(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64, // ✅ FIXED TYPE
	nn *paragon.Network,
) {
	fmt.Println("\n---------Student Network Iterative Distillation on Training Data----------")

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

	for _, p := range params {
		fmt.Printf("\n⚙️  maxUpdate=%.2f  damping=%.2f\n", p.MaxUpdate, p.Damping)
		student := createStudentNet()
		lr := 0.01
		lastScore := 0.0
		//tick := 0

		fmt.Printf("%-12s %-20s %-20s %s\n", "Iteration", "Teacher ADHD", "Student ADHD", "Δ")

		for iter := 0; iter < 3; iter++ {
			totalError := 0.0

			for i := range trainSetInputs {
				input := trainSetInputs[i]

				nn.Forward(input)
				targetVec := nn.ExtractOutput()

				student.Forward(input)
				predVec := student.ExtractOutput()

				for j := range targetVec {
					err := targetVec[j] - predVec[j]
					if err > 0.1 {
						err = 0.1
					} else if err < -0.1 {
						err = -0.1
					}

					adjustNetworkUpstream(student, input, err, lr, p.MaxUpdate, p.Damping)

					// adjustNetworkUpstream(student, input, err, lr, p.MaxUpdate, p.Damping)
					//adjustNetworkUpstreamSmart(student, input, err, lr, p.MaxUpdate, p.Damping)

					//adjustNetworkBehavioralPulse(student, input, err, lr, p.MaxUpdate, p.Damping)

					//adjustNetworkUpstreamNoSignalAdjsutment(student, input, err, lr, p.MaxUpdate, p.Damping)

					//adjustNetworkUpstreamDepthScaled(student, input, err, lr, p.MaxUpdate, p.Damping)

					//adjustNetworkWaveProp(student, input, err, lr, p.MaxUpdate, p.Damping)
					//adjustNetworkPulseFlow(student, input, err, lr, p.MaxUpdate, p.Damping)

					//adjustNetworkSTDPDirect(student, input, err, lr, p.MaxUpdate, p.Damping)

					//adjustNetworkSparseEcho(student, input, err, lr, p.MaxUpdate, p.Damping)

					//adjustNetworkHebbError(student, input, err, lr, p.MaxUpdate, p.Damping)

					//adjustNetworkMomentum(student, input, err, lr, p.MaxUpdate, p.Damping)

					//adjustNetworkFeatureEcho(student, input, err, lr, p.MaxUpdate, p.Damping)

					/*for k := 0; k < 3; k++ {
						adjustNetworkPhaseTunedContrast(student, input, err, lr, p.MaxUpdate, p.Damping, k)

					}*/

					totalError += err * err
				}
			}

			// Evaluate student
			var trainExpected, trainPredicted []float64
			for i := range trainSetInputs {
				student.Forward(trainSetInputs[i])
				pred := float64(paragon.ArgMax(student.ExtractOutput()))
				trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0])) // ✅ FIXED
				trainExpected = append(trainExpected, trueLabel)
				trainPredicted = append(trainPredicted, pred)
			}
			student.EvaluateModel(trainExpected, trainPredicted)
			currentScore := student.Performance.Score

			// Evaluate teacher
			var teachExpected, teachPredicted []float64
			for i := range trainSetInputs {
				nn.Forward(trainSetInputs[i])
				pred := float64(paragon.ArgMax(nn.ExtractOutput()))
				trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0])) // ✅ FIXED
				teachExpected = append(teachExpected, trueLabel)
				teachPredicted = append(teachPredicted, pred)
			}
			nn.EvaluateModel(teachExpected, teachPredicted)
			teacherScore := nn.Performance.Score

			var symbol string
			if currentScore > lastScore {
				symbol = "⬆"
			} else if currentScore < lastScore {
				symbol = "⬇"
			} else {
				symbol = "="
			}
			fmt.Printf("%-12d %-20.2f %-20.2f %s\n", iter, teacherScore, currentScore, symbol)
			lastScore = currentScore
		}
	}
}

func runStudentDistillationPermuteErrLR(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64,
	nn *paragon.Network,
) {
	fmt.Println("\n---------Student Network Permuted Error/LR Sweep (🧪 Experimental Upstream Divergence)----------")

	// ✅ Fixed settings
	maxUpdate := 0.10
	damping := 0.01

	// 🎛️ Permutations to try
	errorVariants := []float64{-0.3, -0.1, -0.05, 0.05, 0.1, 0.3}
	learningRates := []float64{0.001, 0.01, 0.05, 0.1, 0.2}

	fmt.Printf("%-12s %-12s %-20s\n", "Error", "LR", "Student ADHD")

	for _, errOffset := range errorVariants {
		for _, lr := range learningRates {
			student := createStudentNet()

			// For all training samples
			for i := range trainSetInputs {
				input := trainSetInputs[i]

				nn.Forward(input)
				targetVec := nn.ExtractOutput()

				student.Forward(input)
				predVec := student.ExtractOutput()

				// Adjust each output element using offset err
				for j := range targetVec {
					rawErr := targetVec[j] - predVec[j]
					err := rawErr + errOffset

					// Clamp err in case offset is too large
					if err > 0.1 {
						err = 0.1
					} else if err < -0.1 {
						err = -0.1
					}

					adjustNetworkUpstream(student, input, err, lr, maxUpdate, damping)
				}
			}

			// 🧠 Evaluate the student after all adjustments
			var expected, predicted []float64
			for i := range trainSetInputs {
				student.Forward(trainSetInputs[i])
				pred := float64(paragon.ArgMax(student.ExtractOutput()))
				trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
				expected = append(expected, trueLabel)
				predicted = append(predicted, pred)
			}
			student.EvaluateModel(expected, predicted)
			fmt.Printf("%-12.2f %-12.3f %-20.2f\n", errOffset, lr, student.Performance.Score)
		}
	}
}

func runStudentDistillationPermuteErrLRExtreme(
	trainSetInputs [][][]float64,
	trainSetTargets [][][]float64,
	nn *paragon.Network,
) {
	fmt.Println("\n---------Student Network Permuted Error/LR Sweep (🧪 Extreme Range Exploration)----------")

	maxUpdate := 0.10
	damping := 0.01

	errs := []float64{-1.0, -0.5, -0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 0.3, 0.5, 1.0}
	lrs := []float64{0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0}

	fmt.Printf("%-12s %-12s %-20s\n", "Error", "LR", "Student ADHD")

	for _, errVal := range errs {
		for _, lr := range lrs {
			student := createStudentNet()

			for i := range trainSetInputs {
				input := trainSetInputs[i]
				adjustNetworkUpstream(student, input, errVal, lr, maxUpdate, damping)
			}

			var expected, predicted []float64
			for i := range trainSetInputs {
				student.Forward(trainSetInputs[i])
				pred := float64(paragon.ArgMax(student.ExtractOutput()))
				trueLabel := float64(paragon.ArgMax(trainSetTargets[i][0]))
				expected = append(expected, trueLabel)
				predicted = append(predicted, pred)
			}

			student.EvaluateModel(expected, predicted)
			fmt.Printf("%-12.2f %-12.4f %-20.2f\n", errVal, lr, student.Performance.Score)
		}
	}
}
