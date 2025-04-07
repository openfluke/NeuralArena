package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"paragon"
)

func main() {
	// 1. Download & load the MNIST data
	if err := ensureMNIST("mnist_data"); err != nil {
		log.Fatalf("MNIST setup failed: %v", err)
	}
	trainIn, trainT, _ := loadMNIST("mnist_data", true)
	testIn, testT, _ := loadMNIST("mnist_data", false)

	// Split data into training and validation sets
	trainIn, trainT, valIn, valT := paragon.SplitDataset(trainIn, trainT, 0.8)
	fmt.Printf("Data samples → Train:%d, Val:%d, Test:%d\n", len(trainIn), len(valIn), len(testIn))

	staticParitition := true
	if staticParitition {
		// 2. Build a network partitioned by tags (even=0, odd=1)
		layerSizes := []struct{ Width, Height int }{
			{28, 28}, // Input layer: 28x28
			{16, 16}, // Hidden layer: 16x16 (split into two regions)
			{10, 1},  // Output layer: 10x1 (shared, 10 classes)
		}
		activations := []string{"leaky_relu", "leaky_relu", "softmax"}
		fullyConnected := []bool{true, false, true}

		net := paragon.NewNetwork(layerSizes, activations, fullyConnected)

		// 3. Train the network with two passes per epoch
		const epochs = 10
		const lr = 0.01
		const totalTags = 2

		for e := 0; e < epochs; e++ {
			// Train on even digits (tag 0)
			trainSingleTag(net, trainIn, trainT, 0, totalTags, lr)
			// Train on odd digits (tag 1)
			trainSingleTag(net, trainIn, trainT, 1, totalTags, lr)
			fmt.Printf("Completed epoch %d/%d\n", e+1, epochs)
		}

		// 4. Evaluate on train, validation, and test sets
		fmt.Println("\n=== Final Combined Evaluation (Tag Partition) ===")
		fmt.Printf("Train Accuracy: %.2f%%\n", evalPartition(net, trainIn, trainT, totalTags)*100)
		fmt.Printf("Val   Accuracy: %.2f%%\n", evalPartition(net, valIn, valT, totalTags)*100)
		fmt.Printf("Test  Accuracy: %.2f%%\n", evalPartition(net, testIn, testT, totalTags)*100)
	} else {
		fmt.Println("tried so many things and nothing meeting requirements")
	}

}

// trainSingleTag trains the network on samples matching the desired tag's parity
func trainSingleTag(net *paragon.Network, inputs, targets [][][]float64, desiredTag, totalTags int, lr float64) {
	for i := range inputs {
		label := argMax(targets[i][0])
		if label%2 != desiredTag {
			continue
		}
		net.ForwardTagged(inputs[i], totalTags, desiredTag)
		net.BackwardTagged(targets[i], lr, totalTags, desiredTag)
	}
}

// evalPartition evaluates accuracy by switching partitions based on label parity
func evalPartition(net *paragon.Network, inputs, targets [][][]float64, totalTags int) float64 {
	correct := 0
	for i := range inputs {
		label := argMax(targets[i][0])
		tag := label % totalTags // 0 for even, 1 for odd
		net.ForwardTagged(inputs[i], totalTags, tag)
		pred := argMaxNeurons(net.Layers[net.OutputLayer].Neurons[0])
		if pred == label {
			correct++
		}
	}
	return float64(correct) / float64(len(inputs))
}

// argMax returns the index of the largest element in a slice
func argMax(arr []float64) int {
	maxIdx := 0
	for i := 1; i < len(arr); i++ {
		if arr[i] > arr[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}

// argMaxNeurons returns the index of the neuron with the highest value
func argMaxNeurons(neurons []*paragon.Neuron) int {
	maxIdx := 0
	maxVal := neurons[0].Value
	for i, n := range neurons {
		if n.Value > maxVal {
			maxVal = n.Value
			maxIdx = i
		}
	}
	return maxIdx
}

// -------------------- MNIST Helpers --------------------

func ensureMNIST(dir string) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	files := []struct{ gz, out string }{
		{"train-images-idx3-ubyte.gz", "train-images-idx3-ubyte"},
		{"train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte"},
		{"t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte"},
		{"t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte"},
	}
	for _, f := range files {
		gzPath := filepath.Join(dir, f.gz)
		outPath := filepath.Join(dir, f.out)
		if _, err := os.Stat(outPath); os.IsNotExist(err) {
			if _, err := os.Stat(gzPath); os.IsNotExist(err) {
				if err := download(baseURL+f.gz, gzPath); err != nil {
					return err
				}
			}
			if err := unzip(gzPath, outPath); err != nil {
				return err
			}
		}
	}
	return nil
}

const baseURL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

func download(url, dest string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, resp.Body)
	return err
}

func unzip(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	gz, err := gzip.NewReader(in)
	if err != nil {
		return err
	}
	defer gz.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, gz)
	return err
}

func loadMNIST(dir string, train bool) ([][][]float64, [][][]float64, error) {
	prefix := "t10k"
	if train {
		prefix = "train"
	}
	imgPath := filepath.Join(dir, prefix+"-images-idx3-ubyte")
	lblPath := filepath.Join(dir, prefix+"-labels-idx1-ubyte")

	fImg, err := os.Open(imgPath)
	if err != nil {
		return nil, nil, err
	}
	defer fImg.Close()
	var hdr [16]byte
	if _, err := fImg.Read(hdr[:]); err != nil {
		return nil, nil, err
	}
	num := int(binary.BigEndian.Uint32(hdr[4:8]))
	buf := make([]byte, 28*28)

	fLbl, err := os.Open(lblPath)
	if err != nil {
		return nil, nil, err
	}
	defer fLbl.Close()
	var lh [8]byte
	if _, err := fLbl.Read(lh[:]); err != nil {
		return nil, nil, err
	}

	inp := make([][][]float64, num)
	tgt := make([][][]float64, num)
	for i := 0; i < num; i++ {
		if _, err := fImg.Read(buf); err != nil {
			return nil, nil, err
		}
		img := make([][]float64, 28)
		for r := 0; r < 28; r++ {
			img[r] = make([]float64, 28)
			for c := 0; c < 28; c++ {
				img[r][c] = float64(buf[r*28+c]) / 255.0
			}
		}
		inp[i] = img

		var lb [1]byte
		if _, err := fLbl.Read(lb[:]); err != nil {
			return nil, nil, err
		}
		oneHot := make([][]float64, 1)
		oneHot[0] = make([]float64, 10)
		oneHot[0][int(lb[0])] = 1.0
		tgt[i] = oneHot
	}
	return inp, tgt, nil
}
