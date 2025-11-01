package main

import (
	"fmt"
	"math/rand"
	"time"

	"paragon"
)

// helper: make an H×W input grid in float64
func mkInput(h, w int) [][]float64 {
	in := make([][]float64, h)
	for y := 0; y < h; y++ {
		in[y] = make([]float64, w)
		for x := 0; x < w; x++ {
			in[y][x] = rand.Float64()*2 - 1 // [-1,1]
		}
	}
	return in
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// ------------------------------
	// Shared model spec
	// ------------------------------
	layerSizes := []struct{ Width, Height int }{
		{8, 8}, // input
		{8, 8}, // mixed layer (we’ll toggle attention here)
		{4, 4}, // output
	}
	activations := []string{"relu", "relu", "softmax"}
	fully := []bool{false, true, true}

	// ------------------------------
	// A) Dense-only network
	// ------------------------------
	netDense, err := paragon.NewNetwork[float32](layerSizes, activations, fully)
	if err != nil {
		panic(err)
	}
	// (SliceTypes default to "dense" for every column)

	// ------------------------------
	// B) Mixed network with attention columns
	// ------------------------------
	netAttn, err := paragon.NewNetwork[float32](layerSizes, activations, fully)
	if err != nil {
		panic(err)
	}

	// Mark alternating columns in layer 1 as attention.
	// Use whichever API you implemented:
	//  - If you added SetSliceTypes:
	//      netAttn.SetSliceTypes(1, []string{"dense","attn","dense","attn","dense","attn","dense","attn"})
	//  - If not, set directly:
	netAttn.Layers[1].SliceTypes = []string{"dense", "attn", "dense", "attn", "dense", "attn", "dense", "attn"}

	// Attach attention config (single-head, shared across those attn columns).
	netAttn.Layers[1].Attn = &paragon.AttnConfig[float32]{
		DK:       48,      // head size
		UseWo:    true,    // project to column via Wo (dk × Hcurr)
		Share:    "layer", // one param set shared by all attn columns
		Dropout:  0.0,     // (used only during training if you add it)
		PosEnc2D: true,    // tiny 2D pos enc to help localization
	}

	// ------------------------------
	// Forward both models on the same input
	// ------------------------------
	in := mkInput(layerSizes[0].Height, layerSizes[0].Width)

	netDense.Forward(in)
	outDense := netDense.GetOutput()

	netAttn.Forward(in)
	outAttn := netAttn.GetOutput()

	// ------------------------------
	// Print a quick diff
	// ------------------------------
	fmt.Println("Dense-only output (first 8 vals):")
	for i := 0; i < len(outDense) && i < 8; i++ {
		fmt.Printf(" %.4f", outDense[i])
	}
	fmt.Println()

	fmt.Println("Mixed (dense+attn) output (first 8 vals):")
	for i := 0; i < len(outAttn) && i < 8; i++ {
		fmt.Printf(" %.4f", outAttn[i])
	}
	fmt.Println()

	// Optional: show which columns are attention in layer 1
	fmt.Println("Layer 1 slice types:", netAttn.Layers[1].SliceTypes)
}
