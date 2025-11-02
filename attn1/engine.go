package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"paragon"
)

/* ===== helpers ===== */

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func mkInput(h, w int, r *rand.Rand) [][]float64 {
	in := make([][]float64, h)
	for y := 0; y < h; y++ {
		in[y] = make([]float64, w)
		for x := 0; x < w; x++ {
			in[y][x] = r.Float64()*2 - 1 // [-1,1]
		}
	}
	return in
}

func loadBundle[T paragon.Numeric](path string) (*paragon.Network[T], error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	nAny, err := paragon.ImportBundleJSON(string(b))
	if err != nil {
		return nil, err
	}
	n, ok := nAny.(*paragon.Network[T])
	if !ok {
		return nil, fmt.Errorf("bundle type mismatch for %s", path)
	}
	return n, nil
}

func createAndSave[T paragon.Numeric](
	path, id string,
	builder func() (*paragon.Network[T], error),
) (*paragon.Network[T], error) {
	net, err := builder()
	if err != nil {
		return nil, err
	}
	seed := time.Now().UnixNano() // only used to annotate the bundle
	jsonStr, err := paragon.ExportBundleJSON(id, net, &seed)
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(path, []byte(jsonStr), 0o644); err != nil {
		return nil, err
	}
	return net, nil
}

func loadOrCreate[T paragon.Numeric](
	path, id string,
	builder func() (*paragon.Network[T], error),
) (*paragon.Network[T], error) {
	if fileExists(path) {
		return loadBundle[T](path)
	}
	return createAndSave[T](path, id, builder)
}

/* ===== main ===== */

func main() {
	// Use a FIXED seed for inputs so repeated runs are comparable.
	// Change to time.Now().UnixNano() if you want fresh inputs each run.
	inputRng := rand.New(rand.NewSource(1337))

	// Shared model spec
	layerSizes := []struct{ Width, Height int }{
		{8, 8},
		{8, 8},
		{4, 4},
	}
	activations := []string{"relu", "relu", "softmax"}
	fully := []bool{false, true, true}

	// A) Dense-only (load if exists; else create once and save)
	densePath := "dense_bundle.json"
	netDense, err := loadOrCreate[float32](densePath, "dense_v1", func() (*paragon.Network[float32], error) {
		return paragon.NewNetwork[float32](layerSizes, activations, fully)
	})
	if err != nil {
		panic(err)
	}

	// B) Mixed attention (same rule: load if exists; else create+save once)
	attnPath := "attn_bundle.json"
	netAttn, err := loadOrCreate[float32](attnPath, "mixed_attn_v1", func() (*paragon.Network[float32], error) {
		n, err := paragon.NewNetwork[float32](layerSizes, activations, fully)
		if err != nil {
			return nil, err
		}
		n.Layers[1].SliceTypes = []string{"dense", "attn", "dense", "attn", "dense", "attn", "dense", "attn"}
		n.Layers[1].Attn = &paragon.AttnConfig[float32]{
			DK:       48,
			UseWo:    true,
			Share:    "layer",
			Dropout:  0.0,
			PosEnc2D: true,
		}
		return n, nil
	})
	if err != nil {
		panic(err)
	}

	// Forward both on the SAME deterministic input
	in := mkInput(layerSizes[0].Height, layerSizes[0].Width, inputRng)

	netDense.Forward(in)
	outDense := netDense.GetOutput()

	netAttn.Forward(in)
	outAttn := netAttn.GetOutput()

	fmt.Println("Dense-only output (first 8 vals):")
	for i := 0; i < len(outDense) && i < 8; i++ {
		fmt.Printf(" %.15e", outDense[i])
	}
	fmt.Println()

	fmt.Println("Mixed (dense+attn) output (first 8 vals):")
	for i := 0; i < len(outAttn) && i < 8; i++ {
		fmt.Printf(" %.15e", outAttn[i])
	}
	fmt.Println()

	fmt.Println("Layer 1 slice types (attn model):", netAttn.Layers[1].SliceTypes)
}
