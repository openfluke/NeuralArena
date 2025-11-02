package main

import (
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"

	"paragon"
)

/* ========= helpers ========= */

func sha12(v []float64) string {
	h := sha1.New()
	for _, x := range v {
		fmt.Fprintf(h, "%.9f,", x)
	}
	return hex.EncodeToString(h.Sum(nil))[:12]
}

func printFull(out []float64, label string) {
	fmt.Println(label)
	for i := range out {
		if i%8 == 0 && i != 0 {
			fmt.Println()
		}
		fmt.Printf(" %.15e", out[i])
	}
	fmt.Println()
}

func forwardReturn[T paragon.Numeric](n *paragon.Network[T], X [][]float64) []float64 {
	n.Forward(X)
	return n.GetOutput()
}

/* ========= deterministic inputs ========= */

func zeroInput(h, w int) [][]float64 {
	X := make([][]float64, h)
	for r := 0; r < h; r++ {
		X[r] = make([]float64, w)
	}
	return X
}
func fixedTwoQuery(h, w int) [][]float64 {
	X := zeroInput(h, w)
	X[1][6] = 1
	X[4][2] = 1
	return X
}
func randInput(h, w int, seed int64) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	X := make([][]float64, h)
	for r := 0; r < h; r++ {
		row := make([]float64, w)
		for c := 0; c < w; c++ {
			row[c] = rng.Float64()
		}
		X[r] = row
	}
	return X
}

/* ========= model builders (same layout) ========= */

func buildShapes() ([]paragon.GridSpec, []string, []bool) {
	// 8x8 -> 4x4 -> 4x4 -> 1x4
	return []paragon.GridSpec{
			{Width: 8, Height: 8},
			{Width: 4, Height: 4},
			{Width: 4, Height: 4}, // attn/dense mix
			{Width: 4, Height: 1},
		},
		[]string{"relu", "relu", "relu", "softmax"},
		[]bool{false, true, true, true}
}

type attnKnobs[T paragon.Numeric] struct {
	share       string // "layer" | "per-slice"
	useNorm     bool
	useReplay   bool
	forceReplay bool
	posEncAmp   float64
	normEps     float64
	dk          int
	heads       int
}

func buildDense[T paragon.Numeric]() *paragon.Network[T] {
	sizes, acts, fully := buildShapes()
	n, err := paragon.BuildGridNet[T](paragon.BuildOpts[T]{
		Sizes: sizes, Activations: acts, FullyConn: fully,
	})
	if err != nil {
		panic(err)
	}
	return n
}

func buildAttnMix[T paragon.Numeric](k attnKnobs[T]) *paragon.Network[T] {
	sizes, acts, fully := buildShapes()
	// layer 2 mix: dense/attn/dense/attn
	slices := make([][]string, len(sizes))
	slices[2] = []string{"dense", "attn", "dense", "attn"}

	attn := make([]*paragon.AttnConfig[T], len(sizes))
	attn[2] = &paragon.AttnConfig[T]{
		Heads:       k.heads,
		DK:          k.dk,
		UseWo:       true,
		Share:       k.share,
		Dropout:     0,
		PosEnc2D:    true,
		UseNorm:     k.useNorm,
		PosEncAmp:   k.posEncAmp,
		NormEps:     k.normEps,
		UseReplay:   k.useReplay,
		ForceReplay: k.forceReplay,
		ReplayGain:  1.1,
	}

	n, err := paragon.BuildGridNet[T](paragon.BuildOpts[T]{
		Sizes: sizes, Activations: acts, FullyConn: fully,
		SliceTypes: slices, Attn: attn,
	})
	if err != nil {
		panic(err)
	}
	return n
}

/* ========= save / load ========= */

func saveBundle(path, id string, n *paragon.Network[float32]) {
	js, err := paragon.ExportBundleJSON[float32](id, n, nil)
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(path, []byte(js), 0o644); err != nil {
		panic(err)
	}
}

func loadBundle(path string) *paragon.Network[float32] {
	b, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	anyNet, err := paragon.ImportBundleJSON(string(b))
	if err != nil {
		panic(err)
	}
	rt, ok := anyNet.(*paragon.Network[float32])
	if !ok {
		panic("bundle type mismatch")
	}
	return rt
}

/* ========= main ========= */

func main() {
	outDir := "bundles_v3"
	_ = os.MkdirAll(outDir, 0o755)

	type spec struct {
		name string
		make func() *paragon.Network[float32]
	}

	perHeadDK := map[int]int{1: 64, 2: 32, 3: 24}
	var variants []spec

	// dense baseline
	variants = append(variants, spec{
		name: "dense",
		make: func() *paragon.Network[float32] { return buildDense[float32]() },
	})

	// attention variants across heads 1,2,3
	type vcfg struct {
		base, share         string
		norm, replay, force bool
	}
	bases := []vcfg{
		{"attnLayer_noNorm", "layer", false, false, false},
		{"attnLayer_norm", "layer", true, false, false},
		{"attnPerSlice_noNorm", "per-slice", false, false, false},
		{"attnPerSlice_norm", "per-slice", true, false, false},
		{"attnLayer_norm_replay", "layer", true, true, true},
	}
	for _, h := range []int{1, 2, 3} {
		for _, b := range bases {
			name := fmt.Sprintf("%s-H%d", b.base, h)
			hh := h
			v := b
			variants = append(variants, spec{
				name: name,
				make: func() *paragon.Network[float32] {
					return buildAttnMix[float32](attnKnobs[float32]{
						share: v.share, useNorm: v.norm,
						useReplay: v.replay, forceReplay: v.force,
						posEncAmp: 0.02, normEps: 1e-6,
						dk: perHeadDK[hh], heads: hh,
					})
				},
			})
		}
	}

	// Prepare deterministic inputs
	X0 := zeroInput(8, 8)
	X2Q := fixedTwoQuery(8, 8)
	XR := randInput(8, 8, 12345)

	fmt.Println("=== Forward outputs (load if exists, else build+save) ===")
	for _, sp := range variants {
		path := filepath.Join(outDir, sp.name+".bundle.json")

		var net *paragon.Network[float32]
		if _, err := os.Stat(path); err == nil {
			// load existing
			net = loadBundle(path)
		} else {
			// build fresh and save once
			net = sp.make()
			// a forward ensures lazy params (e.g., attn) materialize
			_ = forwardReturn(net, X0)
			saveBundle(path, sp.name, net)
		}

		// do forwards and print FULL vectors (plus short hashes)
		o0 := forwardReturn(net, X0)
		o2 := forwardReturn(net, X2Q)
		or := forwardReturn(net, XR)

		fmt.Printf("\n[%s]\n", sp.name)
		fmt.Printf(" zero  hash=%s len=%d\n", sha12(o0), len(o0))
		printFull(o0, "  zero :")

		fmt.Printf(" twoQ  hash=%s len=%d\n", sha12(o2), len(o2))
		printFull(o2, "  twoQ :")

		fmt.Printf(" rand  hash=%s len=%d\n", sha12(or), len(or))
		printFull(or, "  rand :")
	}

	fmt.Println("\nDone.")
}
