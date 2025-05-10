package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"runtime"
	"strings"

	"paragon"
)

const (
	bookURL        = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
	keepParagraphs = 300
	seqLen         = 6
	maxVocab       = 256
	epochsStd      = 10
	epochsDyn      = 10
	baseLR         = 0.002
	valSplit       = 0.2
	genTokens      = 12
	fixedSeed      = 1337
)

var (
	// package‐level BPE model, so helper funcs like oneHotSeq can see it
	bpe         *BPE
	demoPrompts = []string{"holmes", "time", "night", "door", "murder", "detective"}
)

// result holds accuracy + generations for each mode
type result struct {
	mode string
	acc  float64
	gen  map[string]string
}

func main() {
	runtime.GOMAXPROCS(int(0.8 * float64(runtime.NumCPU())))
	rand.Seed(fixedSeed)

	fmt.Println("Fetching text...")
	paras := fetchParagraphs(bookURL, keepParagraphs)

	fmt.Println("Learning BPE merges…")
	bpe = TrainBPE(paras, maxVocab)
	fmt.Printf("✓ learned BPE vocab size %d tokens\n", len(bpe.vocab))

	inputs, targets := generateClozeData(bpe, paras, "[MASK]")
	trainX, trainY, valX, valY := prepareClozeData(bpe, inputs, targets)

	template := buildModel(seqLen*len(bpe.vocab), len(bpe.vocab))
	modes := []string{"standard", "static", "dynamic"}

	for _, mode := range modes {
		net := Clone(template)
		switch mode {
		case "standard":
			trainWithProgress(net, trainX, trainY, epochsStd, mode)
		case "static":
			trainWithProgress(net, trainX, trainY, epochsStd, mode)
		case "dynamic":
			trainWithProgress(net, trainX, trainY, epochsDyn, mode)
		}

		// Evaluation
		acc := accuracy(net, valX, valY)
		fmt.Printf("\n%s done. val-acc: %.2f%%\n", mode, 100*acc)

		// Test predictions
		for i := 0; i < 5; i++ {
			fmt.Printf("\nPrompt: %s\n", inputs[i])
			fmt.Printf("Truth : %s\n", targets[i])
			fmt.Printf("Guess : %s\n", predictMaskedWord(net, inputs[i]))
		}
	}
}

/* ── BPE training & encoding ───────────────────────────────── */

// BPE holds the learned merges and token→ID map.
type BPE struct {
	vocab     map[string]int
	merges    [][2]string
	endOfWord string
}

// TrainBPE learns up to `vocabSize` tokens from your paragraphs.
func TrainBPE(paras []string, vocabSize int) *BPE {
	wordFreqs := map[string]int{}
	for _, p := range paras {
		for _, w := range strings.Fields(strings.ToLower(p)) {
			if w == "" {
				continue
			}
			chars := strings.Split(w, "")
			chars = append(chars, "</w>")
			key := strings.Join(chars, " ")
			wordFreqs[key]++
		}
	}

	vocab := map[string]int{"<unk>": 0}
	for seq := range wordFreqs {
		for _, sym := range strings.Fields(seq) {
			if _, ok := vocab[sym]; !ok {
				vocab[sym] = len(vocab)
			}
		}
	}

	merges := [][2]string{}
	for len(vocab) < vocabSize {
		pairFreq := map[[2]string]int{}
		for seq, freq := range wordFreqs {
			toks := strings.Fields(seq)
			for i := 0; i < len(toks)-1; i++ {
				p := [2]string{toks[i], toks[i+1]}
				pairFreq[p] += freq
			}
		}
		if len(pairFreq) == 0 {
			break
		}
		var best [2]string
		maxCount := 0
		for p, c := range pairFreq {
			if c > maxCount {
				best, maxCount = p, c
			}
		}
		if maxCount < 1 {
			break
		}
		merges = append(merges, best)
		newSym := best[0] + best[1]
		vocab[newSym] = len(vocab)

		newWordFreqs := map[string]int{}
		pairStr := best[0] + " " + best[1]
		for seq, freq := range wordFreqs {
			seq2 := strings.ReplaceAll(seq, pairStr, newSym)
			seq2 = strings.Join(strings.Fields(seq2), " ")
			newWordFreqs[seq2] = freq
		}
		wordFreqs = newWordFreqs
	}

	return &BPE{vocab: vocab, merges: merges, endOfWord: "</w>"}
}

// Encode applies your learned merges to `text` and returns subword tokens.
func (b *BPE) Encode(text string) []string {
	out := []string{}
	for _, w := range strings.Fields(strings.ToLower(text)) {
		toks := append(strings.Split(w, ""), b.endOfWord)
		for _, m := range b.merges {
			pairStr := m[0] + " " + m[1]
			joined := strings.Join(toks, " ")
			if !strings.Contains(joined, pairStr) {
				continue
			}
			joined = strings.ReplaceAll(joined, pairStr, m[0]+m[1])
			toks = strings.Fields(joined)
		}
		for _, sym := range toks {
			if sym != b.endOfWord {
				out = append(out, sym)
			}
		}
	}
	return out
}

/* ── data preparation ─────────────────────────────────────── */

func prepareData(bpe *BPE, paras []string) (trainX, trainY, valX, valY [][][]float64) {
	var X, Y [][][]float64
	for _, p := range paras {
		toks := bpe.Encode(p)
		ids := make([]int, len(toks))
		for i, t := range toks {
			if id, ok := bpe.vocab[t]; ok {
				ids[i] = id
			} else {
				ids[i] = 0
			}
		}
		for i := 0; i+seqLen < len(ids); i++ {
			X = append(X, [][]float64{oneHotSeq(ids[i : i+seqLen]...)})
			Y = append(Y, [][]float64{oneHot(ids[i+seqLen], len(bpe.vocab))})
		}
	}
	valN := int(valSplit * float64(len(X)))
	perm := rand.Perm(len(X))
	for i, idx := range perm {
		if i < valN {
			valX = append(valX, X[idx])
			valY = append(valY, Y[idx])
		} else {
			trainX = append(trainX, X[idx])
			trainY = append(trainY, Y[idx])
		}
	}
	return
}

func fetchParagraphs(url string, n int) []string {
	resp, _ := http.Get(url)
	defer resp.Body.Close()
	buf := new(bytes.Buffer)
	io.Copy(buf, resp.Body)
	all := paragraphs(buf.String())
	rand.Shuffle(len(all), func(i, j int) { all[i], all[j] = all[j], all[i] })
	if n > len(all) {
		n = len(all)
	}
	return all[:n]
}

func paragraphs(txt string) []string {
	sc := bufio.NewScanner(strings.NewReader(txt))
	var out, cur []string
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			if len(cur) > 0 {
				out = append(out, strings.Join(cur, " "))
				cur = nil
			}
			continue
		}
		cur = append(cur, line)
	}
	if len(cur) > 0 {
		out = append(out, strings.Join(cur, " "))
	}
	return out
}

/* ── training modes ───────────────────────────────────────── */

func trainStandard(net *paragon.Network, x, y [][][]float64) {
	for ep := 0; ep < epochsStd; ep++ {
		lr := cosineLR(ep, epochsStd)
		net.Train(x, y, 1, lr, false)
	}
}

func trainStatic(net *paragon.Network, x, y [][][]float64) {
	for l := 1; l < net.OutputLayer; l++ {
		layer := &net.Layers[l]
		layer.ReplayEnabled = true
		layer.MaxReplay = 5
		layer.ReplayPhase = "before"
	}
	trainStandard(net, x, y)
}

func trainDynamic(net *paragon.Network, x, y [][][]float64) {
	for ep := 0; ep < epochsDyn; ep++ {
		budget := max(4, 10-ep/6)
		for l := 1; l < net.OutputLayer; l++ {
			layer := &net.Layers[l]
			layer.ReplayEnabled = true
			layer.ReplayBudget = budget
			layer.ReplayGateFunc = EntropyGate(layer)
			layer.ReplayGateToReps = func(e float64) int {
				return 1 + int(math.Pow(e, 1.1)*float64(layer.ReplayBudget-1))
			}
		}
		lr := cosineLR(ep, epochsDyn)
		net.Train(x, y, 1, lr, false)
	}
}

/* ── generation ───────────────────────────────────────────── */

func generateFromPrompt(net *paragon.Network, prompt string, n int) string {
	// tokenize & pad
	toks := bpe.Encode(prompt)
	var ctx []int
	if len(toks) >= seqLen {
		for i := len(toks) - seqLen; i < len(toks); i++ {
			ctx = append(ctx, bpe.vocab[toks[i]])
		}
	} else {
		pad := make([]int, seqLen-len(toks))
		ctx = append(pad, func() []int {
			out := make([]int, len(toks))
			for i, t := range toks {
				out[i] = bpe.vocab[t]
			}
			return out
		}()...)
	}
	seq := append([]int(nil), ctx...)

	for i := 0; i < n; i++ {
		in := [][]float64{oneHotSeq(seq[len(seq)-seqLen:]...)}
		net.Forward(in)
		net.ApplySoftmax()
		seq = append(seq, sample(net.ExtractOutput()))
	}

	// decode just the new tokens
	var out []string
	for _, id := range seq[len(ctx):] {
		for tok, tid := range bpe.vocab {
			if tid == id {
				out = append(out, tok)
				break
			}
		}
	}
	return strings.Join(out, "")
}

/* ── utilities ─────────────────────────────────────────────── */

func oneHotSeq(ids ...int) []float64 {
	vec := make([]float64, len(ids)*len(bpe.vocab))
	for pos, id := range ids {
		vec[pos*len(bpe.vocab)+id] = 1
	}
	return vec
}

func oneHot(id, size int) []float64 {
	v := make([]float64, size)
	v[id] = 1
	return v
}

func sample(p []float64) int {
	r, s := rand.Float64(), 0.0
	for i, v := range p {
		s += v
		if r < s {
			return i
		}
	}
	return len(p) - 1
}

func accuracy(net *paragon.Network, X, Y [][][]float64) float64 {
	c := 0
	for i, in := range X {
		net.Forward(in)
		net.ApplySoftmax()
		if paragon.ArgMax(net.ExtractOutput()) == paragon.ArgMax(Y[i][0]) {
			c++
		}
	}
	return float64(c) / float64(len(X))
}

func cosineLR(step, total int) float64 {
	t := float64(step) / float64(total-1)
	return baseLR * 0.5 * (1 + math.Cos(math.Pi*t))
}

func buildModel(in, out int) *paragon.Network {
	layers := []struct{ Width, Height int }{{in, 1}, {256, 1}, {128, 1}, {out, 1}}
	acts := []string{"leaky_relu", "leaky_relu", "leaky_relu", "softmax"}
	full := []bool{true, true, true, true}
	return paragon.NewNetwork(layers, acts, full)
}

func EntropyGate(layer *paragon.Grid) func([][]float64) float64 {
	return func(_ [][]float64) float64 {
		if len(layer.CachedOutputs) == 0 {
			return 0
		}
		out := paragon.Softmax(layer.CachedOutputs)
		h := 0.0
		for _, p := range out {
			if p > 0 {
				h -= p * math.Log2(p)
			}
		}
		return h / math.Log2(float64(len(out)))
	}
}

func Clone(n *paragon.Network) *paragon.Network {
	b, _ := n.MarshalJSONModel()
	c := &paragon.Network{}
	_ = c.UnmarshalJSONModel(b)
	return c
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func trainWithProgress(net *paragon.Network, x, y [][][]float64, epochs int, mode string) {
	//fmt.Printf("\n--- TRAINING: %s ---\n", mode)
	for ep := 0; ep < epochs; ep++ {
		switch mode {
		case "static":
			for l := 1; l < net.OutputLayer; l++ {
				layer := &net.Layers[l]
				layer.ReplayEnabled = true
				layer.MaxReplay = 5
				layer.ReplayPhase = "before"
			}
		case "dynamic":
			budget := max(4, 10-ep/6)
			for l := 1; l < net.OutputLayer; l++ {
				layer := &net.Layers[l]
				layer.ReplayEnabled = true
				layer.ReplayBudget = budget
				layer.ReplayGateFunc = EntropyGate(layer)
				layer.ReplayGateToReps = func(e float64) int {
					return 1 + int(math.Pow(e, 1.1)*float64(layer.ReplayBudget-1))
				}
			}
		}

		lr := cosineLR(ep, epochs)
		net.Train(x, y, 1, lr, false)

		// Optional: estimate current loss manually
		totalLoss := 0.0
		for i, in := range x[:min(100, len(x))] { // sample subset
			net.Forward(in)
			out := net.ExtractOutput()
			label := y[i][0]
			totalLoss += crossEntropy(out, label)
		}
		avgLoss := totalLoss / float64(min(100, len(x)))
		fmt.Printf("[Epoch %2d/%d] approx loss: %.4f\n", ep+1, epochs, avgLoss)
	}
}

func crossEntropy(pred, label []float64) float64 {
	eps := 1e-9
	loss := 0.0
	for i := range pred {
		loss -= label[i] * math.Log(pred[i]+eps)
	}
	return loss
}

func generateClozeData(bpe *BPE, paras []string, maskToken string) ([]string, []string) {
	var inputs, targets []string
	for _, p := range paras {
		words := strings.Fields(p)
		if len(words) < 6 {
			continue
		}
		// Pick a random word to mask
		maskIndex := rand.Intn(len(words))
		target := words[maskIndex]
		words[maskIndex] = maskToken
		input := strings.Join(words, " ")
		inputs = append(inputs, input)
		targets = append(targets, target)
	}
	return inputs, targets
}

func prepareClozeData(bpe *BPE, inputs, targets []string) (trainX, trainY, valX, valY [][][]float64) {
	var X, Y [][][]float64
	for i := range inputs {
		toks := bpe.Encode(inputs[i])
		targetToks := bpe.Encode(targets[i])
		if len(toks) < seqLen || len(targetToks) == 0 {
			continue
		}

		// Crop/pad sequence
		if len(toks) > seqLen {
			toks = toks[len(toks)-seqLen:]
		} else {
			pad := make([]string, seqLen-len(toks))
			toks = append(pad, toks...)
		}

		ids := make([]int, seqLen)
		for j, tok := range toks {
			ids[j] = bpe.vocab[tok]
		}

		targetID := bpe.vocab[targetToks[0]]
		X = append(X, [][]float64{oneHotSeq(ids...)})
		Y = append(Y, [][]float64{oneHot(targetID, len(bpe.vocab))})
	}

	valN := int(valSplit * float64(len(X)))
	perm := rand.Perm(len(X))
	for i, idx := range perm {
		if i < valN {
			valX = append(valX, X[idx])
			valY = append(valY, Y[idx])
		} else {
			trainX = append(trainX, X[idx])
			trainY = append(trainY, Y[idx])
		}
	}
	return
}

func predictMaskedWord(net *paragon.Network, sentence string) string {
	toks := bpe.Encode(sentence)
	if len(toks) < seqLen {
		pad := make([]string, seqLen-len(toks))
		toks = append(pad, toks...)
	} else {
		toks = toks[len(toks)-seqLen:]
	}

	ids := make([]int, seqLen)
	for i, tok := range toks {
		ids[i] = bpe.vocab[tok]
	}

	in := [][]float64{oneHotSeq(ids...)}
	net.Forward(in)
	net.ApplySoftmax()
	out := net.ExtractOutput()
	id := paragon.ArgMax(out)

	for tok, tid := range bpe.vocab {
		if tid == id {
			return tok
		}
	}
	return "<unk>"
}
