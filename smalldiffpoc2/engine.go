package main

import (
	"fmt"
	"math"
	"math/rand"
	"paragon" // Assuming this is a custom neural network library
	"strings"
	"time"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	// Expanded dataset with 100 famous quotes for better diversity
	sentences := []string{
		"The only thing we have to fear is fear itself. - Franklin D. Roosevelt",
		"I think, therefore I am. - René Descartes",
		"To be or not to be, that is the question. - William Shakespeare",
		"In the middle of difficulty lies opportunity. - Albert Einstein",
		"The unexamined life is not worth living. - Socrates",
		"I have a dream. - Martin Luther King Jr.",
		"Two things are infinite: the universe and human stupidity. - Albert Einstein",
		"The best way to predict the future is to create it. - Peter Drucker",
		"Life is what happens when you're busy making other plans. - John Lennon",
		"Get busy living or get busy dying. - Stephen King",
		"You miss 100% of the shots you don't take. - Wayne Gretzky",
		"The greatest glory in living lies not in never falling, but in rising every time we fall. - Nelson Mandela",
		"The way to get started is to quit talking and begin doing. - Walt Disney",
		"Your time is limited, so don't waste it living someone else's life. - Steve Jobs",
		"If you look at what you have in life, you'll always have more. - Oprah Winfrey",
		"If you set your goals ridiculously high and it's a failure, you will fail above everyone else's success. - James Cameron",
		"Life is what we make it, always has been, always will be. - Grandma Moses",
		"The only impossible journey is the one you never begin. - Tony Robbins",
		"Believe you can and you're halfway there. - Theodore Roosevelt",
		"The purpose of our lives is to be happy. - Dalai Lama",
		"Success is not final, failure is not fatal: It is the courage to continue that counts. - Winston Churchill",
		"It does not matter how slowly you go as long as you do not stop. - Confucius",
		"Everything you’ve ever wanted is on the other side of fear. - George Addair",
		"You must be the change you wish to see in the world. - Mahatma Gandhi",
		"The mind is everything. What you think you become. - Buddha",
		"Strive not to be a success, but rather to be of value. - Albert Einstein",
		"I am not a product of my circumstances. I am a product of my decisions. - Stephen Covey",
		"The only way to do great work is to love what you do. - Steve Jobs",
		"It’s not whether you get knocked down, it’s whether you get up. - Vince Lombardi",
		"The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt",
		"Do what you can, with what you have, where you are. - Theodore Roosevelt",
		"Whatever you are, be a good one. - Abraham Lincoln",
		"Happiness is not something ready made. It comes from your own actions. - Dalai Lama",
		"Act as if what you do makes a difference. It does. - William James",
		"Success is to be measured not so much by the position that one has reached in life as by the obstacles which he has overcome. - Booker T. Washington",
		"You can never cross the ocean until you have the courage to lose sight of the shore. - Christopher Columbus",
		"Whether you think you can or you think you can’t, you’re right. - Henry Ford",
		"The best revenge is massive success. - Frank Sinatra",
		"Don’t judge each day by the harvest you reap but by the seeds that you plant. - Robert Louis Stevenson",
		"Challenges are what make life interesting and overcoming them is what makes life meaningful. - Joshua J. Marine",
		"The harder the conflict, the more glorious the triumph. - Thomas Paine",
		"Life is 10% what happens to me and 90% of how I react to it. - Charles Swindoll",
		"What lies behind us and what lies before us are tiny matters compared to what lies within us. - Ralph Waldo Emerson",
		"Opportunity does not knock, it presents itself when you beat down the door. - Kyle Chandler",
		"Don’t let yesterday take up too much of today. - Will Rogers",
		"It is never too late to be what you might have been. - George Eliot",
		"Every strike brings me closer to the next home run. - Babe Ruth",
		"The journey of a thousand miles begins with one step. - Lao Tzu",
		"You only live once, but if you do it right, once is enough. - Mae West",
		"Keep your face always toward the sunshine—and shadows will fall behind you. - Walt Whitman",
		"The real test is not whether you avoid this failure, because you won’t. It’s whether you let it harden or shame you into inaction, or whether you learn from it; whether you choose to persevere. - Barack Obama",
		"If you’re going through hell, keep going. - Winston Churchill",
		"Turn your wounds into wisdom. - Oprah Winfrey",
		"Failure is the condiment that gives success its flavor. - Truman Capote",
		"The only limit to our realization of tomorrow will be our doubts of today. - Franklin D. Roosevelt",
		"Do not go where the path may lead, go instead where there is no path and leave a trail. - Ralph Waldo Emerson",
		"Creativity is intelligence having fun. - Albert Einstein",
		"I’ve failed over and over and over again in my life and that is why I succeed. - Michael Jordan",
		"To handle yourself, use your head; to handle others, use your heart. - Eleanor Roosevelt",
		"Too many of us are not living our dreams because we are living our fears. - Les Brown",
		"Nothing is impossible, the word itself says ‘I’m possible’! - Audrey Hepburn",
		"Start where you are. Use what you have. Do what you can. - Arthur Ashe",
		"When you reach the end of your rope, tie a knot in it and hang on. - Franklin D. Roosevelt",
		"The secret of getting ahead is getting started. - Mark Twain",
		"In order to succeed, we must first believe that we can. - Nikos Kazantzakis",
		"Either you run the day, or the day runs you. - Jim Rohn",
		"Go confidently in the direction of your dreams. Live the life you have imagined. - Henry David Thoreau",
		"Life shrinks or expands in proportion to one’s courage. - Anaïs Nin",
		"Perfection is not attainable, but if we chase perfection we can catch excellence. - Vince Lombardi",
		"Everything has beauty, but not everyone can see. - Confucius",
		"Many of life’s failures are people who did not realize how close they were to success when they gave up. - Thomas A. Edison",
		"Once you choose hope, anything’s possible. - Christopher Reeve",
		"Try to be a rainbow in someone’s cloud. - Maya Angelou",
		"Don’t let the fear of striking out hold you back. - Babe Ruth",
		"Change your thoughts and you change your world. - Norman Vincent Peale",
		"With the new day comes new strength and new thoughts. - Eleanor Roosevelt",
		"Build your own dreams, or someone else will hire you to build theirs. - Farrah Gray",
		"It always seems impossible until it’s done. - Nelson Mandela",
		"Happiness is not by chance, but by choice. - Jim Rohn",
		"A person who never made a mistake never tried anything new. - Albert Einstein",
		"Winning isn’t everything, but wanting to win is. - Vince Lombardi",
		"The most difficult thing is the decision to act, the rest is merely tenacity. - Amelia Earhart",
		"You may be disappointed if you fail, but you are doomed if you don’t try. - Beverly Sills",
		"Be yourself; everyone else is already taken. - Oscar Wilde",
		"The best time to plant a tree was 20 years ago. The second best time is now. - Chinese Proverb",
		"Whatever you do, do it well. - Walt Disney",
		"Limitations live only in our minds. But if we use our imaginations, our possibilities become limitless. - Jamie Paolinetti",
		"Courage is what it takes to stand up and speak; courage is also what it takes to sit down and listen. - Winston Churchill",
		"Always do your best. What you plant now, you will harvest later. - Og Mandino",
		"Success is walking from failure to failure with no loss of enthusiasm. - Winston Churchill",
		"Dream big and dare to fail. - Norman Vaughan",
		"What you get by achieving your goals is not as important as what you become by achieving your goals. - Zig Ziglar",
		"Hardships often prepare ordinary people for an extraordinary destiny. - C.S. Lewis",
		"Life is either a daring adventure or nothing at all. - Helen Keller",
		"Don’t watch the clock; do what it does. Keep going. - Sam Levenson",
		"Believe in yourself! Have faith in your abilities! - Norman Vincent Peale",
		"The power of imagination makes us infinite. - John Muir",
		"Don’t be afraid to give up the good to go for the great. - John D. Rockefeller",
		"I find that the harder I work, the more luck I seem to have. - Thomas Jefferson",
	}

	// Build vocabulary from the dataset
	vocab, reverseVocab := buildVocabulary(sentences)
	vocabSize := len(vocab)

	// Model parameters adjusted for larger dataset
	maxSeqLength := 30    // Increased to handle longer sentences
	totalSteps := 5       // Number of diffusion steps (unchanged)
	epochs := 300         // More epochs for better convergence
	learningRate := 0.001 // Slightly lower learning rate for stability

	// Define network architecture
	layerSizes := []struct{ Width, Height int }{
		{Width: vocabSize * maxSeqLength, Height: 1}, // Input: flattened one-hot sequence
		{Width: 512, Height: 1},                      // Hidden layer with 512 neurons for increased capacity
		{Width: vocabSize * maxSeqLength, Height: 1}, // Output: flattened probabilities
	}
	activations := []string{"linear", "relu", "softmax"} // Output layer uses softmax
	fullyConnected := []bool{true, true, true}

	// Initialize the model
	model := paragon.NewNetwork(layerSizes, activations, fullyConnected)

	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		for _, sentence := range sentences {
			sequence := tokenize(sentence, vocab)
			padded := padSequence(sequence, maxSeqLength, vocab["[PAD]"])
			t := rand.Intn(totalSteps + 1) // Random diffusion step
			masked := maskSequence(padded, t, totalSteps, vocab)
			inputFlat := flattenOneHot(sequenceToOneHot(masked, vocabSize, maxSeqLength))
			targetFlat := flattenOneHot(sequenceToOneHot(padded, vocabSize, maxSeqLength))

			// Forward pass
			model.Forward([][]float64{inputFlat})
			outputFlat := model.GetOutput()
			loss := computeLoss(outputFlat, targetFlat)
			totalLoss += loss

			// Backward pass
			model.Backward([][]float64{targetFlat}, learningRate)
		}
		avgLoss := totalLoss / float64(len(sentences))
		if epoch%10 == 0 { // Print loss every 10 epochs to reduce clutter
			fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, avgLoss)
		}
	}

	// Generate text after training
	generated := generate(model, vocab, reverseVocab, maxSeqLength, totalSteps, vocabSize)
	fmt.Println("Generated text:", generated)
}

// buildVocabulary creates word-to-index and index-to-word mappings
func buildVocabulary(sentences []string) (map[string]int, map[int]string) {
	vocab := make(map[string]int)
	reverseVocab := make(map[int]string)
	idx := 0
	for _, sentence := range sentences {
		words := strings.Fields(sentence)
		for _, word := range words {
			if _, exists := vocab[word]; !exists {
				vocab[word] = idx
				reverseVocab[idx] = word
				idx++
			}
		}
	}
	vocab["[PAD]"] = idx
	reverseVocab[idx] = "[PAD]"
	idx++
	vocab["[MASK]"] = idx
	reverseVocab[idx] = "[MASK]"
	return vocab, reverseVocab
}

// tokenize converts a sentence to a sequence of token indices
func tokenize(sentence string, vocab map[string]int) []int {
	words := strings.Fields(sentence)
	sequence := make([]int, len(words))
	for i, word := range words {
		sequence[i] = vocab[word]
	}
	return sequence
}

// padSequence pads or truncates a sequence to a fixed length
func padSequence(sequence []int, length int, padToken int) []int {
	padded := make([]int, length)
	for i := 0; i < length; i++ {
		if i < len(sequence) {
			padded[i] = sequence[i]
		} else {
			padded[i] = padToken
		}
	}
	return padded
}

// maskSequence applies diffusion by randomly masking tokens
func maskSequence(sequence []int, step int, totalSteps int, vocab map[string]int) []int {
	masked := make([]int, len(sequence))
	copy(masked, sequence)
	maskProb := float64(step) / float64(totalSteps)
	for i := range masked {
		if rand.Float64() < maskProb {
			masked[i] = vocab["[MASK]"]
		}
	}
	return masked
}

// sequenceToOneHot converts a sequence to a one-hot encoded 2D slice
func sequenceToOneHot(sequence []int, vocabSize int, maxSeqLength int) [][]float64 {
	oneHot := make([][]float64, maxSeqLength)
	for i := 0; i < maxSeqLength; i++ {
		oneHot[i] = make([]float64, vocabSize)
		if i < len(sequence) {
			token := sequence[i]
			if token >= 0 && token < vocabSize {
				oneHot[i][token] = 1.0
			}
		}
	}
	return oneHot
}

// flattenOneHot flattens a 2D one-hot encoded sequence into a 1D slice
func flattenOneHot(oneHot [][]float64) []float64 {
	flat := make([]float64, len(oneHot)*len(oneHot[0]))
	for i := 0; i < len(oneHot); i++ {
		for j := 0; j < len(oneHot[i]); j++ {
			flat[i*len(oneHot[0])+j] = oneHot[i][j]
		}
	}
	return flat
}

// computeLoss calculates cross-entropy loss between output and target probabilities
func computeLoss(output, target []float64) float64 {
	loss := 0.0
	for i := range output {
		outputVal := output[i]
		if outputVal <= 0 {
			outputVal = 1e-10 // Avoid log(0)
		}
		loss += -target[i] * math.Log(outputVal)
	}
	return loss
}

// generate produces text by iteratively denoising a masked sequence
func generate(model *paragon.Network, vocab map[string]int, reverseVocab map[int]string, maxSeqLength, steps, vocabSize int) string {
	sequence := make([]int, maxSeqLength)
	for i := range sequence {
		sequence[i] = vocab["[MASK]"] // Start with a fully masked sequence
	}
	for step := steps; step > 0; step-- {
		inputFlat := flattenOneHot(sequenceToOneHot(sequence, vocabSize, maxSeqLength))
		model.Forward([][]float64{inputFlat})
		outputFlat := model.GetOutput()
		output := make([][]float64, maxSeqLength)
		for i := 0; i < maxSeqLength; i++ {
			output[i] = outputFlat[i*vocabSize : (i+1)*vocabSize]
			probs := output[i] // Use output directly as probabilities (softmax applied in model)
			sequence[i] = argMax(probs)
		}
	}
	// Decode sequence to text
	words := make([]string, 0)
	for _, token := range sequence {
		if token != vocab["[PAD]"] && token != vocab["[MASK]"] {
			words = append(words, reverseVocab[token])
		}
	}
	return strings.Join(words, " ")
}

// argMax returns the index of the maximum value in a slice
func argMax(slice []float64) int {
	maxIdx := 0
	for i := 1; i < len(slice); i++ {
		if slice[i] > slice[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}
