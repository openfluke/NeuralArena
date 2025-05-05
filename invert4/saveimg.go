package main

import (
	"image"
	"image/color"
	"image/png"
	"os"
)

// SaveFloatImage saves a 2D float64 slice as a grayscale PNG image.
// It assumes values may be out of range, so it rescales them to [0, 255].
func SaveFloatImage(data [][]float64, filename string) error {
	height := len(data)
	width := len(data[0])

	// First, find min and max
	min, max := data[0][0], data[0][0]
	for _, row := range data {
		for _, val := range row {
			if val < min {
				min = val
			}
			if val > max {
				max = val
			}
		}
	}
	scale := 255.0 / (max - min)

	img := image.NewGray(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			val := (data[y][x] - min) * scale
			gray := uint8(val)
			img.SetGray(x, y, color.Gray{Y: gray})
		}
	}

	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}
