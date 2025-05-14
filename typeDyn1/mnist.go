package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"os"
	"paragon"
	"path/filepath"
)

// ------------------- Util Functions -------------------

func ensureMNISTDownloads(targetDir string) error {
	if err := os.MkdirAll(targetDir, os.ModePerm); err != nil {
		return err
	}
	files := []struct {
		compressed   string
		uncompressed string
	}{
		{"train-images-idx3-ubyte.gz", "train-images-idx3-ubyte"},
		{"train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte"},
		{"t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte"},
		{"t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte"},
	}
	for _, f := range files {
		cPath := filepath.Join(targetDir, f.compressed)
		uPath := filepath.Join(targetDir, f.uncompressed)
		if _, err := os.Stat(uPath); os.IsNotExist(err) {
			if _, err := os.Stat(cPath); os.IsNotExist(err) {
				fmt.Printf("Downloading %s...\n", f.compressed)
				if err := downloadFile(baseURL+f.compressed, cPath); err != nil {
					return err
				}
			}
			fmt.Printf("Unzipping %s...\n", f.compressed)
			if err := unzipFile(cPath, uPath); err != nil {
				return err
			}
		}
	}
	return nil
}

func downloadFile(url, path string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	out, err := os.Create(path)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, resp.Body)
	return err
}

func unzipFile(src, dest string) error {
	fSrc, err := os.Open(src)
	if err != nil {
		return err
	}
	defer fSrc.Close()
	gzReader, err := gzip.NewReader(fSrc)
	if err != nil {
		return err
	}
	defer gzReader.Close()
	fDest, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer fDest.Close()
	_, err = io.Copy(fDest, gzReader)
	return err
}

func loadMNISTData(dir string, training bool) ([][][]float64, [][][]float64, error) {
	prefix := "train"
	if !training {
		prefix = "t10k"
	}
	imgPath := filepath.Join(dir, prefix+"-images-idx3-ubyte")
	lblPath := filepath.Join(dir, prefix+"-labels-idx1-ubyte")

	imgFile, err := os.Open(imgPath)
	if err != nil {
		return nil, nil, err
	}
	defer imgFile.Close()

	var header [16]byte
	if _, err := imgFile.Read(header[:]); err != nil {
		return nil, nil, err
	}
	num := int(binary.BigEndian.Uint32(header[4:8]))
	rows := int(binary.BigEndian.Uint32(header[8:12]))
	cols := int(binary.BigEndian.Uint32(header[12:16]))

	images := make([][][]float64, num)
	buf := make([]byte, rows*cols)
	for i := 0; i < num; i++ {
		if _, err := imgFile.Read(buf); err != nil {
			return nil, nil, err
		}
		img := make([][]float64, rows)
		for r := 0; r < rows; r++ {
			img[r] = make([]float64, cols)
			for c := 0; c < cols; c++ {
				img[r][c] = float64(buf[r*cols+c]) / 255.0
			}
		}
		images[i] = img
	}

	lblFile, err := os.Open(lblPath)
	if err != nil {
		return nil, nil, err
	}
	defer lblFile.Close()

	var lblHeader [8]byte
	if _, err := lblFile.Read(lblHeader[:]); err != nil {
		return nil, nil, err
	}
	labels := make([][][]float64, num)
	for i := 0; i < num; i++ {
		var b [1]byte
		if _, err := lblFile.Read(b[:]); err != nil {
			return nil, nil, err
		}
		labels[i] = labelToTarget(int(b[0]))
	}

	return images, labels, nil
}

func labelToTarget(label int) [][]float64 {
	target := make([][]float64, 1)
	target[0] = make([]float64, 10)
	target[0][label] = 1.0
	return target
}

func extractOutput(nn *paragon.Network[float32]) []float64 {
	outWidth := nn.Layers[nn.OutputLayer].Width
	out := make([]float64, outWidth)
	for x := 0; x < outWidth; x++ {
		out[x] = float64(nn.Layers[nn.OutputLayer].Neurons[0][x].Value)
	}
	return out
}
