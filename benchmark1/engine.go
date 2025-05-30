package main

import (
	"fmt"
	"time"

	"paragon" // Replace with full path like "github.com/openfluke/paragon" if needed
)

func main() {
	// Duration for each benchmark
	benchmarkDuration := 2 * time.Second

	fmt.Println("🧪 Running all numerical type benchmarks with duration:", benchmarkDuration)
	paragon.RunAllBenchmarks(benchmarkDuration)

	fmt.Println("\n🔍 Reflecting over methods on paragon.Network[float64]...")

	// Instantiate a generic network to inspect methods
	net := &paragon.Network[float64]{} // or int8, float32, etc.

	// Print method metadata as JSON
	if jsonStr, err := net.GetphaseMethodsJSON(); err != nil {
		fmt.Println("Error retrieving method metadata:", err)
	} else {
		fmt.Println(jsonStr)
	}
}
