package main

import (
	"fmt"
	"paragon"
)

//"paragon"

func main() {
	fmt.Println("Building partitioning nas")

	cubes, _ := paragon.ReadCSV("cubes.csv")
	lstCubes, _ := paragon.Cleaner(cubes, []int{0}, []int{})
	lstConvCubes, lstLabelCubes, _ := paragon.Converter(lstCubes, []int{0})

	paragon.PrintTable(lstCubes)
	paragon.PrintTable(lstLabelCubes)
	paragon.PrintTable(lstConvCubes)

	links, _ := paragon.ReadCSV("links.csv")
	lstLinks, _ := paragon.Cleaner(links, []int{0, 1}, []int{3, 4, 5, 6, 7})
	lstConvLinks, lstLabelLinks, _ := paragon.Converter(lstLinks, []int{0, 1, 2})

	paragon.PrintTable(lstLinks)
	paragon.PrintTable(lstLabelLinks)
	paragon.PrintTable(lstConvLinks)
}
