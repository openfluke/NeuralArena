package main

import (
	"fmt"
	"os"

	"paragon"
)

func main() {
    // Test GetAllGPUInfo
    gpus, err := paragon.GetAllGPUInfo()
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error getting GPU info: %v\n", err)
        os.Exit(1)
    }

    // Print GPU information
    fmt.Printf("Found %d GPUs:\n", len(gpus))
    for _, gpu := range gpus {
        fmt.Printf("\nGPU %s:\n", gpu["index"])
        fmt.Printf("  Name: %s\n", gpu["name"])
        fmt.Printf("  Driver Description: %s\n", gpu["driverDescription"])
        fmt.Printf("  Adapter Type: %s\n", gpu["adapterType"])
        fmt.Printf("  Vendor ID: %s\n", gpu["vendorId"])
        fmt.Printf("  Vendor Name: %s\n", gpu["vendorName"])
        fmt.Printf("  Architecture: %s\n", gpu["architecture"])
        fmt.Printf("  Device ID: %s\n", gpu["deviceId"])
        fmt.Printf("  Backend Type: %s\n", gpu["backendType"])
        fmt.Printf("  Max Compute Invocations per Workgroup: %s\n", gpu["maxComputeInvocations"])
        fmt.Printf("  Max Compute Workgroup Size X: %s\n", gpu["maxComputeWorkgroupSizeX"])
        fmt.Printf("  Max Compute Workgroup Size Y: %s\n", gpu["maxComputeWorkgroupSizeY"])
        fmt.Printf("  Max Compute Workgroup Size Z: %s\n", gpu["maxComputeWorkgroupSizeZ"])
        fmt.Printf("  Max Buffer Size: %s MB\n", gpu["maxBufferSizeMB"])
        fmt.Printf("  Max Storage Buffer Size: %s MB\n", gpu["maxStorageBufferSizeMB"])
        fmt.Printf("  Max Uniform Buffer Size: %s MB\n", gpu["maxUniformBufferSizeMB"])
        fmt.Printf("  Max Compute Workgroup Storage Size: %s KB\n", gpu["maxComputeWorkgroupStorageKB"])
    }

    // Example: Use GPU info for MNIST training
    // for _, gpu := range gpus {
    //     if gpu["adapterType"] == "discrete-gpu" {
    //         fmt.Printf("Assigning conv layers to %s\n", gpu["driverDescription"])
    //         // Parse maxBufferSizeMB for buffer allocation
    //         // Parse maxComputeInvocations for WGSL shader workgroup size
    //     }
    // }
}