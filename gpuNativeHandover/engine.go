package main

/*
#include <dlfcn.h>
#include <stdio.h>

// Define WebGPU types as opaque pointers (simplified for dynamic loading)
typedef void* WGPUInstance;
typedef void* WGPUInstanceDescriptor;

// Define the function pointer type for wgpuCreateInstance
typedef WGPUInstance (*WGPUProcCreateInstance)(const WGPUInstanceDescriptor*);

// Helper C function to call the function pointer
WGPUInstance callWGPUCreateInstance(void* fp, const WGPUInstanceDescriptor* desc) {
    WGPUProcCreateInstance func = (WGPUProcCreateInstance)fp;
    return func(desc);
}
*/
import "C"
import (
	"fmt"
)

func main() {
	// Load the shared library
	lib := C.dlopen(C.CString("./libwebgpu_dawn.so"), C.RTLD_LAZY)
	if lib == nil {
		panic("Failed to load libwebgpu_dawn.so")
	}
	defer C.dlclose(lib)

	// Retrieve the wgpuCreateInstance symbol
	proc := C.dlsym(lib, C.CString("wgpuCreateInstance"))
	if proc == nil {
		panic("Failed to find wgpuCreateInstance symbol")
	}
	fmt.Printf("Function pointer: %p\n", proc)

	// Call the function via the C helper, passing nil for the descriptor
	instance := C.callWGPUCreateInstance(proc, nil)
	if instance == nil {
		panic("Failed to create WebGPU instance")
	}
	fmt.Printf("Instance: %p\n", instance)
}
