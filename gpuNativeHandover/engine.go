package main

/*
#cgo CFLAGS: -I./webgpu
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "webgpu/webgpu.h"

// Define missing WebGPU types to ensure compilation
typedef uint32_t WGPUBufferUsageFlags;
typedef uint32_t WGPUMapMode;

typedef enum {
    WGPUSType_Invalid = 0x00000000,
    WGPUSType_ShaderModuleWGSLDescriptor = 0x00000005,
} WGPUSType;

typedef struct {
    const WGPUChainedStruct* nextInChain;
    WGPUShaderModule module;
    const char* entryPoint;
    uint32_t constantCount;
    const WGPUConstantEntry* constants;
} WGPUProgrammableStageDescriptor;

// Helper C functions to call function pointers
WGPUInstance callWGPUCreateInstance(void* fp, const WGPUInstanceDescriptor* desc) {
    WGPUProcCreateInstance func = (WGPUProcCreateInstance)fp;
    return func(desc);
}

void callWGPUInstanceRequestAdapter(void* fp, WGPUInstance instance, const WGPURequestAdapterOptions* options, WGPURequestAdapterCallbackInfo callbackInfo) {
    WGPUProcInstanceRequestAdapter func = (WGPUProcInstanceRequestAdapter)fp;
    func(instance, options, callbackInfo);
}

void callWGPUAdapterRequestDevice(void* fp, WGPUAdapter adapter, const WGPUDeviceDescriptor* descriptor, WGPURequestDeviceCallbackInfo callbackInfo) {
    WGPUProcAdapterRequestDevice func = (WGPUProcAdapterRequestDevice)fp;
    func(adapter, descriptor, callbackInfo);
}

WGPUQueue callWGPUDeviceGetQueue(void* fp, WGPUDevice device) {
    WGPUProcDeviceGetQueue func = (WGPUProcDeviceGetQueue)fp;
    return func(device);
}

WGPUBuffer callWGPUDeviceCreateBuffer(void* fp, WGPUDevice device, const WGPUBufferDescriptor* desc) {
    WGPUProcDeviceCreateBuffer func = (WGPUProcDeviceCreateBuffer)fp;
    return func(device, desc);
}

WGPUCommandEncoder callWGPUDeviceCreateCommandEncoder(void* fp, WGPUDevice device, const WGPUCommandEncoderDescriptor* desc) {
    WGPUProcDeviceCreateCommandEncoder func = (WGPUProcDeviceCreateCommandEncoder)fp;
    return func(device, desc);
}

void callWGPUCommandEncoderCopyBufferToBuffer(void* fp, WGPUCommandEncoder encoder, WGPUBuffer source, uint64_t sourceOffset, WGPUBuffer destination, uint64_t destinationOffset, uint64_t size) {
    WGPUProcCommandEncoderCopyBufferToBuffer func = (WGPUProcCommandEncoderCopyBufferToBuffer)fp;
    func(encoder, source, sourceOffset, destination, destinationOffset, size);
}

WGPUCommandBuffer callWGPUCommandEncoderFinish(void* fp, WGPUCommandEncoder encoder, const WGPUCommandBufferDescriptor* desc) {
    WGPUProcCommandEncoderFinish func = (WGPUProcCommandEncoderFinish)fp;
    return func(encoder, desc);
}

void callWGPUQueueSubmit(void* fp, WGPUQueue queue, uint32_t commandCount, WGPUCommandBuffer* commands) {
    WGPUProcQueueSubmit func = (WGPUProcQueueSubmit)fp;
    func(queue, commandCount, commands);
}

WGPUShaderModule callWGPUDeviceCreateShaderModule(void* fp, WGPUDevice device, const WGPUShaderModuleDescriptor* desc) {
    WGPUProcDeviceCreateShaderModule func = (WGPUProcDeviceCreateShaderModule)fp;
    return func(device, desc);
}

WGPUComputePipeline callWGPUDeviceCreateComputePipeline(void* fp, WGPUDevice device, const WGPUComputePipelineDescriptor* desc) {
    WGPUProcDeviceCreateComputePipeline func = (WGPUProcDeviceCreateComputePipeline)fp;
    return func(device, desc);
}

WGPUBindGroupLayout callWGPUComputePipelineGetBindGroupLayout(void* fp, WGPUComputePipeline pipeline, uint32_t groupIndex) {
    WGPUProcComputePipelineGetBindGroupLayout func = (WGPUProcComputePipelineGetBindGroupLayout)fp;
    return func(pipeline, groupIndex);
}

WGPUBindGroup callWGPUDeviceCreateBindGroup(void* fp, WGPUDevice device, const WGPUBindGroupDescriptor* desc) {
    WGPUProcDeviceCreateBindGroup func = (WGPUProcDeviceCreateBindGroup)fp;
    return func(device, desc);
}

WGPUComputePassEncoder callWGPUCommandEncoderBeginComputePass(void* fp, WGPUCommandEncoder encoder, const WGPUComputePassDescriptor* desc) {
    WGPUProcCommandEncoderBeginComputePass func = (WGPUProcCommandEncoderBeginComputePass)fp;
    return func(encoder, desc);
}

void callWGPUComputePassEncoderSetPipeline(void* fp, WGPUComputePassEncoder pass, WGPUComputePipeline pipeline) {
    WGPUProcComputePassEncoderSetPipeline func = (WGPUProcComputePassEncoderSetPipeline)fp;
    func(pass, pipeline);
}

void callWGPUComputePassEncoderSetBindGroup(void* fp, WGPUComputePassEncoder pass, uint32_t groupIndex, WGPUBindGroup group, uint32_t dynamicOffsetCount, const uint32_t* dynamicOffsets) {
    WGPUProcComputePassEncoderSetBindGroup func = (WGPUProcComputePassEncoderSetBindGroup)fp;
    func(pass, groupIndex, group, dynamicOffsetCount, dynamicOffsets);
}

void callWGPUComputePassEncoderDispatchWorkgroups(void* fp, WGPUComputePassEncoder pass, uint32_t workgroupCountX, uint32_t workgroupCountY, uint32_t workgroupCountZ) {
    WGPUProcComputePassEncoderDispatchWorkgroups func = (WGPUProcComputePassEncoderDispatchWorkgroups)fp;
    func(pass, workgroupCountX, workgroupCountY, workgroupCountZ);
}

void callWGPUComputePassEncoderEnd(void* fp, WGPUComputePassEncoder pass) {
    WGPUProcComputePassEncoderEnd func = (WGPUProcComputePassEncoderEnd)fp;
    func(pass);
}

void callWGPUBufferMapAsync(void* fp, WGPUBuffer buffer, WGPUMapMode mode, size_t offset, size_t size, WGPUBufferMapCallbackInfo callbackInfo) {
    WGPUProcBufferMapAsync func = (WGPUProcBufferMapAsync)fp;
    func(buffer, mode, offset, size, callbackInfo);
}

void* callWGPUBufferGetMappedRange(void* fp, WGPUBuffer buffer, size_t offset, size_t size) {
    WGPUProcBufferGetMappedRange func = (WGPUProcBufferGetMappedRange)fp;
    return func(buffer, offset, size);
}

void callWGPUBufferUnmap(void* fp, WGPUBuffer buffer) {
    WGPUProcBufferUnmap func = (WGPUProcBufferUnmap)fp;
    func(buffer);
}

void callWGPUInstanceRelease(void* fp, WGPUInstance instance) {
    WGPUProcInstanceRelease func = (WGPUProcInstanceRelease)fp;
    func(instance);
}

void callWGPUAdapterRelease(void* fp, WGPUAdapter adapter) {
    WGPUProcAdapterRelease func = (WGPUProcAdapterRelease)fp;
    func(adapter);
}

void callWGPUDeviceRelease(void* fp, WGPUDevice device) {
    WGPUProcDeviceRelease func = (WGPUProcDeviceRelease)fp;
    func(device);
}

void callWGPUQueueRelease(void* fp, WGPUQueue queue) {
    WGPUProcQueueRelease func = (WGPUProcQueueRelease)fp;
    func(queue);
}

void callWGPUBufferRelease(void* fp, WGPUBuffer buffer) {
    WGPUProcBufferRelease func = (WGPUProcBufferRelease)fp;
    func(buffer);
}

void callWGPUCommandEncoderRelease(void* fp, WGPUCommandEncoder encoder) {
    WGPUProcCommandEncoderRelease func = (WGPUProcCommandEncoderRelease)fp;
    func(encoder);
}

void callWGPUComputePassEncoderRelease(void* fp, WGPUComputePassEncoder pass) {
    WGPUProcComputePassEncoderRelease func = (WGPUProcComputePassEncoderRelease)fp;
    func(pass);
}

void callWGPUComputePipelineRelease(void* fp, WGPUComputePipeline pipeline) {
    WGPUProcComputePipelineRelease func = (WGPUProcComputePipelineRelease)fp;
    func(pipeline);
}

void callWGPUBindGroupLayoutRelease(void* fp, WGPUBindGroupLayout layout) {
    WGPUProcBindGroupLayoutRelease func = (WGPUProcBindGroupLayoutRelease)fp;
    func(layout);
}

void callWGPUBindGroupRelease(void* fp, WGPUBindGroup group) {
    WGPUProcBindGroupRelease func = (WGPUProcBindGroupRelease)fp;
    func(group);
}

void callWGPUShaderModuleRelease(void* fp, WGPUShaderModule module) {
    WGPUProcShaderModuleRelease func = (WGPUProcShaderModuleRelease)fp;
    func(module);
}

// Callback for adapter request
void adapterCallback(WGPURequestAdapterStatus status, WGPUAdapter adapter, const char* message, void* userdata) {
    if (status == WGPURequestAdapterStatus_Success) {
        *(WGPUAdapter*)userdata = adapter;
    } else {
        fprintf(stderr, "Adapter request failed: %s\n", message);
    }
}

// Callback for device request
void deviceCallback(WGPURequestDeviceStatus status, WGPUDevice device, const char* message, void* userdata) {
    if (status == WGPURequestDeviceStatus_Success) {
        *(WGPUDevice*)userdata = device;
    } else {
        fprintf(stderr, "Device request failed: %s\n", message);
    }
}

// Callback for buffer mapping
void bufferMapCallback(WGPUMapAsyncStatus status, void* userdata) {
    if (status != WGPUMapAsyncStatus_Success) {
        fprintf(stderr, "Buffer map failed: %d\n", status);
    }
}
*/
import "C"
import (
	"fmt"
	"math"
	"math/rand"
	"time"
	"unsafe"
)

// WGSL shader for vector addition
const wgslCode = `
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> c : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&a)) {
        c[i] = a[i] + b[i];
    }
}
`

func main() {
	// Vector size for computation
	const size = 1000000

	// Generate random input vectors
	rand.Seed(time.Now().UnixNano())
	a := generateVector(size)
	b := generateVector(size)

	// CPU computation
	startCPU := time.Now()
	cCPU := addVectorsCPU(a, b)
	durationCPU := time.Since(startCPU)

	// GPU computation
	startGPU := time.Now()
	cGPU, err := addVectorsGPU(a, b)
	if err != nil {
		fmt.Println("GPU computation failed:", err)
		return
	}
	durationGPU := time.Since(startGPU)

	// Verify results
	if !vectorsEqual(cCPU, cGPU, 1e-5) {
		fmt.Println("Error: CPU and GPU results do not match!")
		return
	}

	// Print timings
	fmt.Printf("CPU time: %v\n", durationCPU)
	fmt.Printf("GPU time: %v\n", durationGPU)
}

// addVectorsCPU performs vector addition on the CPU
func addVectorsCPU(a, b []float32) []float32 {
	c := make([]float32, len(a))
	for i := range a {
		c[i] = a[i] + b[i]
	}
	return c
}

// addVectorsGPU performs vector addition on the GPU using Dawn
func addVectorsGPU(a, b []float32) ([]float32, error) {
	// Load the shared library
	lib := C.dlopen(C.CString("./libwebgpu_dawn.so"), C.RTLD_LAZY)
	if lib == nil {
		return nil, fmt.Errorf("failed to load libwebgpu_dawn.so")
	}
	defer C.dlclose(lib)

	// Map function symbols
	functions := map[string]unsafe.Pointer{
		"wgpuCreateInstance":                       C.dlsym(lib, C.CString("wgpuCreateInstance")),
		"wgpuInstanceRequestAdapter":               C.dlsym(lib, C.CString("wgpuInstanceRequestAdapter")),
		"wgpuAdapterRequestDevice":                 C.dlsym(lib, C.CString("wgpuAdapterRequestDevice")),
		"wgpuDeviceGetQueue":                       C.dlsym(lib, C.CString("wgpuDeviceGetQueue")),
		"wgpuDeviceCreateBuffer":                   C.dlsym(lib, C.CString("wgpuDeviceCreateBuffer")),
		"wgpuDeviceCreateCommandEncoder":           C.dlsym(lib, C.CString("wgpuDeviceCreateCommandEncoder")),
		"wgpuCommandEncoderCopyBufferToBuffer":     C.dlsym(lib, C.CString("wgpuCommandEncoderCopyBufferToBuffer")),
		"wgpuCommandEncoderFinish":                 C.dlsym(lib, C.CString("wgpuCommandEncoderFinish")),
		"wgpuQueueSubmit":                          C.dlsym(lib, C.CString("wgpuQueueSubmit")),
		"wgpuDeviceCreateShaderModule":             C.dlsym(lib, C.CString("wgpuDeviceCreateShaderModule")),
		"wgpuDeviceCreateComputePipeline":          C.dlsym(lib, C.CString("wgpuDeviceCreateComputePipeline")),
		"wgpuComputePipelineGetBindGroupLayout":    C.dlsym(lib, C.CString("wgpuComputePipelineGetBindGroupLayout")),
		"wgpuDeviceCreateBindGroup":                C.dlsym(lib, C.CString("wgpuDeviceCreateBindGroup")),
		"wgpuCommandEncoderBeginComputePass":       C.dlsym(lib, C.CString("wgpuCommandEncoderBeginComputePass")),
		"wgpuComputePassEncoderSetPipeline":        C.dlsym(lib, C.CString("wgpuComputePassEncoderSetPipeline")),
		"wgpuComputePassEncoderSetBindGroup":       C.dlsym(lib, C.CString("wgpuComputePassEncoderSetBindGroup")),
		"wgpuComputePassEncoderDispatchWorkgroups": C.dlsym(lib, C.CString("wgpuComputePassEncoderDispatchWorkgroups")),
		"wgpuComputePassEncoderEnd":                C.dlsym(lib, C.CString("wgpuComputePassEncoderEnd")),
		"wgpuBufferMapAsync":                       C.dlsym(lib, C.CString("wgpuBufferMapAsync")),
		"wgpuBufferGetMappedRange":                 C.dlsym(lib, C.CString("wgpuBufferGetMappedRange")),
		"wgpuBufferUnmap":                          C.dlsym(lib, C.CString("wgpuBufferUnmap")),
		"wgpuInstanceRelease":                      C.dlsym(lib, C.CString("wgpuInstanceRelease")),
		"wgpuAdapterRelease":                       C.dlsym(lib, C.CString("wgpuAdapterRelease")),
		"wgpuDeviceRelease":                        C.dlsym(lib, C.CString("wgpuDeviceRelease")),
		"wgpuQueueRelease":                         C.dlsym(lib, C.CString("wgpuQueueRelease")),
		"wgpuBufferRelease":                        C.dlsym(lib, C.CString("wgpuBufferRelease")),
		"wgpuCommandEncoderRelease":                C.dlsym(lib, C.CString("wgpuCommandEncoderRelease")),
		"wgpuComputePassEncoderRelease":            C.dlsym(lib, C.CString("wgpuComputePassEncoderRelease")),
		"wgpuComputePipelineRelease":               C.dlsym(lib, C.CString("wgpuComputePipelineRelease")),
		"wgpuBindGroupLayoutRelease":               C.dlsym(lib, C.CString("wgpuBindGroupLayoutRelease")),
		"wgpuBindGroupRelease":                     C.dlsym(lib, C.CString("wgpuBindGroupRelease")),
		"wgpuShaderModuleRelease":                  C.dlsym(lib, C.CString("wgpuShaderModuleRelease")),
	}

	for name, ptr := range functions {
		if ptr == nil {
			return nil, fmt.Errorf("failed to find %s symbol", name)
		}
	}

	// Create instance
	var instanceDescriptor C.WGPUInstanceDescriptor
	instanceDescriptor.nextInChain = nil
	instance := C.callWGPUCreateInstance(functions["wgpuCreateInstance"], &instanceDescriptor)
	if instance == nil {
		return nil, fmt.Errorf("failed to create WebGPU instance")
	}
	defer C.callWGPUInstanceRelease(functions["wgpuInstanceRelease"], instance)

	// Request adapter (asynchronous with callback)
	var adapter C.WGPUAdapter
	var adapterCallbackInfo C.WGPURequestAdapterCallbackInfo
	adapterCallbackInfo.mode = C.WGPUCallbackMode_WaitAnyOnly
	adapterCallbackInfo.callback = C.WGPURequestAdapterCallback(C.adapterCallback)
	adapterCallbackInfo.userdata = unsafe.Pointer(&adapter)
	C.callWGPUInstanceRequestAdapter(functions["wgpuInstanceRequestAdapter"], instance, nil, adapterCallbackInfo)
	if adapter == nil {
		return nil, fmt.Errorf("failed to request adapter")
	}
	defer C.callWGPUAdapterRelease(functions["wgpuAdapterRelease"], adapter)

	// Create device (asynchronous with callback)
	var device C.WGPUDevice
	var deviceCallbackInfo C.WGPURequestDeviceCallbackInfo
	deviceCallbackInfo.mode = C.WGPUCallbackMode_WaitAnyOnly
	deviceCallbackInfo.callback = C.WGPURequestDeviceCallback(C.deviceCallback)
	deviceCallbackInfo.userdata = unsafe.Pointer(&device)
	C.callWGPUAdapterRequestDevice(functions["wgpuAdapterRequestDevice"], adapter, nil, deviceCallbackInfo)
	if device == nil {
		return nil, fmt.Errorf("failed to create device")
	}
	defer C.callWGPUDeviceRelease(functions["wgpuDeviceRelease"], device)

	// Get queue
	queue := C.callWGPUDeviceGetQueue(functions["wgpuDeviceGetQueue"], device)
	if queue == nil {
		return nil, fmt.Errorf("failed to get queue")
	}
	defer C.callWGPUQueueRelease(functions["wgpuQueueRelease"], queue)

	// Buffer size in bytes
	dataSize := C.uint64_t(len(a) * 4)

	// Create staging buffers for input data
	var stagingADescriptor C.WGPUBufferDescriptor
	stagingADescriptor.size = dataSize
	stagingADescriptor.usage = C.WGPUBufferUsageFlags(WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc)
	stagingA := C.callWGPUDeviceCreateBuffer(functions["wgpuDeviceCreateBuffer"], device, &stagingADescriptor)
	if stagingA == nil {
		return nil, fmt.Errorf("failed to create staging buffer A")
	}
	defer C.callWGPUBufferRelease(functions["wgpuBufferRelease"], stagingA)

	var stagingBDescriptor C.WGPUBufferDescriptor
	stagingBDescriptor.size = dataSize
	stagingBDescriptor.usage = C.WGPUBufferUsageFlags(WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc)
	stagingB := C.callWGPUDeviceCreateBuffer(functions["wgpuDeviceCreateBuffer"], device, &stagingBDescriptor)
	if stagingB == nil {
		return nil, fmt.Errorf("failed to create staging buffer B")
	}
	defer C.callWGPUBufferRelease(functions["wgpuBufferRelease"], stagingB)

	// Map and write input data
	if err := writeBuffer(functions, stagingA, a); err != nil {
		return nil, err
	}
	if err := writeBuffer(functions, stagingB, b); err != nil {
		return nil, err
	}

	// Create device buffers
	var bufferADescriptor C.WGPUBufferDescriptor
	bufferADescriptor.size = dataSize
	bufferADescriptor.usage = C.WGPUBufferUsageFlags(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst)
	bufferA := C.callWGPUDeviceCreateBuffer(functions["wgpuDeviceCreateBuffer"], device, &bufferADescriptor)
	if bufferA == nil {
		return nil, fmt.Errorf("failed to create buffer A")
	}
	defer C.callWGPUBufferRelease(functions["wgpuBufferRelease"], bufferA)

	var bufferBDescriptor C.WGPUBufferDescriptor
	bufferBDescriptor.size = dataSize
	bufferBDescriptor.usage = C.WGPUBufferUsageFlags(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst)
	bufferB := C.callWGPUDeviceCreateBuffer(functions["wgpuDeviceCreateBuffer"], device, &bufferBDescriptor)
	if bufferB == nil {
		return nil, fmt.Errorf("failed to create buffer B")
	}
	defer C.callWGPUBufferRelease(functions["wgpuBufferRelease"], bufferB)

	var bufferCDescriptor C.WGPUBufferDescriptor
	bufferCDescriptor.size = dataSize
	bufferCDescriptor.usage = C.WGPUBufferUsageFlags(WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc)
	bufferC := C.callWGPUDeviceCreateBuffer(functions["wgpuDeviceCreateBuffer"], device, &bufferCDescriptor)
	if bufferC == nil {
		return nil, fmt.Errorf("failed to create buffer C")
	}
	defer C.callWGPUBufferRelease(functions["wgpuBufferRelease"], bufferC)

	// Copy from staging to device buffers
	encoder := C.callWGPUDeviceCreateCommandEncoder(functions["wgpuDeviceCreateCommandEncoder"], device, nil)
	if encoder == nil {
		return nil, fmt.Errorf("failed to create command encoder")
	}
	defer C.callWGPUCommandEncoderRelease(functions["wgpuCommandEncoderRelease"], encoder)

	C.callWGPUCommandEncoderCopyBufferToBuffer(functions["wgpuCommandEncoderCopyBufferToBuffer"], encoder, stagingA, 0, bufferA, 0, dataSize)
	C.callWGPUCommandEncoderCopyBufferToBuffer(functions["wgpuCommandEncoderCopyBufferToBuffer"], encoder, stagingB, 0, bufferB, 0, dataSize)
	cmdBuffer := C.callWGPUCommandEncoderFinish(functions["wgpuCommandEncoderFinish"], encoder, nil)
	if cmdBuffer == nil {
		return nil, fmt.Errorf("failed to finish command encoder")
	}
	defer C.free(unsafe.Pointer(cmdBuffer))

	C.callWGPUQueueSubmit(functions["wgpuQueueSubmit"], queue, 1, &cmdBuffer)

	// Create shader module
	shaderCode := C.CString(wgslCode)
	defer C.free(unsafe.Pointer(shaderCode))
	var wgslDescriptor C.WGPUShaderModuleWGSLDescriptor
	wgslDescriptor.chain.sType = C.WGPUSType_ShaderModuleWGSLDescriptor
	wgslDescriptor.code = shaderCode
	var shaderModuleDescriptor C.WGPUShaderModuleDescriptor
	shaderModuleDescriptor.nextInChain = (*C.WGPUChainedStruct)(unsafe.Pointer(&wgslDescriptor))
	shaderModule := C.callWGPUDeviceCreateShaderModule(functions["wgpuDeviceCreateShaderModule"], device, &shaderModuleDescriptor)
	if shaderModule == nil {
		return nil, fmt.Errorf("failed to create shader module")
	}
	defer C.callWGPUShaderModuleRelease(functions["wgpuShaderModuleRelease"], shaderModule)

	// Create compute pipeline
	var programmableStage C.WGPUProgrammableStageDescriptor
	programmableStage.module = shaderModule
	programmableStage.entryPoint = C.CString("main")
	defer C.free(unsafe.Pointer(programmableStage.entryPoint))
	var pipelineDescriptor C.WGPUComputePipelineDescriptor
	pipelineDescriptor.compute = programmableStage
	pipeline := C.callWGPUDeviceCreateComputePipeline(functions["wgpuDeviceCreateComputePipeline"], device, &pipelineDescriptor)
	if pipeline == nil {
		return nil, fmt.Errorf("failed to create compute pipeline")
	}
	defer C.callWGPUComputePipelineRelease(functions["wgpuComputePipelineRelease"], pipeline)

	// Create bind group layout and bind group
	bindGroupLayout := C.callWGPUComputePipelineGetBindGroupLayout(functions["wgpuComputePipelineGetBindGroupLayout"], pipeline, 0)
	if bindGroupLayout == nil {
		return nil, fmt.Errorf("failed to create bind group layout")
	}
	defer C.callWGPUBindGroupLayoutRelease(functions["wgpuBindGroupLayoutRelease"], bindGroupLayout)

	bindGroupEntries := []C.WGPUBindGroupEntry{
		{binding: 0, buffer: bufferA, offset: 0, size: dataSize},
		{binding: 1, buffer: bufferB, offset: 0, size: dataSize},
		{binding: 2, buffer: bufferC, offset: 0, size: dataSize},
	}
	var bindGroupDescriptor C.WGPUBindGroupDescriptor
	bindGroupDescriptor.layout = bindGroupLayout
	bindGroupDescriptor.entries = &bindGroupEntries[0]
	bindGroupDescriptor.entryCount = C.uint(len(bindGroupEntries))
	bindGroup := C.callWGPUDeviceCreateBindGroup(functions["wgpuDeviceCreateBindGroup"], device, &bindGroupDescriptor)
	if bindGroup == nil {
		return nil, fmt.Errorf("failed to create bind group")
	}
	defer C.callWGPUBindGroupRelease(functions["wgpuBindGroupRelease"], bindGroup)

	// Execute compute shader
	encoder = C.callWGPUDeviceCreateCommandEncoder(functions["wgpuDeviceCreateCommandEncoder"], device, nil)
	if encoder == nil {
		return nil, fmt.Errorf("failed to create command encoder for compute")
	}
	defer C.callWGPUCommandEncoderRelease(functions["wgpuCommandEncoderRelease"], encoder)

	pass := C.callWGPUCommandEncoderBeginComputePass(functions["wgpuCommandEncoderBeginComputePass"], encoder, nil)
	if pass == nil {
		return nil, fmt.Errorf("failed to begin compute pass")
	}
	defer C.callWGPUComputePassEncoderRelease(functions["wgpuComputePassEncoderRelease"], pass)

	C.callWGPUComputePassEncoderSetPipeline(functions["wgpuComputePassEncoderSetPipeline"], pass, pipeline)
	C.callWGPUComputePassEncoderSetBindGroup(functions["wgpuComputePassEncoderSetBindGroup"], pass, 0, bindGroup, 0, nil)
	workgroupCount := uint32(math.Ceil(float64(len(a)) / 64.0))
	C.callWGPUComputePassEncoderDispatchWorkgroups(functions["wgpuComputePassEncoderDispatchWorkgroups"], pass, C.uint(workgroupCount), 1, 1)
	C.callWGPUComputePassEncoderEnd(functions["wgpuComputePassEncoderEnd"], pass)
	cmdBuffer = C.callWGPUCommandEncoderFinish(functions["wgpuCommandEncoderFinish"], encoder, nil)
	if cmdBuffer == nil {
		return nil, fmt.Errorf("failed to finish compute command encoder")
	}
	defer C.free(unsafe.Pointer(cmdBuffer))

	C.callWGPUQueueSubmit(functions["wgpuQueueSubmit"], queue, 1, &cmdBuffer)

	// Read result
	var stagingCDescriptor C.WGPUBufferDescriptor
	stagingCDescriptor.size = dataSize
	stagingCDescriptor.usage = C.WGPUBufferUsageFlags(WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst)
	stagingC := C.callWGPUDeviceCreateBuffer(functions["wgpuDeviceCreateBuffer"], device, &stagingCDescriptor)
	if stagingC == nil {
		return nil, fmt.Errorf("failed to create staging buffer C")
	}
	defer C.callWGPUBufferRelease(functions["wgpuBufferRelease"], stagingC)

	encoder = C.callWGPUDeviceCreateCommandEncoder(functions["wgpuDeviceCreateCommandEncoder"], device, nil)
	if encoder == nil {
		return nil, fmt.Errorf("failed to create command encoder for result")
	}
	defer C.callWGPUCommandEncoderRelease(functions["wgpuCommandEncoderRelease"], encoder)

	C.callWGPUCommandEncoderCopyBufferToBuffer(functions["wgpuCommandEncoderCopyBufferToBuffer"], encoder, bufferC, 0, stagingC, 0, dataSize)
	cmdBuffer = C.callWGPUCommandEncoderFinish(functions["wgpuCommandEncoderFinish"], encoder, nil)
	if cmdBuffer == nil {
		return nil, fmt.Errorf("failed to finish result command encoder")
	}
	defer C.free(unsafe.Pointer(cmdBuffer))

	C.callWGPUQueueSubmit(functions["wgpuQueueSubmit"], queue, 1, &cmdBuffer)

	return readBuffer(functions, stagingC, len(a))
}

// writeBuffer maps a buffer and writes data to it
func writeBuffer(functions map[string]unsafe.Pointer, buffer C.WGPUBuffer, data []float32) error {
	var callbackInfo C.WGPUBufferMapCallbackInfo
	callbackInfo.mode = C.WGPUCallbackMode_WaitAnyOnly
	callbackInfo.callback = C.WGPUBufferMapCallback(C.bufferMapCallback)
	C.callWGPUBufferMapAsync(functions["wgpuBufferMapAsync"], buffer, C.WGPUMapMode(WGPUMapMode_Write), 0, C.size_t(len(data)*4), callbackInfo)
	dataPtr := C.callWGPUBufferGetMappedRange(functions["wgpuBufferGetMappedRange"], buffer, 0, C.size_t(len(data)*4))
	if dataPtr == nil {
		return fmt.Errorf("failed to map buffer for writing")
	}
	C.memcpy(dataPtr, unsafe.Pointer(&data[0]), C.size_t(len(data)*4))
	C.callWGPUBufferUnmap(functions["wgpuBufferUnmap"], buffer)
	return nil
}

// readBuffer maps a buffer and reads data from it
func readBuffer(functions map[string]unsafe.Pointer, buffer C.WGPUBuffer, size int) ([]float32, error) {
	var callbackInfo C.WGPUBufferMapCallbackInfo
	callbackInfo.mode = C.WGPUCallbackMode_WaitAnyOnly
	callbackInfo.callback = C.WGPUBufferMapCallback(C.bufferMapCallback)
	C.callWGPUBufferMapAsync(functions["wgpuBufferMapAsync"], buffer, C.WGPUMapMode(WGPUMapMode_Read), 0, C.size_t(size*4), callbackInfo)
	dataPtr := C.callWGPUBufferGetMappedRange(functions["wgpuBufferGetMappedRange"], buffer, 0, C.size_t(size*4))
	if dataPtr == nil {
		return nil, fmt.Errorf("failed to map buffer for reading")
	}
	result := make([]float32, size)
	C.memcpy(unsafe.Pointer(&result[0]), dataPtr, C.size_t(size*4))
	C.callWGPUBufferUnmap(functions["wgpuBufferUnmap"], buffer)
	return result, nil
}

// generateVector creates a random vector
func generateVector(size int) []float32 {
	vec := make([]float32, size)
	for i := range vec {
		vec[i] = rand.Float32() * 100
	}
	return vec
}

// vectorsEqual checks if two vectors are equal within a tolerance
func vectorsEqual(a, b []float32, epsilon float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > float64(epsilon) {
			return false
		}
	}
	return true
}
