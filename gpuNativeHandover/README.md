# Native WebGPU/Dawn Integration

Demonstrates native GPU acceleration using Dawn (Google's WebGPU implementation) for neural network computations.

## What It Does

- **Native Dawn binding** — Uses CGO to call Dawn WebGPU functions directly
- **WGSL compute shaders** — Vector addition shader as proof of concept
- **CPU vs GPU benchmarking** — Compares performance on vector operations
- **Full pipeline** — Device creation, buffer management, shader compilation, dispatch

## Requirements

- **Dawn library** — `libwebgpu_dawn.so` must be present
- **CGO** — Requires C compiler for bindings

## Key Files

| File | Description |
|------|-------------|
| `engine.go` | Main Go code with CGO bindings |
| `libwebgpu_dawn.so` | Dawn shared library (~24MB) |
| `webgpu/` | WebGPU header files |

## WGSL Shader

```wgsl
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
```

## Functions

- `addVectorsCPU()` — CPU baseline
- `addVectorsGPU()` — GPU accelerated version
- `writeBuffer()` / `readBuffer()` — GPU memory operations

## Running

```bash
go run .
```

Requires Dawn library in library path.
