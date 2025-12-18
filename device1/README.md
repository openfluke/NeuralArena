# GPU Device Information

Queries and displays detailed information about available GPU devices using WebGPU/paragon.

## What It Does

- **Enumerates GPUs** — Lists all available GPU devices
- **Displays specs** — Shows capabilities and limits for each GPU
- **Helps planning** — Useful for deciding how to distribute workloads

## Output Fields

| Field | Description |
|-------|-------------|
| Name | GPU device name |
| Driver Description | Driver/backend info |
| Adapter Type | Discrete, integrated, etc. |
| Vendor ID/Name | GPU manufacturer |
| Architecture | GPU architecture name |
| Backend Type | Vulkan, Metal, D3D12, etc. |
| Max Compute Invocations | Workgroup thread limits |
| Max Buffer Size | Maximum buffer allocation |
| Max Storage Buffer | Max SSBO size |

## Example Output

```
Found 1 GPUs:

GPU 0:
  Name: NVIDIA GeForce RTX 3080
  Driver Description: ...
  Max Buffer Size: 2048 MB
  Max Compute Invocations: 1024
```

## Running

```bash
go run .
```
