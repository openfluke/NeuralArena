#include <dlfcn.h>
#include <stdio.h>

typedef void* WGPUInstance;
typedef WGPUInstance (*WGPUProcCreateInstance)(void*);

int main() {
    void* lib = dlopen("./libwebgpu_dawn.so", RTLD_LAZY);
    if (!lib) {
        printf("Failed to load library: %s\n", dlerror());
        return 1;
    }
    WGPUProcCreateInstance wgpuCreateInstance = (WGPUProcCreateInstance)dlsym(lib, "wgpuCreateInstance");
    if (!wgpuCreateInstance) {
        printf("Failed to find symbol: %s\n", dlerror());
        return 1;
    }
    printf("Function pointer: %p\n", wgpuCreateInstance);
    WGPUInstance instance = wgpuCreateInstance(NULL);
    printf("Instance: %p\n", instance);
    dlclose(lib);
    return 0;
}