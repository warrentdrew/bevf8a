#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

// #include <device_functions.h>
__device__ __forceinline__ half atomicMax(half *addr, half val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    unsigned short int addr_as_short = __half_as_ushort(*addr);
    // while(val > __ushort_as_half(addr_as_short)) {
        while(__hgt(val, __ushort_as_half(addr_as_short))) {
        unsigned short int old = addr_as_short;
        if((addr_as_short = atomicCAS((unsigned short int*)addr, old, __half_as_ushort(val))) == old) {
            break;
        }
    }
    return __ushort_as_half(addr_as_short);
#else
    // Dummy implementation for unsupported architecture
    return  __float2half(0.f);
#endif /* defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) */
}

__device__ __forceinline__ float atomicMax(float *addr, float val) {
    int addr_as_int = __float_as_int(*addr);
    while(val > __int_as_float(addr_as_int)) {
        int old = addr_as_int;
        if((addr_as_int = atomicCAS((int*)addr, old, __float_as_int(val))) == old) {
            break;
        }
    }
    return __int_as_float(addr_as_int);
}