#ifndef CUDA_HIP_WRAPPER_H
#define CUDA_HIP_WRAPPER_H

#ifdef USE_HIP

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <hipfft.h>

// cuda.h adapters
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#define cudaGetDevice hipGetDevice
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemset hipMemset
#define cudaSetDevice hipSetDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

// cuComplex.h adapters
#define cuDoubleComplex hipDoubleComplex
#define cuFloatComplex hipFloatComplex

// cufft.h adapters
#define CUFFT_ALLOC_FAILED HIPFFT_ALLOC_FAILED
#define CUFFT_C2C HIPFFT_C2C
#define CUFFT_EXEC_FAILED HIPFFT_EXEC_FAILED
#define CUFFT_INCOMPLETE_PARAMETER_LIST HIPFFT_INCOMPLETE_PARAMETER_LIST
#define CUFFT_INTERNAL_ERROR HIPFFT_INTERNAL_ERROR
#define CUFFT_INVALID_DEVICE HIPFFT_INVALID_DEVICE
#define CUFFT_INVALID_PLAN HIPFFT_INVALID_PLAN
#define CUFFT_INVALID_SIZE HIPFFT_INVALID_SIZE
#define CUFFT_INVALID_TYPE HIPFFT_INVALID_TYPE
#define CUFFT_INVALID_TYPE HIPFFT_INVALID_TYPE
#define CUFFT_INVALID_VALUE HIPFFT_INVALID_VALUE
#define CUFFT_NOT_IMPLEMENTED HIPFFT_NOT_IMPLEMENTED
#define CUFFT_NOT_SUPPORTED HIPFFT_NOT_SUPPORTED
#define CUFFT_NO_WORKSPACE HIPFFT_NO_WORKSPACE
#define CUFFT_PARSE_ERROR HIPFFT_PARSE_ERROR
#define CUFFT_SETUP_FAILED HIPFFT_SETUP_FAILED
#define CUFFT_SUCCESS HIPFFT_SUCCESS
#define CUFFT_UNALIGNED_DATA HIPFFT_UNALIGNED_DATA
#define CUFFT_Z2Z HIPFFT_Z2Z
#define cufftDestroy hipfftDestroy
#define cufftExecC2C hipfftExecC2C
#define cufftExecZ2Z hipfftExecZ2Z
#define cufftHandle hipfftHandle
#define cufftPlanMany hipfftPlanMany
#define cufftResult_t hipfftResult_t

// helper_cuda.h adapters
#define __DRIVER_TYPES_H__

#else

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

#endif

#endif
