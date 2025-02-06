#pragma once
#include<cuda_runtime.h>
#include "utils.h"


template<typename T>
void drift(T* __restrict__ x, T* __restrict__ y, T* __restrict__ z,
        const T* __restrict__ vx, const T* __restrict__ vy, const T* __restrict__ vz, 
        const T dt, 
        const T fac,
        const int N);



// cuda kernels here
template<typename T>
__global__ void _drift(T* __restrict__ x, T* __restrict__ y, T* __restrict__ z,
         const T* __restrict__ vx, const T* __restrict__ vy, const T* __restrict__ vz,
         const T dt, 
         const T fac, 
         const int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid < N) {
        x[tid] += fac * vx[tid];
        y[tid] += fac * vy[tid];
        z[tid] += fac * vz[tid];
    }

}

template<typename T, int BLOCKSIZE> 
void drift(T* __restrict__ x, T* __restrict__ y, T* __restrict__ z,
         const T* __restrict__ vx, const T* __restrict__ vy, const T* __restrict__ vz,
         const T dt,
         const T fac,
         const int N) {
    dim3 block(BLOCKSIZE);
    dim3 grid((BLOCKSIZE+N-1)/BLOCKSIZE);

    _drift<T><<<grid,block>>>(x, y, z, vx, vy, vz, dt, fac, N);

}

