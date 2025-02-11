#pragma once
#include<cuda_runtime.h>
#include "utils.h"


template<typename T>
void drift(vec3<T>* __restrict__ pos,
        const vec3<T>* __restrict__ vel, 
        const T dt, 
        const T fac,
        const int N);



// cuda kernels here
template<typename T>
__global__ void _drift(vec3<T>* __restrict__ pos,
         const vec3<T>* __restrict__ vel,
         const T dt, 
         const T fac, 
         const int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid < N) {
        pos[tid] += vel[tid] * (dt* fac);
    }

}

template<typename T, int BLOCKSIZE> 
void drift(vec3<T>* __restrict__ pos,
         const vec3<T>* __restrict__ vel,
         const T dt,
         const T fac,
         const int N) {
    dim3 block(BLOCKSIZE);
    dim3 grid((BLOCKSIZE+N-1)/BLOCKSIZE);

    _drift<T><<<grid,block>>>(pos, vel, dt, fac, N);

}

