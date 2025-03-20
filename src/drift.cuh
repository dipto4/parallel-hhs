#pragma once
#include<cuda_runtime.h>
#include "particle_system.cuh"
#include "utils.cuh"


template<typename T, int BLOCKSIZE>
void drift(particle_system<T>* __restrict__ sys, 
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
        pos[tid] = pos[tid] + vel[tid] * (dt* fac);
    }

}

template<typename T, int BLOCKSIZE> 
void drift(particle_system<T>* __restrict__ sys,
         const T dt,
         const T fac,
         const int N) {
    dim3 block(BLOCKSIZE);
    dim3 grid((BLOCKSIZE+N-1)/BLOCKSIZE);

    _drift<T><<<grid,block>>>(sys->pos, sys->vel, dt, fac, N);
    //gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() );


}

