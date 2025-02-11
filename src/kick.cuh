#pragma once
#include<cuda_runtime.h>
#include "vec.cuh"
#include "utils.cuh"

template<typename T, int BLOCKSIZE>
void kick_slow(const vec3<T>* __restrict__ pos, 
        vec3<T>* __restrict__ vel, 
        vec3<T>* __restrict__ acc,
        const T* __restrict__ m,
        T* dt, T* fac, const int N);

template<typename T, int BLOCKSIZE>
void kick_sf(const vec3<T>* __restrict__ so_pos,
        const T* __restrict__ so_m,
        const vec3<T>* __restrict__ si_pos, 
        vec3<T>* __restrict__ si_vel, 
        vec3<T>* __restrict__ si_acc,
        const T* __restrict__ si_m,
        T* dt, T* fac, const int N);

template<typename T, int BLOCKSIZE>
__device__ void pp_force(const vec3<T>& sink, const vec3<T>* __restrict__ sources, const T* __restrict__ m, vec3<T>& acc) {
#pragma unroll
    for(size_t i = 0 ; i < BLOCKSIZE ; i++) {
        vec3<T> dx = sources[i] - sink;
        T d2 = dx.norm2();
        if(d2 == (T) 0.0) continue;
        T d = sqrt(d2);
        T one_over_d3 = 1.0/(d2 * d);
        vec3<T> ai =  dx * (-m[i] * one_over_d3);
        acc = acc + ai;
    }

}

template<typename T, int BLOCKSIZE>
__global__ void _kick_slow(const vec3<T>* __restrict__ pos,
        vec3<T>* __restrict__ vel,
        vec3<T>* __restrict__ acc,
        T* __restrict__ m,
        const T dt, const T fac, const int N) {

    // steps: O(N^2) force calculation
    // kick step

    __shared__ vec3<T> cache[BLOCKSIZE];
    __shared__ T cache_m[BLOCKSIZE];

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    vec3<T> cur_part = pos[tid];
    vec3<T> acc_i;
    for(size_t i = 0, tile = 0; i < N; i+=BLOCKSIZE, tile++) {
        cache[threadIdx.x] = pos[blockDim.x * tile + threadIdx.x];
        cache_m[threadIdx.x] = m[blockDim.x * tile + threadIdx.x]; 
        __syncthreads();

        pp_force<T,BLOCKSIZE>(cur_part, cache, cache_m, acc_i);

        __syncthreads();

    }

    acc[tid] = acc_i;    
    
    vel[tid] = vel[tid] + acc_i * fac;



}


template<typename T, int BLOCKSIZE>
__global__ void _kick_sf(const vec3<T>* __restrict__ so_pos,
        T* __restrict__ so_m,
        const vec3<T>* __restrict__ si_pos, 
        vec3<T>* __restrict__ si_vel, 
        vec3<T>* __restrict__ si_acc, 
        const T dt, const T fac, const int N
        ) {

    __shared__ vec3<T> cache[BLOCKSIZE];
    __shared__ T cache_m[BLOCKSIZE];

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    vec3<T> cur_part = si_pos[tid];
    vec3<T> acc_i;
    for(size_t i = 0, tile = 0; i < N; i+=BLOCKSIZE, tile++) {
        cache[threadIdx.x] = so_pos[blockDim.x * tile + threadIdx.x];
        cache_m[threadIdx.x] = so_m[blockDim.x * tile + threadIdx.x];

        __syncthreads();

        pp_force<T,BLOCKSIZE>(cur_part, cache, cache_m, acc_i);

        __syncthreads();

    }

    si_acc[tid] = acc_i;    

    si_vel[tid] = si_vel[tid] +  acc_i * (fac * dt);
}


