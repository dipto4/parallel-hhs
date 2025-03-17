#pragma once
#include<cuda_runtime.h>
#include "vec.cuh"
#include "utils.cuh"

template<typename T, int BLOCKSIZE>
void kick_slow(particle_system<T>* part,
        T dt, T fac, const int N);

template<typename T, int BLOCKSIZE>
void kick_sf(particle_system<T>* so_sys,
        particle_system<T>* si_sys,
        T dt, T fac, const int N);

template<typename T, int BLOCKSIZE>
__device__ void pp_force(const vec3<T>& sink, const vec3<T>* __restrict__ sources, const T* __restrict__ m, vec3<T>& acc, 
        const int iGlobal, const int jTile, const int N) {
//#pragma unroll
    for(size_t jLocal = 0 ; jLocal < BLOCKSIZE ; jLocal++) {
        int jGlobal = jLocal + jTile;
        if(jGlobal >= N) break;

        vec3<T> dx = sources[jLocal] - sink;
        T d2 = dx.norm2();
        if(d2 == (T) 0.0) continue;
        T d = sqrt(d2);
        T one_over_d3 = 1.0/(d2 * d);
        vec3<T> ai =  dx * (-m[jLocal] * one_over_d3);
        acc = acc + ai;
    }

}

template<typename T, int BLOCKSIZE>
__global__ void _kick_slow(const vec3<T>* __restrict__ pos,
        vec3<T>* __restrict__ vel,
        vec3<T>* __restrict__ acc,
        const T* __restrict__ m,
        const T dt, const T fac, const int N) {

    // steps: O(N^2) force calculation
    // kick step

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= N) return;

    vec3<T> cur_part = pos[tid];
    vec3<T> acc_i(0.0, 0.0, 0.0);
    __shared__ vec3<T> cache[BLOCKSIZE];
    __shared__ T cache_m[BLOCKSIZE];




    for(int jTile = 0; jTile < N; jTile+=BLOCKSIZE) {
        int jLoad = jTile + threadIdx.x;
        if(jLoad < N) {
            cache[threadIdx.x] = pos[jLoad];
            cache_m[threadIdx.x] = m[jLoad];
        }
        __syncthreads();

        pp_force<T,BLOCKSIZE>(cur_part, cache, cache_m, acc_i, tid, jTile, N);

        __syncthreads();

    }

    acc[tid] = acc_i;    
    
    vel[tid] = vel[tid] + acc_i * (fac * dt);



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
    vec3<T> acc_i(0.0, 0.0, 0.0);
    for(size_t i = 0, tile = 0; i < N; i+=BLOCKSIZE, tile++) {
        cache[threadIdx.x] = so_pos[blockDim.x * tile + threadIdx.x];
        cache_m[threadIdx.x] = so_m[blockDim.x * tile + threadIdx.x];

        __syncthreads();

        //pp_force<T,BLOCKSIZE>(cur_part, cache, cache_m, acc_i);

        __syncthreads();

    }

    si_acc[tid] = acc_i;    

    si_vel[tid] = si_vel[tid] +  acc_i * (fac * dt);
}



template<typename T, int BLOCKSIZE>
void kick_slow(particle_system<T>* sys,
        T dt, T fac, const int N) {
    dim3 block(BLOCKSIZE);
    dim3 grid((N+block.x-1)/block.x);
    _kick_slow<T,BLOCKSIZE><<<grid,block>>>(sys->pos, sys->vel, sys->acc, sys->m,
            dt, fac, N);

}

template<typename T, int BLOCKSIZE>
void kick_sf(particle_system<T>* so_sys,
        particle_system<T>* si_sys,
        T dt, T fac, const int N) {
    dim3 block(BLOCKSIZE);
    dim3 grid((N+block.x-1)/block.x);
    _kick_sf<T,BLOCKSIZE><<<grid,block>>>(so_sys->pos, so_sys->m, 
            si_sys->pos, si_sys->vel, si_sys->acc, dt, fac, N);
}

