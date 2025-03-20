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
        T dt, T fac, const int N, const int M);

template<typename T, int BLOCKSIZE>
__device__ void pp_force(const vec3<T>& sink, const vec3<T>* __restrict__ sources, const T* __restrict__ m, vec3<T>& acc, 
        const int iGlobal, const int jTile, const int N) {
//#pragma unroll
    for(int jLocal = 0 ; jLocal < BLOCKSIZE ; jLocal++) {
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
__device__ void pp_force2(const vec3<T>& sink, const vec3<T>* __restrict__ sources, const T* __restrict__ m, vec3<T>& acc, 
        const int iGlobal, const int jTile, const int N) {
//#pragma unroll
    for(int jLocal = 0 ; jLocal < BLOCKSIZE ; jLocal++) {
        int jGlobal = jLocal + jTile;
        if(jGlobal >= N) break;

        vec3<T> dx = sources[jLocal] - sink;
        T d2 = dx.norm2();
        /*printf("GPUDEBUG tid=%i, jLocal=%i, sources.x=%f, sources.y=%f, sources.z=%f, m=%f\n", iGlobal, jLocal, sources[jLocal].x, sources[jLocal].y, sources[jLocal].z, m[jLocal]);*/
        //printf("GPUDEBUG tid=%i, jLocal=%i, d2=%f\n", iGlobal, jLocal, d2);
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
    //if(tid >= N) return;
    bool isActive = (tid < N);
    vec3<T> cur_part = isActive ? pos[tid] : vec3<T>(0.0, 0.0, 0.0);
    //vec3<T> cur_part = pos[tid];
    vec3<T> acc_i(0.0, 0.0, 0.0);
    __shared__ vec3<T> cache[BLOCKSIZE];
    __shared__ T cache_m[BLOCKSIZE];


    //printf("GPUDEBUG tid=%i, vel.x=%f, vel.y=%f, vel.z=%f\n",tid, si_vel[tid].x, si_vel[tid].y, si_vel[tid].z);


    for(int jTile = 0; jTile < N; jTile+=BLOCKSIZE) {
        int jLoad = jTile + threadIdx.x;
        if(jLoad < N) {
            cache[threadIdx.x] = pos[jLoad];
            cache_m[threadIdx.x] = m[jLoad];
        } else {
            cache[threadIdx.x] = vec3<T>(0.0,0.0,0.0);
            cache_m[threadIdx.x] = (T) 0.0;
        }
        __syncthreads();
        if(isActive) {
            pp_force<T,BLOCKSIZE>(cur_part, cache, cache_m, acc_i, tid, jTile, N);
        }
        __syncthreads();

    }
    if(isActive) {
        acc[tid] = acc_i;    
    
        vel[tid] = vel[tid] + acc_i * (fac * dt);
    


    }
}

/* N : number of sink particles
   M : number of source particles */ 

template<typename T, int BLOCKSIZE>
__global__ void _kick_sf(const vec3<T>* __restrict__ so_pos,
        T* __restrict__ so_m,
        const vec3<T>* __restrict__ si_pos, 
        vec3<T>* __restrict__ si_vel, 
        vec3<T>* __restrict__ si_acc, 
        const T dt, const T fac, const int N,
        const int M) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    bool isActive = (tid < N);
    vec3<T> cur_part = isActive ? si_pos[tid] : vec3<T>(0.0, 0.0, 0.0);

    //if(tid >= N) return;

    __shared__ vec3<T> cache[BLOCKSIZE];
    __shared__ T cache_m[BLOCKSIZE];
    //printf("GPUDEBUG tid=%i, vel.x=%f, vel.y=%f, vel.z=%f\n",tid, si_vel[tid].x, si_vel[tid].y, si_vel[tid].z);

    //vec3<T> cur_part = si_pos[tid];
    vec3<T> acc_i(0.0, 0.0, 0.0);
    for(int jTile = 0 ; jTile < M; jTile+=BLOCKSIZE) {
        int jLoad = jTile + threadIdx.x;
        if(jLoad < M) {
            cache[threadIdx.x] = so_pos[jLoad];
            cache_m[threadIdx.x] = so_m[jLoad];
        } else {
            cache[threadIdx.x] = vec3<T>(0.0, 0.0, 0.0);
            cache_m[threadIdx.x] = (T)0.0;
        }
        __syncthreads();
        if(isActive) {
            pp_force2<T,BLOCKSIZE>(cur_part, cache, cache_m, acc_i, tid, jTile, M);
        }
        __syncthreads();

    }

    if (isActive) {

        si_acc[tid] = acc_i;    

        si_vel[tid] = si_vel[tid] +  acc_i * (fac * dt);
    }
}


template<typename T, int BLOCKSIZE>
void kick_slow(particle_system<T>* sys,
        T dt, T fac, const int N) {
    dim3 block(BLOCKSIZE);
    dim3 grid((N+block.x-1)/block.x);
    _kick_slow<T,BLOCKSIZE><<<grid,block>>>(sys->pos, sys->vel, sys->acc, sys->m,
            dt, fac, N);
        //gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() );

}

template<typename T, int BLOCKSIZE>
void kick_sf(particle_system<T>* so_sys,
        particle_system<T>* si_sys,
        T dt, T fac, const int N, const int M) {
    dim3 block(BLOCKSIZE);
    dim3 grid((N+block.x-1)/block.x);
    _kick_sf<T,BLOCKSIZE><<<grid,block>>>(so_sys->pos, so_sys->m, 
            si_sys->pos, si_sys->vel, si_sys->acc, dt, fac, N, M);
    //gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() );

}

