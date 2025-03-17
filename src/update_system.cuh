#pragma once
#include<cuda_runtime.h>
#include "particle_system.cuh"

// kernel to update the particles
template<typename T>
void update_system(particle_system<T>* prev_level_sys, particle_system<T>* cur_level_sys, int N);

// updates the system from values in sys2 to values in sys1
template<typename T>
__global__ void _update_system(particle_system<T>* prev_level_sys, particle_system<T>* cur_level_sys, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid < N) {
        int p_id = cur_level_sys->parent_id[tid];

        prev_level_sys->pos[p_id] = cur_level_sys->pos[tid];
        prev_level_sys->vel[p_id] = cur_level_sys->vel[tid];
        prev_level_sys->m[p_id] = cur_level_sys->m[tid];


        prev_level_sys->timestep[p_id] = cur_level_sys->timestep[tid];

    }
}



template<typename T, int BLOCKSIZE>
void update_system(particle_system<T>* prev_level_sys, particle_system<T>* cur_level_sys, int N) {
    
    if(N == 0) return;

    dim3 block(BLOCKSIZE);
    dim3 grid((N+block.x-1)/block.x);

    _update_system<<<grid,block>>>(prev_level_sys->d_ptr, cur_level_sys->d_ptr, N);


}
