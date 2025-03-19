#pragma once

#include "particle_system.cuh"
#define HUGE 1.e30


template<typename T, int BLOCKSIZE>
void timesteps(particle_system<T>* sys, nbodysystem_globals<T>* globals, const int N);

template<typename T, int BLOCKSIZE>
__device__ void _get_pp_timestep(vec3<T>& sink_pos, vec3<T>& sink_vel, T sink_m,
        vec3<T>* __restrict__ source_pos, vec3<T>* __restrict__ source_vel,
        T* __restrict__ source_m, T& minval, nbody_params<T>* params,
        int iGlobal, int jTile, const int N) {

//#pragma unroll
    for(int jLocal = 0 ; jLocal < BLOCKSIZE; jLocal++) {
        int jGlobal = jLocal + jTile;

        if(jGlobal >= N) break;

        if(jGlobal == iGlobal) continue;

        vec3<T> xj = source_pos[jLocal];
        vec3<T> vj = source_vel[jLocal];

        vec3<T> dr = sink_pos - xj;
        vec3<T> dv = sink_vel - vj;

        T dr2 = dr.norm2();
        T dv2 = dv.norm2();
    
        //printf("from timestep.cuh jglobal=%i, dr2=%f\n",jGlobal,dr2);
        //printf("from timestep.cuh jglobal=%i, dv2=%f\n",jGlobal,dv2);

        T r = sqrt(dr2);
        T vdotdr2 = (dr * dv) / (dr2);

        T tau = params->eta / M_SQRT2 * r * sqrt(r / (source_m[jLocal] + sink_m));
        
        //printf("from timestep.cuh jglobal=%i, 1/tau=%f\n",jGlobal, source_m[jLocal] + sink_m);
        T dtau = 3 * tau * vdotdr2 / 2;
        dtau = (dtau < 1) ? dtau : 1;
        tau /= (1 - dtau / 2);
 
        if(tau < minval) minval = tau;

        tau = params->eta * r / sqrt(dv2);

        //printf("from timestep.cuh jglobal=%i, 1/tau=%f\n",jGlobal, source_m[jLocal] + sink_m);
        dtau = tau * vdotdr2 * (1 + (sink_m + source_m[jLocal]) / (dv2 * r));
        dtau = (dtau < 1)? dtau: 1;

        tau /= (1 - dtau / 2);
        

        if(tau < minval) minval = tau;


    }

        
}

template<typename T, int BLOCKSIZE>
__global__ void _timesteps(vec3<T>* __restrict__ pos, vec3<T>* __restrict__ vel,
        T* __restrict__ m, T* __restrict__ timestep, nbody_params<T>* params, const int N) {
    int tid = BLOCKSIZE * blockIdx.x + threadIdx.x;
    
    if(tid >= N) return;
    vec3<T> xi = pos[tid];
    vec3<T> vi = vel[tid];
    T mi = m[tid];

    T minval = (T) HUGE;
    __shared__ vec3<T> tilePos[BLOCKSIZE];
    __shared__ vec3<T> tileVel[BLOCKSIZE];
    __shared__ T tileM[BLOCKSIZE];

   
    for(int jTile = 0 ; jTile < N; jTile += BLOCKSIZE) {
        int jLoad = jTile + threadIdx.x;

        if(jLoad < N) {
            tilePos[threadIdx.x] = pos[jLoad];
            tileVel[threadIdx.x] = vel[jLoad];
            tileM[threadIdx.x] = m[jLoad];
        }
        __syncthreads();

        _get_pp_timestep<T,BLOCKSIZE>(xi, vi, mi, 
                tilePos, tileVel, tileM, minval, params,
                tid, jTile, N);

        __syncthreads();
    }
    
    timestep[tid] = minval;
}



template<typename T, int BLOCKSIZE>
void timesteps(particle_system<T>* sys, nbodysystem_globals<T>* globals, const int N) {
    dim3 block(BLOCKSIZE);
    dim3 grid((block.x + N - 1) / BLOCKSIZE);
    
#ifdef CUDA_DEBUG
    _timesteps<T, BLOCKSIZE><<<grid,block>>>(sys->pos, sys->vel, sys->m, sys->timestep, globals->params, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
#else
    _timesteps<T, BLOCKSIZE><<<grid,block>>>(sys->pos, sys->vel, sys->m, sys->timestep, globals->params, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
#endif

}
