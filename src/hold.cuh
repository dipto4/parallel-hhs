// CUDA parallel version for the HOLD integrator
// runs completely on NVIDIA GPUs
// Diptajyoti Mukherjee

#pragma once
#include<cuda_runtime.h>
#include "drift.cuh"
#include "kick.cuh"
#include "split.cuh"
#include "particle_system.cuh"
#include "update_system.cuh"
#include "timestep.cuh"

#define UNROLL_FAC 1
// should include a timestep as well
template<typename T, int BLOCKSIZE>
void hold_step(int clevel, nbodysystem_globals<T>* globals, particle_system<T>* sys, 
        const int N_total, T stime, T etime, T dt, bool calc_timestep);

template<typename T>
void hold(int clevel, particle_system<T>* sys, 
        nbodysystem_buffers<T>* buffers, 
        const int N_total, T stime, T etime, T dt, bool calc_timestep);


/*
template<typename T>
void hold(int clevel, particle_system* sys, 
        nbodysystem_buffers* buffers, 
        const int N_total, T stime, T dt, bool calc_timestep) {
    int clevel = 0;
    if(calc_timestep) timesteps(total);

    hold_step(clevel + 1, sys, buffers, N_total, stime, stime+dt, dt, false);
}*/


/*  TODO: extra routine needed where the total particle system is updated
   at the end of each hold step

    Dipto (Feb 13) : How does this work?
    Dipto (Feb 16) : Cheap fix is to add a parent index to every slow fast 
    split in subsequent levels

    TODO: extra routine needed where new slow and fast buffers are created
    during each clevel

    Dipto (Feb 13): Added this
*/

template<typename T, int BLOCKSIZE>
void hold_step(int clevel, nbodysystem_globals<T>* globals, particle_system<T>* total, 
        const int N_total, T stime, T etime, T dt, bool calc_timestep) {
    
    // add temporary buffers here
    nbodysystem_buffers<T>* clevel_buffers = new nbodysystem_buffers<T>(N_total);
 
#ifdef DEBUG_HOLD
    //debug_cuPrintArr<<<1,1>>>(total->timestep, N_total); 
    printf("clevel = %i, dt = %f, slow.n = %i, fast.n = %i [BEFORE]\n", clevel, dt, clevel_buffers->part->N_slow, clevel_buffers->part->N_fast);
    printf("positions [BEFORE]\n");
    debug_cuPrintVecArr<<<1,1>>>(total->pos, N_total);
    cudaDeviceSynchronize();
    printf("velocities [BEFORE]\n");
    debug_cuPrintVecArr<<<1,1>>>(total->vel, N_total);
    cudaDeviceSynchronize();
    printf("mass [BEFORE]\n");
    debug_cuPrintArr<<<1,1>>>(total->m, N_total);
    cudaDeviceSynchronize();


#endif
   
    if(calc_timestep) {
        timesteps<T, BLOCKSIZE>(total, globals, N_total);
    }
    // remember to add the relevant template information
    split<T, BLOCKSIZE, UNROLL_FAC>(total, 
            clevel_buffers->slow, clevel_buffers->fast,
            globals->predicate, globals->scanned_predicate,
            dt, N_total, 
            clevel_buffers->part);
    
    cudaDeviceSynchronize();

#ifdef DEBUG_HOLD
    //debug_cuPrintArr<<<1,1>>>(total->timestep, N_total); 
    printf("clevel = %i, dt = %f, slow.n = %i, fast.n = %i\n", clevel, dt, clevel_buffers->part->N_slow, clevel_buffers->part->N_fast);
    printf("timesteps\n");
    debug_cuPrintArr<<<1,1>>>(total->timestep, N_total);
    cudaDeviceSynchronize();
    printf("positions\n");
    debug_cuPrintVecArr<<<1,1>>>(total->pos, N_total);
    cudaDeviceSynchronize();
    printf("velocities\n");
    debug_cuPrintVecArr<<<1,1>>>(total->vel, N_total);
    cudaDeviceSynchronize();
    printf("mass\n");
    debug_cuPrintArr<<<1,1>>>(total->m, N_total);
    cudaDeviceSynchronize();

#endif


    if(clevel_buffers->part->N_fast == 0) {
#ifdef DEBUG_INTEGRATOR
    globals->print_simtime();  
#endif
    globals->simtime += dt;
        /* update the global simtime*/
        /* to do that have another struct that keeps
           track of the global simtime */
    }

    if(clevel_buffers->part->N_fast > 0)  
        hold_step<T, BLOCKSIZE>(clevel + 1, 
                globals,
                clevel_buffers->fast, clevel_buffers->part->N_fast, 
                stime, stime + dt / 2, dt / 2, 
                false);

    if(clevel_buffers->part->N_slow > 0)
       drift<T, BLOCKSIZE>(clevel_buffers->slow, dt, (T) 0.5 , clevel_buffers->part->N_slow);
    
#ifdef DEBUG_HOLD_SF
    if(clevel_buffers->part->N_slow > 0 && clevel==0) {
        printf("slow-fast after D\n");
        debug_cuPrintVecArr<<<1,1>>>(clevel_buffers->slow->vel, clevel_buffers->part->N_slow);
        //debug_cuPrintVecArr<<<1,1>>>(clevel_buffers->fast->vel, clevel_buffers->part->N_fast);
    cudaDeviceSynchronize();
    }
#endif

    // TODO: add eps parameter here
    if(clevel_buffers->part->N_slow > 0) {
        kick_slow<T, BLOCKSIZE>(clevel_buffers->slow, dt, (T) 1.0, clevel_buffers->part->N_slow);

        if(clevel_buffers->part->N_fast > 0) {
            //TODO: add eps parameter here
            kick_sf<T, BLOCKSIZE>(clevel_buffers->fast, clevel_buffers->slow, (T) 1.0, dt, clevel_buffers->part->N_slow, clevel_buffers->part->N_fast);
            kick_sf<T, BLOCKSIZE>(clevel_buffers->slow, clevel_buffers->fast, (T) 1.0, dt, clevel_buffers->part->N_fast, clevel_buffers->part->N_slow);
            //kick_sf<T, BLOCKSIZE>(clevel_buffers->fast, clevel_buffers->slow, (T) 1.0, dt, clevel_buffers->part->N_slow, clevel_buffers->part->N_fast);
        }
    }
#ifdef DEBUG_HOLD_SF
    if(clevel_buffers->part->N_slow > 0 && clevel==0) {
        printf("slow-fast after DK\n");
        debug_cuPrintVecArr<<<1,1>>>(clevel_buffers->slow->vel, clevel_buffers->part->N_slow);
        //debug_cuPrintVecArr<<<1,1>>>(clevel_buffers->fast->vel, clevel_buffers->part->N_fast);
    cudaDeviceSynchronize();
    }
#endif

    // add eps parameter here
    if(clevel_buffers->part->N_slow > 0)
        drift<T, BLOCKSIZE>(clevel_buffers->slow, dt, (T) 0.5, clevel_buffers->part->N_slow);
     // values in clevel-1 need to be updated here
    

     //update_system<T,BLOCKSIZE>(total, clevel_buffers->fast, clevel_buffers->part->N_fast);    
    //update_system<T,BLOCKSIZE>(total, clevel_buffers->slow, clevel_buffers->part->N_slow);    

   
    if(clevel_buffers->part->N_fast > 0)
        hold_step<T, BLOCKSIZE>(clevel + 1,
                globals,
            clevel_buffers->fast, 
            clevel_buffers->part->N_fast, stime, stime + dt / 2, dt / 2, 
            true);
   

    // values in clevel-1 need to be updated here
    update_system<T,BLOCKSIZE>(total, clevel_buffers->fast, clevel_buffers->part->N_fast);    
    update_system<T,BLOCKSIZE>(total, clevel_buffers->slow, clevel_buffers->part->N_slow);    


}



// hold step has the following substeps

// calculate_timestep
// split system into slow and fast
// hold for fast subsystem
// dkd for fast subssytem
// calculte S-S and S-F interacitons
// hold system again


