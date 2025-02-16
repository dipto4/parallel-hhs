// CUDA parallel version for the HOLD integrator
// runs completely on NVIDIA GPUs
// Diptajyoti Mukherjee

#pragma once

#include "drift.cuh"
#include "kick.cuh"
#include "scan.cuh"
#include "particle_system.cuh"

// should include a timestep as well

template<typename T>
void hold_step(int clevel, particle_system* sys, 
        nbodysystem_buffers* buffers, 
        const int N_total, T stime ,T etime, T dt,bool calc_timestep);

template<typename T>
void hold(int clevel, particle_system* sys, 
        nbodysystem_buffers* buffers, 
        const int N_total, T stime, T etime, T dt, bool calc_timestep);



template<typename T>
void hold(int clevel, particle_system* sys, 
        nbodysystem_buffers* buffers, 
        const int N_total, T stime, T dt, bool calc_timestep) {
    int clevel = 0;
    if(calc_timestep) timesteps(total);

    hold_step(clevel + 1, sys, buffers, N_total, stime, stime+dt, dt, false);
}


/*  TODO: extra routine needed where the total particle system is updated
   at the end of each hold step

    Dipto (Feb 13) : How does this work?


    TODO: extra routine needed where new slow and fast buffers are created
    during each clevel

    Dipto (Feb 13): Added this
*/

template<typename T>
void hold_step(int clevel, nbodysystem_globals<T>* globals, particle_system<T>* sys, 
        nbodysystem_buffers<T>* buffers,
        const int N_total, T stime, T etime, T dt, bool calc_timestep) {

    // add temporary buffers here
    nbodysystem_buffers<T>* clevel_buffers(N_total);
    
    if(calc_timestep) {
        timesteps(total);
    }
    // remember to add the relevant template information
    split<>(total, 
            clevel_buffers->slow, clevel_buffers->fast,
            globals->predicate, globals->scanned_predicate,
            dt, N_total, 
            buffers->part);
    
    cudaDeviceSynchronize();

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
        hold_step(clevel + 1, 
                buffers->fast,
                globals, 
                clevel_buffers, clevel_buffers->part->N_fast, 
                stime, stime + dt / 2, dt / 2, 
                false);

    if(clevel_buffers->part->N_slow > 0)
       drift(clevel_buffers->slow, dt, (T) 0.5 , clevel_buffers->part->N_slow);

    // TODO: add eps parameter here
    if(clevel_buffers->part->N_slow > 0) {
        kick_self(clevel_buffers->slow, dt, (T) 1.0, clevel_buffers->part->N_slow);

        if(clevel_buffers->part->N_fast > 0) {
            //TODO: add eps parameter here
            kick_sf(clevel_buffers->slow, clevel_buffers->fast, dt, (T) 1.0, dt, clevel_buffers->part->N_slow);
            kick_sf(clevel_buffers->fast, clevel_buffers->slow, dt, (T) 1.0, dt, clevel_buffers->part->N_slow);
        }
    }
    
    // add eps parameter here
    if(clevel_buffers->part->N_slow > 0)
        drift(clevel_buffers->slow, dt, (T) 0.5, clevel_buffers->part->N_slow);
    
    // values in clevel-1 need to be updated here
    update_system<T,BLOCKSIZE>(buffers->fast, clevel_buffers->fast, clevel_buffers->part->N_fast);    
    update_system<T,BLOCKSIZE>(buffers->fast, clevel_buffers->slow, clevel_buffers->part->N_slow);    

    if(clevel_buffers->part->N_fast > 0)
        hold_step(clevel + 1, 
            buffers->fast,
            globals,
            clevel_buffers, 
            clevel_buffers->part->N_fast, stime, stime + dt / 2, dt / 2, 
            true);

}



// hold step has the following substeps

// calculate_timestep
// split system into slow and fast
// hold for fast subsystem
// dkd for fast subssytem
// calculte S-S and S-F interacitons
// hold system again


