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
void hold_step();

template<typename T>
void hold(particle_system* sys, 
        nbodysystem_buffers* buffers, 
        const int N_total, T dt);


// hold step has the following substeps

// calculate_timestep
// split system into slow and fast
// hold for fast subsystem
// dkd for fast subssytem
// calculte S-S and S-F interacitons
// hold system again


