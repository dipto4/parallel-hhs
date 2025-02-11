/* particle system in SoA layout*/
#pragma once
#include<iostream>
#include<cuda_runtime.h>
#include "vec.cuh"
#include "utils.cuh"

struct partition {
    int N_slow;
    int N_fast;
};

template<typename T>
struct nbody_params {
    T eta;
    T eps;
}


template<typename T>
class particle_system {
    public:
        vec3<T>* pos;
        vec3<T>* vel;
        vec3<T>* acc;
        T* timestep;
        T* m;
        int N;
        char store; // either 'g' or 'c' indicating where the 
                    // data is stored so it is not illegally 
                    // accessed
        particle_system(int N_init, char s) : N(N_init), store(s) {
            pos = nullptr;
            vel = nullptr;
            acc = nullptr;
            m = nullptr;
        } 
        
        void gpu_alloc();
        void cpu_alloc();
        
        void gpu_free();
        void cpu_free();

        void host_to_gpu(particle_system* host_system);
        void gpu_to_host(particle_system* gpu_system);

};

/* this is only required for the GPU version of the code so
   only GPU mallocs and frees and needed*/
template<typename T>
class nbodysystem_buffers {
    public:
        int* predicate;
        int* scanned_predicate;
        particle_system<T>* slow;
        particle_system<T>* fast;
        int N_total;
        partition* part;
        nbody_params<T>* params; 

        nbodysystem_buffers(N_init) : N_total(N_init) {
            std::cout<<"Initializing slow, fast, predicate and scanned_predicate buffers"
            gpuErrchk( cudaMalloc(&predicate, sizeof(int) * N_init) );
            gpuErrchk( cudaMalloc(&scanned_predicate) , sizeof(int) * N_init );
            gpuErrchk( cudaMalloc(&part), sizeof(partition) );
            gpuErrchk( cudaMalloc(&params), sizeof(params) );
            slow = new particle_system<T>(N_init, 'g');
            fast = new particle_system<T>(N_init, 'g');
            slow->gpu_alloc();
            fast->gpu_alloc();
            //gpuErrchk(  );
            //gpuErrchk(  );
        }

        ~nbodysystem_buffers() {
            gpuErrchk ( cudaFree(predicate) );
            gpuErrchk ( cudaFree(scanned_predicate) );
            gpuErrchk ( cudaFree(partition) );
            gpuErrchk ( cudaFree(nbody_params) );
            slow->gpu_free();
            fast->gpu_free();
        }
};


template<typename T>
inline void particle_system::gpu_alloc() {
    // check if particle system correct
    if(store == 'c')  {
        std::cerr<<"Particle system store type is CPU but trying to allocate on the GPU! Exiting";
        exit(1);
    }

    size_t total_vector_size = N * sizeof(vec3<T>);
    size_t total_nonvector_size = N * sizeof(T);
    
    gpuErrchk ( cudaMalloc(&pos, total_vector_size) );
    gpuErrchk ( cudaMalloc(&vel, total_vector_size) );
    gpuErrchk ( cudaMalloc(&acc, total_vector_size) );
    gpuErrchk ( cudaMalloc(&timestep , total_nonvector_size) );
    gpuErrchk ( cudaMalloc(&m , total_nonvector_size) );

}

template<typename T>
inline void particle_system::gpu_free() {
    if(store == 'c')  {
        std::cerr<<"Particle system store type is CPU but trying to free on the GPU! Exiting";
        exit(1);
    }
    
    gpuErrchk ( cudaFree(pos) );
    gpuErrchk ( cudaFree(vel) );
    gpuErrchk ( cudaFree(acc) );
    gpuErrchk ( cudaFree(timestep) );
    gpuErrchk ( cudaFree(m) );
    
}

template<typename T>
inline void particle_system::cpu_alloc() {
     if(store == 'g')  {
        std::cerr<<"Particle system store type is GPU but trying to allocate on the CPU! Exiting";
        exit(1);
    }
    
    pos = new vec3<T>[N];
    vel = new vec3<T>[N];
    acc = new vec3<T>[N];
    timestep = new T[N];
    m = new T[N];
}


template<typename T>
inline void particle_system::cpu_free() {
    if(store == 'g')  {
        std::cerr<<"Particle system store type is GPU but trying to free on the CPU! Exiting";
        exit(1);
    }

    delete [] pos;
    delete [] vel;
    delete [] acc;
    delete [] timestep;
    delete [] m;
    
}

template<typename T>
inline void particle_system::host_to_gpu(particle_system* host_system) {
    if(store == 'c') {
        std::cerr<<"Particle system store type is CPU but trying to transfer from the CPU! Exiting";
        exit(1);
    }
    
    size_t total_vector_size = N * sizeof(vec3<T>);
    size_t total_nonvector_size = N * sizeof(T);
    
    gpuErrchk ( cudaMemcpy(pos, host_system->pos, total_vector_size, cudaMemcpyHostToDevice ) ) ;
    gpuErrchk ( cudaMemcpy(vel, host_system->vel, total_vector_size, cudaMemcpyHostToDevice ) ) ;
    gpuErrchk ( cudaMemcpy(acc, host_system->acc, total_vector_size, cudaMemcpyHostToDevice ) ) ;
    gpuErrchk ( cudaMemcpy(m, host_system->m, total_nonvector_size, cudaMemcpyHostToDevice ) ) ;
    gpuErrchk ( cudaMemcpy(timestep, host_system->timestep, total_nonvector_size, cudaMemcpyHostToDevice ) ) ;

}

template<typename T>
inline void particle_sytem::gpu_to_host(particle_system* gpu_system) {
    if(store == 'g') {
        std::cerr<<"Particle system store type is GPU but trying to transfer from the GPU! Exiting";
        exit(1);
    }
    
    size_t total_vector_size = N * sizeof(vec3<T>);
    size_t total_nonvector_size = N * sizeof(T);
    
    gpuErrchk ( cudaMemcpy(pos, gpu_system->pos, total_vector_size, cudaMemcpyDeviceToHost ) ) ;
    gpuErrchk ( cudaMemcpy(vel, gpu_system->vel, total_vector_size, cudaMemcpyDeviceToHost ) ) ;
    gpuErrchk ( cudaMemcpy(acc, gpu_system->acc, total_vector_size, cudaMemcpyDeviceToHost ) ) ;
    gpuErrchk ( cudaMemcpy(m, gpu_system->m, total_nonvector_size, cudaMemcpyDeviceToHost ) ) ;
    gpuErrchk ( cudaMemcpy(timestep, gpu_system->timestep, total_nonvector_size, cudaMemcpyDeviceToHost ) ) ;

}
