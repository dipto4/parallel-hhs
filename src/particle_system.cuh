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
};


template<typename T>
class particle_system {
    public:
        vec3<T>* pos;
        vec3<T>* vel;
        vec3<T>* acc;
        T* timestep;
        T* m;
        int N;
        int* id;
        int* parent_id;
        char store; // either 'g' or 'c' indicating where the 
                    // data is stored so it is not illegally 
                    // accessed
        particle_system(int N_init, char s) : N(N_init), store(s) {
            pos = nullptr;
            vel = nullptr;
            acc = nullptr;
            m = nullptr;
            id = nullptr;
            parent_id = nullptr;
        } 
        
        void gpu_alloc();
        void cpu_alloc();
        
        void gpu_free();
        void cpu_free();

        void host_to_gpu(particle_system* host_system);
        void gpu_to_host(particle_system* gpu_system);

};

/* this is only required for the GPU version of the code so
   only GPU mallocs and frees and needed
TODO: store the partition and nbody_params buffers in unified memory
    so the cpu can access them too
    ensure proper synchronization!

Dipto (Feb 13) : changed to unified memory for some buffers    
 */

template<typename T>
class nbodysystem_globals {
    public:
        int* predicate;
        int* scanned_predicate;
        
        T simtime;

        int N_total;
        nbody_params<T>* params;
         
        nbodysystem_globals(int N_init, T eta, T eps) : N_total(N_init) {
            std::cout<<"Initializing predicate and scanned_predicate buffers"<<std::endl;
            gpuErrchk ( cudaMalloc(&predicate , sizeof(int) * N_total) );
            gpuErrchk ( cudaMalloc(&scanned_predicate , sizeof(int) * N_total) );
            
            gpuErrchk ( cudaMallocManaged(&params, sizeof(nbody_params<T>)) );
            params->eta = eta;
            params->eps = eps;

            simtime = (T) 0.0;
        }

        ~nbodysystem_globals() {
            gpuErrchk ( cudaFree(predicate) );
            gpuErrchk ( cudaFree(scanned_predicate) );
            gpuErrchk ( cudaFree(params) );

        }

        void flush_simtime() {
            simtime = (T) 0.0;
        }
        
        void print_simtime() {
            std::cout<<"simtime: "<<simtime<<std::endl;
        }


};


template<typename T>
class nbodysystem_buffers {
    public:
        particle_system<T>* slow;
        particle_system<T>* fast;
        int N_total;
        partition* part;

        nbodysystem_buffers(int N_init) : N_total(N_init) {
#ifdef CUDA_DEBUG
            std::cout<<"Initializing slow and fast buffers"<<std::endl;
#endif
            slow = new particle_system<T>(N_init, 'g');
            fast = new particle_system<T>(N_init, 'g');
            slow->gpu_alloc();
            fast->gpu_alloc();
            //gpuErrchk(  );
            //gpuErrchk(  );
        }

        ~nbodysystem_buffers() {
            gpuErrchk ( cudaFree(part) );
            slow->gpu_free();
            fast->gpu_free();
        }
};


template<typename T>
inline void particle_system<T>::gpu_alloc() {
    // check if particle system correct
    if(store == 'c')  {
        std::cerr<<"Particle system store type is CPU but trying to allocate on the GPU! Exiting"<<std::endl;
        exit(1);
    }

    size_t total_vector_size = N * sizeof(vec3<T>);
    size_t total_nonvector_size = N * sizeof(T);
    
    gpuErrchk ( cudaMalloc(&pos, total_vector_size) );
    gpuErrchk ( cudaMalloc(&vel, total_vector_size) );
    gpuErrchk ( cudaMalloc(&acc, total_vector_size) );
    gpuErrchk ( cudaMalloc(&timestep , total_nonvector_size) );
    gpuErrchk ( cudaMalloc(&m , total_nonvector_size) );
    gpuErrchk ( cudaMalloc(&id , N * sizeof(int)) );
}

template<typename T>
inline void particle_system<T>::gpu_free() {
    if(store == 'c')  {
        std::cerr<<"Particle system store type is CPU but trying to free on the GPU! Exiting"<<std::endl;
        exit(1);
    }
    
    gpuErrchk ( cudaFree(pos) );
    gpuErrchk ( cudaFree(vel) );
    gpuErrchk ( cudaFree(acc) );
    gpuErrchk ( cudaFree(timestep) );
    gpuErrchk ( cudaFree(m) );
    gpuErrchk ( cudaFree(id) );
    
}

template<typename T>
inline void particle_system<T>::cpu_alloc() {
     if(store == 'g')  {
        std::cerr<<"Particle system store type is GPU but trying to allocate on the CPU! Exiting"<<std::endl;
        exit(1);
    }
    
    pos = new vec3<T>[N];
    vel = new vec3<T>[N];
    acc = new vec3<T>[N];
    timestep = new T[N];
    m = new T[N];
    id = new int[N];
}


template<typename T>
inline void particle_system<T>::cpu_free() {
    if(store == 'g')  {
        std::cerr<<"Particle system store type is GPU but trying to free on the CPU! Exiting"<<std::endl;
        exit(1);
    }

    delete [] pos;
    delete [] vel;
    delete [] acc;
    delete [] timestep;
    delete [] m;
    delete [] id;
    
}

template<typename T>
inline void particle_system<T>::host_to_gpu(particle_system<T>* host_system) {
    if(store == 'c') {
        std::cerr<<"Particle system store type is CPU but trying to transfer from the CPU! Exiting"<<std::endl;
        exit(1);
    }
    
    size_t total_vector_size = N * sizeof(vec3<T>);
    size_t total_nonvector_size = N * sizeof(T);
    
    gpuErrchk ( cudaMemcpy(pos, host_system->pos, total_vector_size, cudaMemcpyHostToDevice ) ) ;
    gpuErrchk ( cudaMemcpy(vel, host_system->vel, total_vector_size, cudaMemcpyHostToDevice ) ) ;
    gpuErrchk ( cudaMemcpy(acc, host_system->acc, total_vector_size, cudaMemcpyHostToDevice ) ) ;
    gpuErrchk ( cudaMemcpy(m, host_system->m, total_nonvector_size, cudaMemcpyHostToDevice ) ) ;
    gpuErrchk ( cudaMemcpy(timestep, host_system->timestep, total_nonvector_size, cudaMemcpyHostToDevice ) ) ;
    gpuErrchk ( cudaMemcpy(id, host_system->id, N * sizeof(int), cudaMemcpyHostToDevice ) ) ;
}

template<typename T>
inline void particle_system<T>::gpu_to_host(particle_system<T>* gpu_system) {
    if(store == 'g') {
        std::cerr<<"Particle system store type is GPU but trying to transfer from the GPU! Exiting"<<std::endl;
        exit(1);
    }
    
    size_t total_vector_size = N * sizeof(vec3<T>);
    size_t total_nonvector_size = N * sizeof(T);
    
    gpuErrchk ( cudaMemcpy(pos, gpu_system->pos, total_vector_size, cudaMemcpyDeviceToHost ) ) ;
    gpuErrchk ( cudaMemcpy(vel, gpu_system->vel, total_vector_size, cudaMemcpyDeviceToHost ) ) ;
    gpuErrchk ( cudaMemcpy(acc, gpu_system->acc, total_vector_size, cudaMemcpyDeviceToHost ) ) ;
    gpuErrchk ( cudaMemcpy(m, gpu_system->m, total_nonvector_size, cudaMemcpyDeviceToHost ) ) ;
    gpuErrchk ( cudaMemcpy(timestep, gpu_system->timestep, total_nonvector_size, cudaMemcpyDeviceToHost ) ) ;
    gpuErrchk ( cudaMemcpy(id, gpu_system->id, N * sizeof(int), cudaMemcpyDeviceToHost ) ) ;

}
