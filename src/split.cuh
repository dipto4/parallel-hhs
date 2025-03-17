/* first optimization: remove shared memory bank conflicts */
/* second optimization: block unrolling */
/* TODO: fix the variable names in the kernels
   _arr_slow_* should be _arr_fast_*
   fast and slow variable names have been interchanged*/
#pragma once
#include<cassert>
#include<cuda_runtime.h>
#include<cmath>
#include<cstdio>
#include "utils.cuh"
#include "particle_system.cuh"

/* macros to help calculate bank offset to avoid shared memory conflicts */
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n)((n) >>NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
template<int UNROLL>
__host__ void scan_predicate_small(int* predicate, int* scanned_predicate, const int N);


template<int UNROLL>
__host__ void scan_predicate_large(int* predicate, int* scanned_predicate, const int N, int BLOCKSIZE);


template<int UNROLL>
__host__ void scan_predicate_large_ev(int* predicate, int* scanned_predicate, const int N, int BLOCKSIZE);

template<typename T>
__global__ void debug_cuPrintArr(T* arr, const int size);

/* FOR DEBUGGING PURPOSES ONLY*/

template<typename T>
__global__ void debug_cuPrintArr(T* arr, const int size) {
    for(int i = 0 ; i < size; i++) {
        printf("i = %i, val = %f\n", i, arr[i] );

    }
}
template<typename T>
__global__ void debug_cuPrintVecArr(vec3<T>* p, const int size) {
    for(int i = 0 ; i < size; i++) {
        printf("i = %i, x = %f, y=%f, z=%f\n", i, p[i].x, p[i].y, p[i].z);
    }
}



template<int UNROLL>
__device__ void __forceinline__ _upsweep(int* _arr_slow, int tid, int BLOCKSIZE) {
    
    for(int stride = 1; stride<BLOCKSIZE; stride<<=1) {
        int idx = (tid + 1) * (stride << 1) - 1;
        if(idx < BLOCKSIZE) {
#pragma unroll
            for(int k = 0 ; k < UNROLL; k++) {
                int bo = k * BLOCKSIZE;
                _arr_slow[bo + idx + CONFLICT_FREE_OFFSET(bo + idx)] += _arr_slow[(idx-stride) + bo + CONFLICT_FREE_OFFSET((idx-stride) + bo)];
            }
        }
        __syncthreads();
    }
}


template<int UNROLL>
__device__ void __forceinline__ _downsweep(int* _arr_slow, int tid, int BLOCKSIZE) {
    
    for(int stride = BLOCKSIZE / 2; stride > 0; stride>>=1) {
        int idx = (tid + 1) * (stride << 1) - 1;
        
        if(idx < BLOCKSIZE) {
#pragma unroll
            for(int k = 0 ; k < UNROLL; k++) {
                int bo = k * BLOCKSIZE;
                int _tmp_s = _arr_slow[idx + bo + CONFLICT_FREE_OFFSET(idx+bo)];

                _arr_slow[idx + bo + CONFLICT_FREE_OFFSET(idx+bo)] += _arr_slow[(idx-stride) + bo + CONFLICT_FREE_OFFSET((idx-stride) + bo)];

                _arr_slow[(idx-stride) + bo + CONFLICT_FREE_OFFSET((idx-stride) + bo)] = _tmp_s;
            
            
            }
        }
        __syncthreads();
    }
}

//using the stream compaction trick to load into slow and fast arrays
/* device side kernels follow*/

template<typename T>
__global__ void _get_predicate(const T* __restrict__ total, 
        int* __restrict__ predicate, 
        const T pivot, 
        const int N) {
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid < N) {
        predicate[tid] = total[tid] <= pivot;
    }
}

/* single block Belloch Scan
   See NVIDIA GPU Gems 3 Chapter 39
   for more details.
   The work efficient algorithm invovles
   1. Upsweep phase (reduction)
   2. Setting the last element to zero
   3. Downsweep phase

   Algorithmic complexity O(N)
   */


template<int UNROLL>
__global__ void _scan_predicate_single_block(const int* __restrict__ predicate, int* __restrict__ scanned_predicate, 
        const int N, 
        const int ELEMENTS_PER_BLOCK) {
    
    __shared__ int _block_factor[UNROLL];
    extern __shared__ int _smem[];

    int* _pred_slow = (int*) _smem;
    int* _scan_pred_slow = _pred_slow + ELEMENTS_PER_BLOCK;
    
    int BLOCKSIZE = blockDim.x;

    int tid = threadIdx.x; //+ blockDim.x * blockIdx.x;
    const int* p_total = predicate;//blockIdx.x * blockDim.x;
    
    //int bankOffset = CONFLICT_FREE_OFFSET(tid);
    
#pragma unroll
    for(int k = 0; k < UNROLL; k++) {
        int i = tid + k * BLOCKSIZE;
        if(i < N) {
            int p_data = p_total[i];
            _pred_slow[i + CONFLICT_FREE_OFFSET(i)] = p_data;
            _scan_pred_slow[i + CONFLICT_FREE_OFFSET(i)] = p_data;
        } else {
            _pred_slow[i + CONFLICT_FREE_OFFSET(i)] = 0;
            _scan_pred_slow[i + CONFLICT_FREE_OFFSET(i)] = 0;
        }
    }
   __syncthreads();

    // upsweep for both pred_slow and pred fast
/*#pragma unroll
    for(int k = 0 ; k < UNROLL; k++) {
        _upsweep(_scan_pred_slow + k * BLOCKSIZE, tid, BLOCKSIZE);
    }
*/

    _upsweep<UNROLL>(_scan_pred_slow, tid, BLOCKSIZE);


    if(tid == 0) {
#pragma unroll
        for(int k = 1 ; k <= UNROLL; k++) {
            _scan_pred_slow[k*BLOCKSIZE - 1 + CONFLICT_FREE_OFFSET(k*BLOCKSIZE - 1)] = 0;
        }
    }

    __syncthreads();

    // downsweep time now
/*#pragma unroll
    for(int k = 0 ; k < UNROLL; k++) {
        _downsweep(_scan_pred_slow + k * BLOCKSIZE, tid, BLOCKSIZE);
    }
*/

    _downsweep<UNROLL>(_scan_pred_slow, tid, BLOCKSIZE);
    

    if(tid == 0) {
        _block_factor[0] = 0;
#pragma unroll
        for(int k = 1 ; k < UNROLL; k++) {
            int bo = k * BLOCKSIZE;
            _block_factor[k] = _block_factor[k-1] + _scan_pred_slow[bo - 1 + CONFLICT_FREE_OFFSET(bo - 1)] + _pred_slow[bo - 1 + CONFLICT_FREE_OFFSET(bo-1)];
        }
    }

    __syncthreads();

#pragma unroll
    for(int k = 0; k < UNROLL; k++) {
        int i = k * BLOCKSIZE + tid;
        if(i < N) {
            int* g_sp = scanned_predicate;
            g_sp[i] = _scan_pred_slow[i + CONFLICT_FREE_OFFSET(i)] + _block_factor[k];
        }
    }

}

template<int UNROLL>
__global__ void _scan_predicate_multi_block(const int* __restrict__ predicate, int* __restrict__ scanned_predicate, 
        int* __restrict__ block_predicate, 
        const int N, 
        const int BLOCKSIZE,
        const int ELEMENTS_PER_BLOCK) {

    __shared__ int _block_factor[UNROLL]; 
    extern __shared__ int _smem[];

    int* _pred_slow = (int*) _smem;
    int* _scan_pred_slow = _pred_slow + ELEMENTS_PER_BLOCK;

    int tid = threadIdx.x; //+ blockDim.x * blockIdx.x;
    const int* p_total = predicate + blockIdx.x * BLOCKSIZE * UNROLL;
    
    //int bankOffset = CONFLICT_FREE_OFFSET(tid);

    //int p_data = p_total[tid];
#pragma unroll
    for(int k = 0; k < UNROLL; k++) {
        int i = tid + k * BLOCKSIZE;
        int p_data = p_total[i];
        _pred_slow[i + CONFLICT_FREE_OFFSET(i)] = p_data;
        _scan_pred_slow[i + CONFLICT_FREE_OFFSET(i)] = p_data;
    }

   __syncthreads();

    // upsweep for both pred_slow and pred fast
/*#pragma unroll 
   for(int k = 0; k < UNROLL; k++) {
        _upsweep(_scan_pred_slow + k * BLOCKSIZE, tid, BLOCKSIZE);
    }*/

    _upsweep<UNROLL>(_scan_pred_slow, tid, BLOCKSIZE);

    __syncthreads();

    if(tid == 0) {
        int _total_sum = 0;
//#pragma unroll
        for(int k = 1; k<=UNROLL; k++) {
            _total_sum += _scan_pred_slow[k*BLOCKSIZE - 1 + CONFLICT_FREE_OFFSET(k * BLOCKSIZE - 1)];
        }
        block_predicate[blockIdx.x] = _total_sum;//_scan_pred_slow[(BLOCKSIZE-1)];
#pragma unroll
        for(int k = 1; k <= UNROLL; k++) {
            _scan_pred_slow[k*BLOCKSIZE - 1 + CONFLICT_FREE_OFFSET(k * BLOCKSIZE - 1)] = 0;
        }

        //_scan_pred_slow[BLOCKSIZE-1] = 0;
    }
    

    __syncthreads();
    

    _downsweep<UNROLL>(_scan_pred_slow, tid, BLOCKSIZE);

    // downsweep time now
    /*for(int k = 0 ; k < UNROLL; k++) {
        _downsweep(_scan_pred_slow + k * BLOCKSIZE, tid, BLOCKSIZE);
    }*/

    if(tid == 0) {
        _block_factor[0] = 0;
#pragma unroll
        for(int k = 1 ; k < UNROLL; k++) {
            int bo = k * BLOCKSIZE;
            _block_factor[k] = _block_factor[k-1] + _scan_pred_slow[bo - 1 + CONFLICT_FREE_OFFSET(bo - 1)] + _pred_slow[bo - 1 + CONFLICT_FREE_OFFSET(bo -1)];
        }
    }

    __syncthreads(); 

    int* g_sp = scanned_predicate + blockIdx.x * BLOCKSIZE * UNROLL;

#pragma unroll
    for(int k = 0 ; k < UNROLL; k++) {
        int i = tid + k * BLOCKSIZE;
        g_sp[i] = _scan_pred_slow[i + CONFLICT_FREE_OFFSET(i)] + _block_factor[k];
    }


    //g_sp[tid] = _scan_pred_slow[tid];


}

template<typename T>
__global__ void _set_predicate(const int* predicate, const int* scanned_predicate, 
        particle_system<T>* total, particle_system<T>* slow, particle_system<T>* fast, 
        const int N,
        partition* part) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid < N) {
        
        if(tid == 0) { 
            part->N_fast = predicate[N-1] + scanned_predicate[N-1];
            part->N_slow = N - part->N_fast;
#ifdef CUDA_DEBUG
            printf("part->N_fast = %i\n", part->N_fast);
            printf("part->N_slow = %i\n", part->N_slow);
#endif

            /*if(part->N_fast == 1) {
                part->N_fast = 0;
                part->N_slow = N;
            }*/
        }
        // !!!!!CAUTION!!!!
        // Do all threads read a consistent value of N_fast?
        // TODO: Probable BUG here!! Needs a lot of checking
        //Is this necessary?
        // !!!!CAUTION!!!!

        __syncthreads();

        
        if(predicate[tid]) {
            int scanned_predicate_tid = scanned_predicate[tid];
#ifdef CUDA_DEBUG
            printf("scanned_predicate_tid=%i\n", scanned_predicate[tid]);
            //printf("total->pos[tid].x=%f\n",total->pos[tid].x);
#endif
            fast->pos[scanned_predicate_tid] = total->pos[tid];
            fast->vel[scanned_predicate_tid] = total->vel[tid];
            fast->m[scanned_predicate_tid] = total->m[tid];
            fast->timestep[scanned_predicate_tid] = total->timestep[tid];
            fast->parent_id[scanned_predicate_tid] = tid;
        }
        
        if(!predicate[tid]) { 
            int scanned_predicate_tid = tid - scanned_predicate[tid];
            slow->pos[scanned_predicate_tid] = total->pos[tid];
            slow->vel[scanned_predicate_tid] = total->vel[tid];
            slow->m[scanned_predicate_tid] = total->m[tid];
            slow->timestep[scanned_predicate_tid] = total->timestep[tid];
            slow->parent_id[scanned_predicate_tid] = tid;
        }



        /*if(part->N_fast == 0) {
            slow->pos[part->N_slow-1] = total->pos[tid];
            slow->vel[part->N_slow-1] = total->vel[tid];
            slow->m[part->N_slow-1] = total->m[tid];
            slow->timestep[part->N_slow-1] = total->timestep[tid];
            slow->parent_id[part->N_slow-1] = tid;
        }*/
    }
}

template<int UNROLL>
__global__ void add(int *scanned_predicate, int *scanned_block_predicate, int BLOCKSIZE) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * BLOCKSIZE * UNROLL;
    for(int k = 0; k < UNROLL; k++) {
	    int i = threadID + k * BLOCKSIZE;
        scanned_predicate[blockOffset + i] += scanned_block_predicate[blockID];
    
    }
}

template<int UNROLL>
__global__ void add(int *output, int BLOCKSIZE, int *n1, int *n2, const int N) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * BLOCKSIZE * UNROLL;
    for(int k = 0 ; k < UNROLL; k++) {
	    int i = threadID + k * BLOCKSIZE;
        if(i < N)
            output[blockOffset + i] += n1[blockID] + n2[blockID];
    }
}

/* host side helpers to call the device side kernels*/


template<typename T, int BLOCKSIZE>
__host__ void get_predicate(T* total, 
        int* predicate, 
        const T pivot, 
        const int N) {
    dim3 block(BLOCKSIZE);
    dim3 grid((block.x + N - 1 ) / block.x );

    _get_predicate<T><<<grid,block>>>(total, predicate, pivot, N);
#ifdef CUDA_DEBUG
    gpuErrchk( cudaDeviceSynchronize() );
#else
    // cudaDeviceSynchronize();
#endif
}


template<typename T, int BLOCKSIZE>
__host__ void set_predicate(const int* predicate, const int* scanned_predicate, 
        particle_system<T>* total, particle_system<T>* slow, particle_system<T>* fast, 
        const int N, 
        partition* part) {
    dim3 block(BLOCKSIZE);
    dim3 grid((block.x + N - 1 ) / block.x );

    _set_predicate<T><<<grid,block>>>(predicate, scanned_predicate, total->d_ptr, slow->d_ptr, fast->d_ptr, N, part);
#ifdef CUDA_DEBUG
    gpuErrchk( cudaDeviceSynchronize() );
#else
    // cudaDeviceSynchronize();
#endif
}


/*this handles the non-even cases*/
template<int UNROLL>
__host__ void scan_predicate_small(int* predicate, int* scanned_predicate, const int N) {
    // TODO: Does setting the blocksize closest to the nearest power of 2 increase efficiency?
    // N is smaller than BLOCKSIZE
    // find power of 2 closest to N
    
    // bit tricks to find the power of 2 closest to N
    unsigned int v = N;
    
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    
    int ADJUSTED_BLOCKSIZE = max(v / UNROLL, 1);

    _scan_predicate_single_block<UNROLL><<<1, ADJUSTED_BLOCKSIZE , v* 2 * sizeof(int)>>>(predicate, scanned_predicate, N, v);
#ifdef CUDA_DEBUG
    gpuErrchk( cudaDeviceSynchronize() );
#else
        //cudaDeviceSynchronize();
#endif

}


template<int UNROLL>
__host__ void scan_predicate_large(int* predicate, int* scanned_predicate, const int N, int BLOCKSIZE) {
    int ELEMENTS_PER_BLOCK = BLOCKSIZE * UNROLL;
    int remainder = N % ELEMENTS_PER_BLOCK;
    //printf("BLOCKSIZE: %i, N: %i\n ", BLOCKSIZE, N);


    if(remainder == 0) {
        scan_predicate_large_ev<UNROLL>(predicate, scanned_predicate, N, BLOCKSIZE);
    } else {
        
        int modN = N - remainder;
        int *rem_predicate = predicate + modN;
        int *rem_scanned_predicate = scanned_predicate+modN;

        scan_predicate_large_ev<UNROLL>(predicate, scanned_predicate, modN, BLOCKSIZE);
        
        scan_predicate_small<UNROLL>(rem_predicate, rem_scanned_predicate, remainder);
        
        int ADJUSTED_BLOCKSIZE = (remainder + UNROLL - 1) / UNROLL;


        add<UNROLL><<<1,ADJUSTED_BLOCKSIZE>>>(rem_scanned_predicate, ADJUSTED_BLOCKSIZE, &(predicate[modN-1]), &(scanned_predicate[modN-1]), remainder);
#ifdef CUDA_DEBUG  
        gpuErrchk( cudaDeviceSynchronize() );
#else
        //   cudaDeviceSynchronize();
#endif
    }
}

template<int UNROLL>
__host__ void scan_predicate_large_ev(int* predicate, int* scanned_predicate, const int N, int BLOCKSIZE) {
    int ELEMENTS_PER_BLOCK = BLOCKSIZE * UNROLL;
    int num_blocks_needed = (N + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

    // malloc space for temp arrays
    int *scanned_block_predicate, *block_predicate;
    cudaMalloc(&scanned_block_predicate, sizeof(int) * num_blocks_needed);
    cudaMalloc(&block_predicate, sizeof(int) * num_blocks_needed);

    
    _scan_predicate_multi_block<UNROLL><<<num_blocks_needed, BLOCKSIZE, 2 * sizeof(int) * ELEMENTS_PER_BLOCK>>>(predicate, scanned_predicate, block_predicate, 
            N, BLOCKSIZE, ELEMENTS_PER_BLOCK);
    
#ifdef CUDA_DEBUG
    gpuErrchk( cudaDeviceSynchronize() );
#else
    //    cudaDeviceSynchronize();
#endif

    //printf("BLOCKSIZE: %i, N: %i\n ", BLOCKSIZE, N);

    if(num_blocks_needed > ELEMENTS_PER_BLOCK) {
        //printf("num_blocks_needed: %i calling scan_pred_large on block_predicate\n", num_blocks_needed);
        scan_predicate_large<UNROLL>(block_predicate, scanned_block_predicate, num_blocks_needed, BLOCKSIZE);
    } else {
        //printf("num_blocks_needed: %i calling scan_pred_small on block_predicate\n", num_blocks_needed);
        scan_predicate_small<UNROLL>(block_predicate, scanned_block_predicate, num_blocks_needed);
    }
#ifdef CUDA_DEBUG_PRINT_VAL    
    printf("DEBUG: Printing block_predicate, scanned_block_predicate\n");
    
    printf("block_predicate values:\n");
    debug_cuPrintArr<<<1,1>>>(block_predicate, num_blocks_needed);
    
    printf("scanned_block_predicate_values\n");
    debug_cuPrintArr<<<1,1>>>(scanned_block_predicate, num_blocks_needed);
#endif

    add<UNROLL><<<num_blocks_needed,BLOCKSIZE>>>(scanned_predicate, scanned_block_predicate, BLOCKSIZE);

#ifdef CUDA_DEBUG
    gpuErrchk( cudaDeviceSynchronize() );

#else
    //    cudaDeviceSynchronize();
#endif

    // free space of temp arrays
    cudaFree(scanned_block_predicate);
    cudaFree(block_predicate);
}


/*

*/

/* host side main calling function*/
template<typename T, int BLOCKSIZE, int UNROLL>
__host__ void split(particle_system<T>* __restrict__ total, 
        particle_system<T>* __restrict__  slow, particle_system<T>* __restrict__ fast,
        int* predicate, int* scanned_predicate, 
        const T pivot,
        const int N, 
        partition* part) {
    
#ifdef CUDA_DEBUG
    static_assert(UNROLL > 0, "UNROLLING FACTOR MUST BE GREATER THAN ZERO!");
#endif

    get_predicate<T,BLOCKSIZE>(total->timestep, predicate,pivot, N);
    int ELEMENTS_PER_BLOCK = UNROLL * BLOCKSIZE;

    if(N > ELEMENTS_PER_BLOCK) {
#ifdef CUDA_DEBUG
    printf("calling scan_predicate_large init N: %i, ELEMENTS_PER_BLOCK: %i, UNROLLING_FACTOR: %i\n", N, ELEMENTS_PER_BLOCK, UNROLL);
#endif
        scan_predicate_large<UNROLL>(predicate, scanned_predicate, N, BLOCKSIZE); 
    } else {
#ifdef CUDA_DEBUG
        printf("calling scan_predicate_small! init N: %i, ELEMENTS_PER_BLOCK: %i, UNROLLING_FACTOR: %i\n", N, ELEMENTS_PER_BLOCK, UNROLL);
#endif  
        scan_predicate_small<UNROLL>(predicate, scanned_predicate,N);

   }
#ifdef CUDA_DEBUG
    //debug_cuPrintVecArr<T><<<1,1>>>(total->d_ptr, N);
    //printf("Preicate:\n");
    //debug_cuPrintArr<<<1,1>>>(predicate,N);
    //printf("Scanned predicate:\n");
    //debug_cuPrintArr<<<1,1>>>(scanned_predicate,N);
#endif
    set_predicate<T,BLOCKSIZE>(predicate, scanned_predicate, total, slow, fast, N, part);
    //cudaDeviceSynchronize();

}
