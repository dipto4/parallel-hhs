/* first optimization: remove shared memory bank conflicts */

#pragma once
#include<cuda_runtime.h>
#include<cstdio>
#include "utils.cuh"

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
//#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)
#define CONFLICT_FREE_OFFSET(n)((n) >>NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))



struct partition {
    int N_slow;
    int N_fast;
};




__host__ void scan_predicate_small(int* predicate, int* scanned_predicate, const int N);
__host__ void scan_predicate_large(int* predicate, int* scanned_predicate, const int N, int BLOCKSIZE);
__host__ void scan_predicate_large_ev(int* predicate, int* scanned_predicate, const int N, int BLOCKSIZE);


/* FOR DEBUGGING PURPOSES ONLY*/

#ifdef CUDA_DEBUG
template<typename T>
__global__ void debug_cuPrintArr(T* arr, const int size) {
    for(int i = 0 ; i < size; i++) {
        printf("i = %i, val = %i\n", i, arr[i] );
    }
}

#endif

__device__ void __forceinline__ _upsweep(int* _arr_slow, int tid, int BLOCKSIZE) {
#pragma unroll
    for(int stride = 1; stride<BLOCKSIZE; stride<<=1) {
        int idx = (tid + 1) * (stride << 1) - 1;
        if(idx < BLOCKSIZE) {
            _arr_slow[idx + CONFLICT_FREE_OFFSET(idx)] += _arr_slow[(idx-stride) + CONFLICT_FREE_OFFSET(idx-stride)];
        }
        __syncthreads();
    }
}

__device__ void __forceinline__ _downsweep(int* _arr_slow, int tid, int BLOCKSIZE) {
#pragma unroll
    for(int stride = BLOCKSIZE / 2; stride > 0; stride>>=1) {
        int idx = (tid + 1) * (stride << 1) - 1;
        
        if(idx < BLOCKSIZE) {
            int _tmp_s = _arr_slow[idx + CONFLICT_FREE_OFFSET(idx)];

            _arr_slow[idx + CONFLICT_FREE_OFFSET(idx)] += _arr_slow[(idx-stride) + CONFLICT_FREE_OFFSET(idx-stride)];

            _arr_slow[(idx-stride) + CONFLICT_FREE_OFFSET(idx-stride)] = _tmp_s;
            
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

__global__ void _scan_predicate_single_block(const int* __restrict__ predicate, int* __restrict__ scanned_predicate, const int N) {
    extern __shared__ int _smem[];

    int* _pred_slow = (int*) _smem;
    int* _scan_pred_slow = _pred_slow + N;
    //__shared__ int _scan_pred_slow[BLOCKSIZE];

    // perform an exclusive Belloch scan
    // process:
    // upsweep -> set final element to zero -> downsweep
    int BLOCKSIZE = blockDim.x;

    int tid = threadIdx.x; //+ blockDim.x * blockIdx.x;
    const int* p_total = predicate;//blockIdx.x * blockDim.x;
    
    int bankOffset = CONFLICT_FREE_OFFSET(tid);


    if(tid < N) { 
        int p_data = p_total[tid];
    
        _pred_slow[tid + bankOffset] = p_data;
        _scan_pred_slow[tid + bankOffset] = p_data;
    } else {
        _pred_slow[tid + bankOffset] = 0;
        _scan_pred_slow[tid + bankOffset] = 0;
    }
/*
*/
    __syncthreads();

    // upsweep for both pred_slow and pred fast

    _upsweep(_scan_pred_slow, tid, BLOCKSIZE);
    
    if(tid == 0) {
        _scan_pred_slow[BLOCKSIZE-1 + CONFLICT_FREE_OFFSET(BLOCKSIZE-1)] = 0;
    }
    

    __syncthreads();

    // downsweep time now
    _downsweep(_scan_pred_slow, tid, BLOCKSIZE);
    
    if(tid < N) {
        int* g_sp = scanned_predicate;
        g_sp[tid] = _scan_pred_slow[tid + bankOffset];
    }
}


__global__ void _scan_predicate_multi_block(const int* __restrict__ predicate, int* __restrict__ scanned_predicate, 
        int* __restrict__ block_predicate, 
        const int N, 
        const int BLOCKSIZE) {
    //__shared__ int _pred_slow[BLOCKSIZE];
    //__shared__ int _scan_pred_slow[BLOCKSIZE];
    
    //int BLOCKSIZE = blockDim.x;

    extern __shared__ int _smem[];

    int* _pred_slow = (int*) _smem;
    int* _scan_pred_slow = _pred_slow + BLOCKSIZE;

    // perform an exclusive Belloch scan
    // process:
    // upsweep -> set final element to zero -> downsweep

    int tid = threadIdx.x; //+ blockDim.x * blockIdx.x;
    const int* p_total = predicate + blockIdx.x * blockDim.x;
    
    int bankOffset = CONFLICT_FREE_OFFSET(tid);

    int p_data = p_total[tid];
    

    _pred_slow[tid + bankOffset] = p_data;
    _scan_pred_slow[tid + bankOffset] = p_data;
/*
*/
    __syncthreads();

    // upsweep for both pred_slow and pred fast

    _upsweep(_scan_pred_slow, tid, BLOCKSIZE);
    
    __syncthreads();

    if(tid == 0) {
        block_predicate[blockIdx.x] = _scan_pred_slow[(BLOCKSIZE-1) + CONFLICT_FREE_OFFSET(BLOCKSIZE-1)];
        _scan_pred_slow[BLOCKSIZE-1 + CONFLICT_FREE_OFFSET(BLOCKSIZE-1)] = 0;
    }
    

    __syncthreads();

    // downsweep time now
    _downsweep(_scan_pred_slow, tid, BLOCKSIZE);
    
    int* g_sp = scanned_predicate + blockIdx.x * blockDim.x;
    g_sp[tid] = _scan_pred_slow[tid + bankOffset];


}

template<typename T>
__global__ void _set_predicate(const int* predicate, const int* scanned_predicate, 
        const T* __restrict__ total, T* __restrict__ slow, T* __restrict__ fast, 
        const int N, 
        partition* part) {
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid < N) {
        
        if(tid == 0) { 
            part->N_slow = predicate[N-1] + scanned_predicate[N-1];
            part->N_fast = N - part->N_slow;
        }

        if(predicate[tid]) slow[scanned_predicate[tid]] = total[tid];
        if(!predicate[tid]) fast[tid - scanned_predicate[tid]] = total[tid];
    }
}


__global__ void add(int *scanned_predicate, int *scanned_block_predicate, int BLOCKSIZE) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * BLOCKSIZE;

	scanned_predicate[blockOffset + threadID] += scanned_block_predicate[blockID];
}

__global__ void add(int *output, int length, int *n1, int *n2) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n1[blockID] + n2[blockID];
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
        const T* total, T* slow, T* fast, 
        const int N, 
        partition* part) {
    dim3 block(BLOCKSIZE);
    dim3 grid((block.x + N - 1 ) / block.x );

    _set_predicate<T><<<grid,block>>>(predicate, scanned_predicate, total, slow, fast, N, part);
#ifdef CUDA_DEBUG
    gpuErrchk( cudaDeviceSynchronize() );
#else
    // cudaDeviceSynchronize();
#endif
}


/*this handles the non-even cases*/
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

    _scan_predicate_single_block<<<1, v , v* 2 * sizeof(int)>>>(predicate, scanned_predicate, v);
#ifdef CUDA_DEBUG
    gpuErrchk( cudaDeviceSynchronize() );
#else
        //cudaDeviceSynchronize();
#endif

}


__host__ void scan_predicate_large(int* predicate, int* scanned_predicate, const int N, int BLOCKSIZE) {
    int remainder = N % BLOCKSIZE;
    //printf("BLOCKSIZE: %i, N: %i\n ", BLOCKSIZE, N);


    if(remainder == 0) {
        scan_predicate_large_ev(predicate, scanned_predicate, N, BLOCKSIZE);
    } else {
        
        int modN = N - remainder;
        int *rem_predicate = predicate + modN;
        int *rem_scanned_predicate = scanned_predicate+modN;

        scan_predicate_large_ev(predicate, scanned_predicate, modN, BLOCKSIZE);
        
        scan_predicate_small(rem_predicate, rem_scanned_predicate, remainder);
        
        add<<<1,remainder>>>(rem_scanned_predicate, remainder, &(predicate[modN-1]), &(scanned_predicate[modN-1]));
#ifdef CUDA_DEBUG  
        gpuErrchk( cudaDeviceSynchronize() );
#else
        //   cudaDeviceSynchronize();
#endif
    }
}


__host__ void scan_predicate_large_ev(int* predicate, int* scanned_predicate, const int N, int BLOCKSIZE) {
    int num_blocks_needed = N / BLOCKSIZE;

    // malloc space for temp arrays
    int *scanned_block_predicate, *block_predicate;
    cudaMalloc(&scanned_block_predicate, sizeof(int) * num_blocks_needed);
    cudaMalloc(&block_predicate, sizeof(int) * num_blocks_needed);

    
    _scan_predicate_multi_block<<<num_blocks_needed, BLOCKSIZE, 2 * sizeof(int) * BLOCKSIZE>>>(predicate, scanned_predicate, block_predicate, N, BLOCKSIZE);
    
#ifdef CUDA_DEBUG
    gpuErrchk( cudaDeviceSynchronize() );
#else
    //    cudaDeviceSynchronize();
#endif

    //printf("BLOCKSIZE: %i, N: %i\n ", BLOCKSIZE, N);

    if(num_blocks_needed > BLOCKSIZE) {
        scan_predicate_large(block_predicate, scanned_block_predicate, num_blocks_needed, BLOCKSIZE);
    } else {
        scan_predicate_small(block_predicate, scanned_block_predicate, num_blocks_needed);
    }
#ifdef CUDA_DEBUG    
    printf("DEBUG: Printing block_predicate, scanned_block_predicate\n");
    
    printf("block_predicate values:\n");
    debug_cuPrintArr<<<1,1>>>(block_predicate, num_blocks_needed);
    
    printf("scanned_block_predicate_values\n");
    debug_cuPrintArr<<<1,1>>>(scanned_block_predicate, num_blocks_needed);
#endif

    add<<<num_blocks_needed,BLOCKSIZE>>>(scanned_predicate, scanned_block_predicate, BLOCKSIZE);

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
template<typename T, int BLOCKSIZE>
__host__ void split(T* total, T* slow, T* fast,
        int* predicate, int* scanned_predicate, 
        const T pivot,
        const int N,
        partition* part) {

    get_predicate<T,BLOCKSIZE>(total, predicate,pivot, N);
    
    if(N > BLOCKSIZE) {
        scan_predicate_large(predicate, scanned_predicate, N, BLOCKSIZE); 
    } else {
        scan_predicate_small(predicate, scanned_predicate, N);

   }
    
    set_predicate<T,BLOCKSIZE>(predicate, scanned_predicate, total, slow, fast, N, part);
    cudaDeviceSynchronize();

}
