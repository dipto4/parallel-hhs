#pragma once
#include<cuda_runtime.h>

struct partition {
    int N_slow;
    int N_fast;
};


template<int BLOCKSIZE>
__device__ void __forceinline__ _upsweep(int* _arr_slow, int tid) {
    for(int stride = 1; stride<BLOCKSIZE; stride<<=1) {
        int idx = (tid + 1) * (stride << 1) - 1;
        if(idx < BLOCKSIZE) {
            _arr_slow[idx] += _arr_slow[idx-stride];
        }
        __syncthreads();
    }
}

template<int BLOCKSIZE>
__device__ void __forceinline__ _downsweep(int* _arr_slow, int tid) {
    for(int stride = BLOCKSIZE / 2; stride > 0; stride>>=1) {
        int idx = (tid + 1) * (stride << 1) - 1;
        
        if(idx < BLOCKSIZE) {
            int _tmp_s = _arr_slow[idx];
            //int _tmp_f = _arr_fast[idx];

            _arr_slow[idx] += _arr_slow[idx-stride];

            _arr_slow[idx-stride] = _tmp_s;
            
        }
        __syncthreads();
    }
}

//using the stream compaction trick to load into slow and fast arrays
template<typename T, int BLOCKSIZE>
__global__ void scan_predicate(T* __restrict__ total, T* __restrict__ slow, T* __restrict__ fast, const T pivot, const int N, partition* part) {
    __shared__ int _pred_slow[BLOCKSIZE];
    __shared__ int _scan_pred_slow[BLOCKSIZE];

    __shared__ int _pred_fast[BLOCKSIZE];
    __shared__ int _scan_pred_fast[BLOCKSIZE];

    // perform an exclusive Belloch scan
    // process:
    // upsweep -> set final element to zero -> downsweep

    int tid = threadIdx.x; //+ blockDim.x * blockIdx.x;
    T* g_total = total + blockIdx.x * blockDim.x;
    //T* g_slow = slow + blockIdx.x 


    T g_data = g_total[tid];
    

    _pred_slow[tid] = g_data <= pivot;
    _scan_pred_slow[tid] = g_data <= pivot;//_pred_slow[tid];
/*
    _pred_fast[tid] = g_data > pivot;
    _scan_pred_fast[tid] = g_data > pivot;//_pred_fast[tid];
*/
    __syncthreads();

    // upsweep for both pred_slow and pred fast

    _upsweep<BLOCKSIZE>(_scan_pred_slow, tid);
    
    if(tid == 0) {
        part->N_slow = _scan_pred_slow[BLOCKSIZE-1] ;
        part->N_fast = N - _scan_pred_slow[BLOCKSIZE-1];//_scan_pred_fast[BLOCKSIZE-1] ;
    
        _scan_pred_slow[BLOCKSIZE-1] = 0;
        //_scan_pred_fast[BLOCKSIZE-1] = 0;
    }
    

    __syncthreads();

    // downsweep time now
    _downsweep<BLOCKSIZE>(_scan_pred_slow, tid);
    
    

    if(_pred_slow[tid]) slow[_scan_pred_slow[tid]] = g_total[tid];
    if(!_pred_slow[tid]) fast[tid - _scan_pred_slow[tid]] = g_total[tid];




}
