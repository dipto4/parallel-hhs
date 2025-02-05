#include<iostream>
#include<cstdlib>
#include<cassert>
#include "harness.h"
#include "../scan_optimized_1.cuh"

typedef float real_t;

template<int NSIZE, int BLOCKSIZE>
void test_split_single_block() {
    real_t *total;
    real_t *slow, *fast;
    real_t *d_slow, *d_fast, *d_total;
    int* d_predicate; int* d_scanned_predicate;

    partition* sizes = new partition();
    partition* d_sizes;

    const size_t N = NSIZE;

    // generate the random numbers first

    slow = new real_t[N];
    fast = new real_t[N];
    total = new real_t[N];
    
    //const real_t lo = 0.;
    const real_t hi = 10.0;
    
    for(int i = 0 ; i < N; i++) {
        float r = static_cast<float> (rand()) / static_cast<float>(RAND_MAX); 

        total[i] = hi * r; 
    }
    
    const real_t pivot = 5.0;

    // allocate resources on the GPU

    cudaMalloc(&d_slow, sizeof(real_t) * N);
    cudaMalloc(&d_fast, sizeof(real_t) * N);
    cudaMalloc(&d_total, sizeof(real_t) * N);
    cudaMalloc(&d_sizes, sizeof(partition));
    cudaMalloc(&d_predicate, sizeof(int) * N);
    cudaMalloc(&d_scanned_predicate, sizeof(int) * N);

    //for(int i = 0 ; i < N; i++) {
    //    std::cout<<total[i]<<std::endl;
   // }


    cudaMemcpy(d_total, total, sizeof(real_t) * N , cudaMemcpyHostToDevice);

    split<real_t, BLOCKSIZE>(d_total, d_slow, d_fast, d_predicate, d_scanned_predicate, pivot, N, d_sizes); 

    //scan_predicate<real_t, N><<<1,N>>>(d_total, d_slow, d_fast, pivot, N,d_sizes);
     

    cudaMemcpy(slow, d_slow, sizeof(real_t) * N , cudaMemcpyDeviceToHost);
    cudaMemcpy(fast, d_fast, sizeof(real_t) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(sizes, d_sizes, sizeof(partition), cudaMemcpyDeviceToHost);
    
    bool sizesEqual = (sizes->N_slow + sizes->N_fast) == N ;
    
    //std::cout<<"N_slow: "<<sizes->N_slow<<std::endl;
    //std::cout<<"N_fast: "<<sizes->N_fast<<std::endl;

    TEST_ASSERT("Total size check (slow + fast == total)", sizesEqual);
    
    int count_checkSlow = 0;
    for(int i = 0 ; i < sizes->N_slow; i++) {
        bool checkSlow = slow[i] <= pivot;
        if(checkSlow) count_checkSlow++;
    }

    TEST_ASSERT("count_checkSlow == sizes->N_slow. Slow subsystem correct", sizes->N_slow == count_checkSlow);

    int count_checkFast = 0;
    for(int i = 0 ; i < sizes->N_fast; i++) {
        bool checkFast = fast[i] > pivot;
        if(checkFast) count_checkFast++;
        //TEST_ASSERT("fast[i] > pivot, i = " + std::to_string(i), checkFast);
    }

    TEST_ASSERT("count_checkFast == sizes->N_fast. Fast subsystem correct", sizes->N_fast == count_checkFast);


    cudaFree(d_slow);
    cudaFree(d_fast);
    cudaFree(d_total);

    delete [] slow;
    delete [] fast;
    delete [] total;

}


template<int NSIZE, int BLOCKSIZE>
void test_split_multi_block() {
    real_t *total;
    real_t *slow, *fast;
    real_t *d_slow, *d_fast, *d_total;
    int* d_predicate; int* d_scanned_predicate;

    partition* sizes = new partition();
    partition* d_sizes;

    const size_t N = NSIZE;

    // generate the random numbers first

    slow = new real_t[N];
    fast = new real_t[N];
    total = new real_t[N];
    
    //const real_t lo = 0.;
    const real_t hi = 10.0;
    
    for(int i = 0 ; i < N; i++) {
        float r = static_cast<float> (rand()) / static_cast<float>(RAND_MAX); 

        total[i] = hi * r; 
    }
    
    const real_t pivot = 5.0;

    // allocate resources on the GPU

    cudaMalloc(&d_slow, sizeof(real_t) * N);
    cudaMalloc(&d_fast, sizeof(real_t) * N);
    cudaMalloc(&d_total, sizeof(real_t) * N);
    cudaMalloc(&d_sizes, sizeof(partition));
    cudaMalloc(&d_predicate, sizeof(int) * N);
    cudaMalloc(&d_scanned_predicate, sizeof(int) * N);

    //for(int i = 0 ; i < N; i++) {
    //    std::cout<<total[i]<<std::endl;
   // }


    cudaMemcpy(d_total, total, sizeof(real_t) * N , cudaMemcpyHostToDevice);

    split<real_t, BLOCKSIZE>(d_total, d_slow, d_fast, d_predicate, d_scanned_predicate, pivot, N, d_sizes); 

    //scan_predicate<real_t, N><<<1,N>>>(d_total, d_slow, d_fast, pivot, N,d_sizes);
     

    cudaMemcpy(slow, d_slow, sizeof(real_t) * N , cudaMemcpyDeviceToHost);
    cudaMemcpy(fast, d_fast, sizeof(real_t) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(sizes, d_sizes, sizeof(partition), cudaMemcpyDeviceToHost);
    
    bool sizesEqual = (sizes->N_slow + sizes->N_fast) == N ;
    
    //print sizes
    //std::cout<<"sizes->N_slow: "<<sizes->N_slow<<std::endl;
    //std::cout<<"sizes->N_fast: "<<sizes->N_fast<<std::endl;
    
    TEST_ASSERT("Total size check (slow + fast == total)", sizesEqual);

    int count_checkSlow = 0;
    for(int i = 0 ; i < sizes->N_slow; i++) {
        bool checkSlow = slow[i] <= pivot;
        if(checkSlow) count_checkSlow++;
    }
    //std::cout<<"count_checkSlow: "<<count_checkSlow<<std::endl;
    TEST_ASSERT("count_checkSlow == sizes->N_slow. Slow subsystem correct", sizes->N_slow == count_checkSlow);

    int count_checkFast = 0;
    for(int i = 0 ; i < sizes->N_fast; i++) {
        bool checkFast = fast[i] > pivot;
        if(checkFast) count_checkFast++;
        //TEST_ASSERT("fast[i] > pivot, i = " + std::to_string(i), checkFast);
    }

    //std::cout<<"count_checkFast: "<<count_checkFast<<std::endl;
    TEST_ASSERT("count_checkFast == sizes->N_fast. Fast subsystem correct", sizes->N_fast == count_checkFast);


    cudaFree(d_slow);
    cudaFree(d_fast);
    cudaFree(d_total);

    delete [] slow;
    delete [] fast;
    delete [] total;

}


int main() {
    std::cout<<"TEST: size and condition for the splitting routine [single block] NSIZE : 512, BLOCKSIZE : 512"<<std::endl;
    test_split_single_block<512, 512>();

    std::cout<<"TEST: size and condition for the splitting routine [single block] NSIZE : 512, BLOCKSIZE : 1024"<<std::endl;
    test_split_single_block<512, 1024>();
    
    std::cout<<"TEST: size and condition for the splitting routine [single block] NSIZE : 1023, BLOCKSIZE : 1024"<<std::endl;
    test_split_single_block<1023, 1024>();

    std::cout<<"TEST: size and condition for the splitting routine [multi block] NSIZE : 2048 BLOCKSIZE : 512"<<std::endl;
    test_split_multi_block<2048,512>();

    std::cout<<"TEST: size and condition for the splitting routine [multi block] NSIZE : 2151 BLOCKSIZE : 512"<<std::endl;
    test_split_multi_block<2151,512>();

    std::cout<<"TEST: size and condition for the splitting routine [multi block] NSIZE : 524288 BLOCKSIZE : 128"<<std::endl;
    test_split_multi_block<524288,128>();

    std::cout<<"TEST: size and condition for the splitting routine [multi block] NSIZE : 588021 BLOCKSIZE : 128"<<std::endl;
    test_split_multi_block<588021,128>();

    std::cout<<"TEST: size and condition for the splitting routine [multi block] NSIZE : 1048576 BLOCKSIZE : 128"<<std::endl;
    test_split_multi_block<1048576,128>();

    std::cout<<"TEST: size and condition for the splitting routine [multi block] NSIZE : 2097152 BLOCKSIZE : 128"<<std::endl;
    test_split_multi_block<2097152,128>();



}
