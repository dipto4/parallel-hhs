#include<iostream>
#include<cstdlib>
#include<cassert>
#include "harness.h"
#include "../scan_single.cuh"

typedef float real_t;

void test_split() {
    real_t *total;
    real_t *slow, *fast;
    real_t *d_slow, *d_fast, *d_total;
    
    partition* sizes = new partition();
    partition* d_sizes;

    const size_t N = 1024;

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
    

    //for(int i = 0 ; i < N; i++) {
    //    std::cout<<total[i]<<std::endl;
   // }


    cudaMemcpy(d_total, total, sizeof(real_t) * N , cudaMemcpyHostToDevice);

    scan_predicate<real_t, N><<<1,N>>>(d_total, d_slow, d_fast, pivot, N,d_sizes);
     

    cudaMemcpy(slow, d_slow, sizeof(real_t) * N , cudaMemcpyDeviceToHost);
    cudaMemcpy(fast, d_fast, sizeof(real_t) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(sizes, d_sizes, sizeof(partition), cudaMemcpyDeviceToHost);
    
    bool sizesEqual = (sizes->N_slow + sizes->N_fast) == N ;

    TEST_ASSERT("Total size check (slow + fast == total)", sizesEqual);
    
    for(int i = 0 ; i < sizes->N_slow; i++) {
        bool checkSlow = slow[i] <= pivot;
        TEST_ASSERT("slow[i] <= pivot , i = " + std::to_string(i), checkSlow);
    }

    for(int i = 0 ; i < sizes->N_fast; i++) {
        bool checkFast = fast[i] > pivot;
        TEST_ASSERT("fast[i] > pivot, i = " + std::to_string(i), checkFast);
    }



    cudaFree(d_slow);
    cudaFree(d_fast);
    cudaFree(d_total);

    delete [] slow;
    delete [] fast;
    delete [] total;

}


int main() {
    std::cout<<"TEST: size and condition for the splitting routine"<<std::endl;
    
    test_split();


}
