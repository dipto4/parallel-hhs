#include<iostream>
#include<cstdlib>
#include<cassert>
#include<vector>
#include<fstream>
#include<ctime>

#include "harness.h"
#include "../scan_optimized_2.cuh"
#include "../cpu_scan.h"

constexpr int UNROLL = 1;

/*template<int p>
  struct createIC {
  constexpr createIC() : arr() {
  arr[0] = 1;
  for(int i = 1; i <= p ; i++) {
  arr[i] = arr[i-1] * 2;
  }
  }
  int arr[p+1];
  };*/


typedef float real_t;

template<int NSIZE, int BLOCKSIZE, int UNROLL>
void perf_split_gpu_warm() {
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


    cudaMemcpy(d_total, total, sizeof(real_t) * N , cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    split<real_t, BLOCKSIZE, UNROLL>(d_total, d_slow, d_fast, d_predicate, d_scanned_predicate, pivot, N, d_sizes); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0;

    cudaEventElapsedTime(&time, start, stop);

    std::cout<<"GPU elapsed time: "<<time<<std::endl;

    cudaMemcpy(slow, d_slow, sizeof(real_t) * N , cudaMemcpyDeviceToHost);
    cudaMemcpy(fast, d_fast, sizeof(real_t) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(sizes, d_sizes, sizeof(partition), cudaMemcpyDeviceToHost);


    cudaFree(d_slow);
    cudaFree(d_fast);
    cudaFree(d_total);

    delete [] slow;
    delete [] fast;
    delete [] total;

}

template<int NSIZE, int BLOCKSIZE, int UNROLL, int NEVENTS>
void perf_split_gpu(std::ofstream &outfile) {
    outfile<<NSIZE<<"\t";
    //float avg_time = 0;
    for(int event = 0 ; event < NEVENTS; event++) {
        srand(time(0));
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


        cudaMemcpy(d_total, total, sizeof(real_t) * N , cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        split<real_t, BLOCKSIZE, UNROLL>(d_total, d_slow, d_fast, d_predicate, d_scanned_predicate, pivot, N, d_sizes); 
        //cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time = 0;

        cudaEventElapsedTime(&time, start, stop);
        outfile<<time<<"\t";
        //avg_time += time;

        cudaMemcpy(slow, d_slow, sizeof(real_t) * N , cudaMemcpyDeviceToHost);
        cudaMemcpy(fast, d_fast, sizeof(real_t) * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(sizes, d_sizes, sizeof(partition), cudaMemcpyDeviceToHost);


        cudaFree(d_slow);
        cudaFree(d_fast);
        cudaFree(d_total);

        delete [] slow;
        delete [] fast;
        delete [] total;

    }
    outfile<<std::endl;


}






int main() {

    std::cout<<"GPU warmup routine! IGNORE THIS CASE"<<std::endl;
    perf_split_gpu_warm<524288,128,1>();
    /* 
       constexpr auto IC_N = createIC<22>();

       std::ofstream outCPUFile("cpu_split.txt");
       std::ofstream outGPUFile("gpu_split_0.txt");

       for(const auto &N : IC_N.arr) {
       perf_split_gpu<N,128>(outGPUFile);
       perf_split_cpu<N>(outCPUFile);
       }

     */
    std::ofstream outGPUFile("gpu_split_unroll_1.txt");
    

    perf_split_gpu<1024,128, UNROLL,10>(outGPUFile);

    perf_split_gpu<2048,128, UNROLL,10>(outGPUFile);

    perf_split_gpu<4096,128, UNROLL,10>(outGPUFile);

    perf_split_gpu<8192,128, UNROLL ,10>(outGPUFile);

    perf_split_gpu<16384,128, UNROLL,10>(outGPUFile);

    perf_split_gpu<32768,128, UNROLL,10>(outGPUFile);

    perf_split_gpu<65536,128, UNROLL,10>(outGPUFile);

    perf_split_gpu<131072,128, UNROLL,10>(outGPUFile);

    perf_split_gpu<262144,128, UNROLL,10>(outGPUFile);

    perf_split_gpu<524288,128, UNROLL,10>(outGPUFile);

    perf_split_gpu<524288*2,128,UNROLL, 10>(outGPUFile);

    perf_split_gpu<524288*2*2,128, UNROLL, 10>(outGPUFile);

    perf_split_gpu<524288*2*2*2,128, UNROLL,10>(outGPUFile);





    outGPUFile.close();

}
