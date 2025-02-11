#include<cuda_runtime.h>
#include "../kick.cuh"
#include "../vec.cuh"
#include "harness.h"

typedef float real_t;

template<typename T>
void get_nbforce_cpu(vec3<T>* pos, vec3<T>* acc, T* m, const int N) {
    for(int i = 0; i < N; i++) {
        //vec3<T> acc(0.0,0.0,0.0);
        for(int j = 0 ; j < N; j++) {
            if(i == j) continue;
            vec3<T> dx = pos[i] - pos[j];
            T d2 = dx.norm2();
            T d = sqrt(d2);
            T one_over_d3 = 1.0 / (d2 * d);
            vec3<T> ai = dx * (-m[j] * one_over_d3) ;
            acc[i] = acc[i] + ai;
        }
    }

}

template<typename T, int NSIZE>
void nbforce_cpu() {
        const size_t N = NSIZE;

        // generate the random numbers first

        vec3<T>* pos; vec3<T>* acc; T* m;

        pos = new vec3<T>[N];
        acc = new vec3<T>[N];
        m = new T[N];

        //const real_t lo = 0.;

        for(int i = 0 ; i < N; i++) {
            T r1 = static_cast<T> (rand()) / static_cast<T>(RAND_MAX); 
            T r2 = static_cast<T> (rand()) / static_cast<T>(RAND_MAX); 
            T r3 = static_cast<T> (rand()) / static_cast<T>(RAND_MAX); 
            
            pos[i].x = r1;
            pos[i].y = r2;
            pos[i].z = r3;
            m[i] = 1.0;
        }
        
        
        get_nbforce_cpu(pos, acc, m, N);

        for(int i = 0 ; i < N; i++) {
            printf("%.16e %.16e %.16e\n", acc[i].x, acc[i].y, acc[i].z);
        }


        delete [] pos;
        delete [] acc;
        delete [] m ;
        
}

template<typename T, int NSIZE>
void nbforce_gpu() {
        const size_t N = NSIZE;

        // generate the random numbers first

        vec3<T>* c_pos; vec3<T>* c_acc; T* c_m;
        vec3<T>* cc_acc;

        c_pos = new vec3<T>[N];
        c_acc = new vec3<T>[N];
        cc_acc = new vec3<T>[N];
        c_m = new T[N];

        //const real_t lo = 0.;

        for(int i = 0 ; i < N; i++) {
            T r1 = static_cast<T> (rand()) / static_cast<T>(RAND_MAX); 
            T r2 = static_cast<T> (rand()) / static_cast<T>(RAND_MAX); 
            T r3 = static_cast<T> (rand()) / static_cast<T>(RAND_MAX); 
            
            c_pos[i].x = r1;
            c_pos[i].y = r2;
            c_pos[i].z = r3;
            c_m[i] = 1.0;
        }
        
        vec3<T>* d_pos; vec3<T>* d_vel; vec3<T>* d_acc; T* d_m;

        cudaMalloc(&d_pos, N * sizeof(vec3<T>));
        cudaMalloc(&d_vel, N * sizeof(vec3<T>));
        cudaMalloc(&d_acc, N * sizeof(vec3<T>));
        cudaMalloc(&d_m, N * sizeof(T));

        cudaMemcpy(d_pos, c_pos, N * sizeof(vec3<T>), cudaMemcpyHostToDevice);
        cudaMemcpy(d_m, c_m, N * sizeof(T), cudaMemcpyHostToDevice);
        
        dim3 block(32);
        dim3 grid((block.x+N-1)/block.x);

        _kick_slow<T,32><<<grid,block>>>(d_pos,d_vel,d_acc, d_m,0.5, 0.5, N);
        
        cudaMemcpy(c_acc, d_acc, N * sizeof(vec3<T>), cudaMemcpyDeviceToHost);

        get_nbforce_cpu(c_pos, cc_acc, c_m, N);

        for(int i = 0 ; i < N; i++) {
            printf("%.16e %.16e %.16e %.16e %.16e %.16e \n", cc_acc[i].x ,c_acc[i].x, cc_acc[i].y, c_acc[i].y, cc_acc[i].z,c_acc[i].z);
        }
        

        cudaFree(d_pos); cudaFree(d_vel); cudaFree(d_acc); cudaFree(d_m);

        delete [] c_pos;
        delete [] c_acc;
        delete [] c_m ;
        
}



int main() {

    nbforce_gpu<real_t, 1024>();

}
