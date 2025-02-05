// reproduction of the scan routine used in HHS integrators
// for performance testing purposes only

#pragma once

#include<cstdio>



#define SWAP(a,b,c) {c t;t=(a);(a)=(b);(b)=t;}


struct partition_c {
    int N_slow;
    int N_fast;
};



template<typename T>
void cpu_split(T* total, T*  slow, T* fast, T pivot, partition_c* sizes,const int N) {
    T *left, *right;
    int i = 0;
    left = total;
    right = &total[N-1];

    while(1) {
        if(i > N) {
            printf("catastrophic failure! \n");
            exit(1);
        }
        i++;
        while(*left <= pivot && left < right) left++;
        while(*right > pivot && left < right) right--;
        

        if(left < right) {
            SWAP(*left, *right, T);
        } else {
            break;
        }

    }


    if(*left <= pivot) left++;
    sizes->N_slow = &total[N-1] - left + 1;
    sizes->N_fast = (left - total);

}


