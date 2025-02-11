#pragma once

//#include "../df64.h"

template<typename T> 
struct vec3 {
    T x, y, z;
    
    __device__ __host__ vec3() : x(0.0), y(0.0), z(0.0) {}

    __device__ __host__ vec3(const T& xi, const T& yi, const T& zi) : x(xi) , y(yi), z(zi) {}

    __device__ __host__ vec3(const vec3& v) : x(v.x), y(v.y), z(v.z) {}
    
    __device__ __host__ vec3& operator=(const vec3& v) {
        if (this != &v) {
            x = v.x;
            y = v.y;
            z = v.z;
        }
        return *this;
    }

    __device__ __host__ vec3 operator+(vec3 const& v) const {
        return vec3(v.x+x, v.y+y, v.z+z);
    }
     __device__ __host__ vec3 operator-(vec3 const& v) const {
        return vec3(x-v.x, y-v.y, z-v.z);
    }
    
    __device__ __host__ T operator*(vec3 const& v) const {
        return v.x * x + v.y * y + v.z * z;
    }

    __device__ __host__ vec3 operator*(T const& v) const {
        return vec3(x * v , y * v , z * v);
    }
    

    __device__ __host__ vec3 operator-() const {
        return vec3(-x,-y,-z);
    }

    __device__ __host__ inline T norm2() {
        return x*x + y*y + z*z;
    }

    __device__ __host__ inline T norm() {
        return sqrt(norm2());
    }
    

};

