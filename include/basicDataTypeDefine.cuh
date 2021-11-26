//
// Created by ascdc on 2021-11-22.
//

#ifndef NTOKEN_BASICDATATYPEDEFINE_CUH
#define NTOKEN_BASICDATATYPEDEFINE_CUH

#include <string>

using charType = uint32_t;
using partWordOfSpeechType = uint16_t;

template <typename T>
struct array{
    T * ptr;
    size_t size;
    size_t cap;
    array()
    :size(0), cap(16)
    {
        auto err = cudaMallocManaged(reinterpret_cast<void**>(&ptr), sizeof(T) * cap);
        if(err != cudaSuccess){
            throw __FILE__ + __LINE__ + __func__ + cudaGetErrorName(err);
        }
    }
    array(const T* rightPtr, const size_t count):
    size(count),
    cap(count){
        auto err = cudaMallocManaged(reinterpret_cast<void **>(&ptr), sizeof(T ) * count);
        if(err != cudaSuccess){
            throw __FILE__ + std::to_string(__LINE__) + __func__ + cudaGetErrorName(err);
        }
        size = count;
        cap = count;
    }
    array(size_t sizeIn, size_t capIn):
    size(sizeIn),
    cap(capIn)
    {
        auto err = cudaMallocManaged(reinterpret_cast<void**>(&ptr), cap * sizeof(T));
        if(err != cudaSuccess){
            throw __FILE__ + std::to_string(__LINE__) + __func__ + cudaGetErrorName(err);
        }
    }

    ~array(){
        auto err = cudaFree(reinterpret_cast<void**>(&ptr));
        if(err != cudaSuccess){
            throw __FILE__ + std::to_string(__LINE__) + __func__ + cudaGetErrorName(err);
        }
    }
    auto __device__ __host__ operator[](size_t id) -> T&{
        return  (ptr[id]);
    }

    auto __device__ __host__ operator[](size_t id) const -> T{
        return (ptr[id]);
    }
    auto operator=(const array &right) -> T&{
        if (right.size >= cap){
            T * tempPtr;
            auto err = cudaMallocManaged(reinterpret_cast<void**>(tempPtr), right.cap * sizeof(T));
            if(err != cudaSuccess){
                throw cudaGetErrorName(err);
            }
            cudaMemcpy(tempPtr, right.ptr, sizeof(T) * right.size);
            cudaFree(ptr);
            ptr = tempPtr;
            cap = right.cap;
            size = right.size;
        } else{
            cudaMemcpy(ptr, right.ptr, sizeof(T) * right.size);
            size = right.size;
        }


    }
    auto __device__ operator=(const array &right) -> T&{
        cudaMemcpy(ptr, right.ptr, sizeof(T) * right.size);
        size = right.size;
    }
};


#endif //NTOKEN_BASICDATATYPEDEFINE_CUH
