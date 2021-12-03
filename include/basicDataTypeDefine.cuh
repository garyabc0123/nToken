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


    array() {
        ptr = nullptr;
        size = 0;
        cap = 0;
    }
    array(const T* rightPtr, const size_t count){
        this->size = count;
        this->cap = count;
        auto err = cudaMallocManaged(reinterpret_cast<void **>(&(this->ptr)), sizeof(T ) * count);
        if(err != cudaSuccess){
            throw __FILE__ + std::to_string(__LINE__) + __func__ + cudaGetErrorName(err);
        }

    }
    array(size_t sizeIn, size_t capIn){
        this->size = sizeIn;
        this->cap = capIn;
        auto err = cudaMallocManaged(reinterpret_cast<void**>(&(this->ptr)), cap * sizeof(T));
        if(err != cudaSuccess){
            throw __FILE__ + std::to_string(__LINE__) + __func__ + cudaGetErrorName(err);
        }
    }

    array(array const &right){
        this->size = right.size;
        this->cap = right.cap;
        auto err = cudaMallocManaged(reinterpret_cast<void**>(&(this->ptr)), right.cap * sizeof(T));
        if(err != cudaSuccess){
            throw __FILE__ + std::to_string(__LINE__) + __func__ + cudaGetErrorName(err);
        }
        memcpy(this->ptr, right.ptr, right.size);
    }

    ~array(){
        if(this->cap == 0){
            return;
        }
        auto err = cudaFree(this->ptr);
        if(err != cudaSuccess){
            throw __FILE__ + std::to_string(__LINE__) + __func__ + cudaGetErrorName(err);
        }
        this->cap = 0;
        this->size = 0;
    }
    auto __device__ __host__ operator[](size_t id) -> T&{
        return  (this->ptr[id]);
    }

    auto __device__ __host__ operator[](size_t id) const -> T{
        return (this->ptr[id]);
    }
    auto __host__ operator=( array const &right) {
        if(right.size == 0){
            return ;
        }
        if (right.size >= cap){
            T * tempPtr;
            auto err = cudaMallocManaged(reinterpret_cast<void**>(&(tempPtr)), right.cap * sizeof(T));
            if(err != cudaSuccess){
                throw cudaGetErrorName(err);
            }
            memcpy(tempPtr, right.ptr, sizeof(T) * right.size);
            cudaFree(ptr);
            this->ptr = tempPtr;
            this->cap = right.cap;
            this->size = right.size;
        } else{
            memcpy(ptr, right.ptr, sizeof(T) * right.size);
            this->size = right.size;
        }
        return  ;

    }
//    auto __device__  operator=(const array &right) -> T&{
//        cudaMemcpy(ptr, right.ptr, sizeof(T) * right.size);
//        size = right.size;
//    }
};


#endif //NTOKEN_BASICDATATYPEDEFINE_CUH
