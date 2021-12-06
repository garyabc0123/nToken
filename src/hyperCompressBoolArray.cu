//
// Created by ascdc on 2021-12-03.
//
#include "hyperCompressBoolArray.cuh"
bool  __device__ __host__ getHyperCompressBoolArray(hyperCompressBoolArray arr, size_t id) {
    size_t address = id / 8;
    uint8_t offset = id % 8;
    return (arr.ptr[address] & ( 1 << offset )) >> offset;
}
void __device__ __host__ setHyperCompressBoolArray(hyperCompressBoolArray arr, size_t id, bool data){
    size_t address = id / 8;
    uint8_t offset = id % 8;
    arr.ptr[address] |= data << offset;

}
size_t __device__ __host__ getHyperCompressBoolArrayNeedSpace(size_t size) {
    return (size % 8 + 1) * sizeof(uint8_t);
}
