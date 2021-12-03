//
// Created by ascdc on 2021-12-03.
//

#ifndef NTOKEN_HYPERCOMPRESSBOOLARRAY_CUH
#define NTOKEN_HYPERCOMPRESSBOOLARRAY_CUH
#include <stdint.h>
struct hyperCompressBoolArray {
    uint8_t *ptr;
    size_t boolSize;
};
auto __device__ __host__ getHyperCompressBoolArray(hyperCompressBoolArray arr, size_t id) -> bool;
auto __device__ __host__ setHyperCompressBoolArray(hyperCompressBoolArray arr, size_t id, bool data);
auto __device__ __host__ getHyperCompressBoolArrayNeedSpace(size_t size) -> size_t;


#endif //NTOKEN_HYPERCOMPAREBOOLARRAY_CUH
