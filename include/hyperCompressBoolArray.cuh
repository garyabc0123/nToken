//
// Created by ascdc on 2021-12-03.
//

#ifndef NTOKEN_HYPERCOMPRESSBOOLARRAY_CUH
#define NTOKEN_HYPERCOMPRESSBOOLARRAY_CUH
#include <stdint.h>
struct hyperCompressBoolArray {
    uint8_t *ptr;
    size_t size;
};
bool __device__ __host__ getHyperCompressBoolArray(hyperCompressBoolArray arr, size_t id) ;
void __device__ __host__ setHyperCompressBoolArray(hyperCompressBoolArray arr, size_t id, bool data);
size_t __device__ __host__ getHyperCompressBoolArrayNeedSpace(size_t size);


#endif //NTOKEN_HYPERCOMPAREBOOLARRAY_CUH
