//
// Created by ascdc on 2021-12-03.
//

#ifndef NTOKEN_HYPERCOMPRESSBOOLARRAY_CUH
#define NTOKEN_HYPERCOMPRESSBOOLARRAY_CUH

struct hyperCompressBoolArray {
    uint8_t *ptr;
    size_t boolSize;
};
auto __device__ __host__ getHyperCompressBoolArray(hyperCompressBoolArray arr, size_t id) -> bool{
    size_t address = id / 8;
    uint8_t offset = id % 8;
    return (arr.ptr[address] & ( 1 << offset )) >> offset;
}
auto __device__ __host__ setHyperCompressBoolArray(hyperCompressBoolArray arr, size_t id, bool data){
    size_t address = id / 8;
    uint8_t offset = id % 8;
    arr.ptr[address] |= data << offset;

}
auto __device__ __host__ getHyperCompressBoolArrayNeedSpace(size_t size) -> size_t{
    return (size % 8 + 1) * sizeof(uint8_t);
}


#endif //NTOKEN_HYPERCOMPAREBOOLARRAY_CUH
