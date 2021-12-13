//
// Created by ascdc on 2021-12-03.
//

#ifndef NTOKEN_GPUSTACK_CUH
#define NTOKEN_GPUSTACK_CUH



template <typename T>
struct gpuStack{
    T *ptr;
    size_t width;
    size_t high;
    size_t *top;
};

template <typename T>
__device__ __host__ auto push(gpuStack<T> stack, size_t id, T data){
    stack.ptr[stack.width * stack.top[id] + id] = data;
    stack.top[id] ++;
}
template <typename T>
__device__ __host__ auto pop(gpuStack<T> stack, size_t id) -> T{
    stack.top[id]--;
    return stack.ptr[stack.width * stack.top[id] + id];
}
template <typename T>
__device__ __host__ auto top(gpuStack<T> stack, size_t id) -> T{

    return stack.ptr[stack.width * stack.top[id] + id];
}
template <typename T>
__device__ __host__ auto down(gpuStack<T> stack, size_t id) -> T{
    size_t myTop = 0;
    return stack.ptr[stack.width * myTop + id];
}

template <typename T>
__device__ __host__ auto isEmpty(gpuStack<T> stack, size_t id) -> bool{
    return stack.top[id] == 0;
}

template <typename T>
__device__ __host__ auto isFull(gpuStack<T> stack, size_t id) -> bool{
    return stack.ptr[id] == stack.high;
}

#endif //NTOKEN_GPUSTACK_CUH
