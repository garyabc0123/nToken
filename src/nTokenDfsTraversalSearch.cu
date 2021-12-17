//
// Created by ascdc on 2021-12-17.
//

#include "nToken.cuh"

struct stackStatus{
    size_t beginID, nowID;
};


//return id
// less than or equal
template<typename T>
size_t __device__ __host__ findingFloor(array<T> input, T number){
    if(number > input.ptr[input.size - 1]){
        return -1;
    }
    if(number < input.ptr[0]){
        return 0;
    }
    if(number == input.ptr[input.size - 1]){
        return input.size - 1;
    }
    size_t begin, end, now;
    begin = 0;
    end = input.size;
    while(end > begin){
        now = (end - begin) / 2 + begin;
        if(input.ptr[now] <= number && input.ptr[now + 1] > number){
            return now;
        }
        if(input.ptr[now] > number){
            begin = now + 1;
        }else{
            end = now - 1;
        }

    }
}

size_t __device__ __host__ findWordDeSentenceID(array<documentSentenceNode> sentence, size_t wordID){
    size_t begin, end, now;
    begin = 0;
    end = sentence.size;
    if(wordID < sentence.ptr[0].nodeBegin){
        return -1;
    }
    if(wordID > sentence.ptr[sentence.size - 1].nodeEnd){
        return -1;
    }
    while(begin < end){
        now = (end - begin) / 2 + begin;
        if(wordID >= sentence.ptr[now].nodeBegin && wordID < sentence.ptr[now].nodeEnd){
            return now;
        }
        if(wordID >= sentence.ptr[now].nodeEnd){
            begin = now + 1;
        }else{
            end = now - 1;
        }
    }
}

void __device__ __host__ dfsWalker(documentToken document, array<array<size_t>> inputPosition, array<size_t> distance, gpuStack<stackStatus> stack, array<size_t> ans, size_t idx){
    size_t mySentenceID = findWordDeSentenceID(document.sentence, inputPosition.ptr[0].ptr[idx]);

    push(stack, idx, stackStatus{
        .beginID = inputPosition.ptr[0].ptr[idx] - 1,
        .nowID = inputPosition.ptr[0].ptr[idx],

    });



    while (stack.top[idx] != inputPosition.size + 1){
        stackStatus nowTop = top(stack, idx);
        if(nowTop.nowID <= nowTop.beginID){
            if(stack.top[idx] <= 1){
                //failed
                return;
            }
            pop(stack, idx);
            nowTop = top(stack, idx);
            pop(stack, idx);
            nowTop.nowID = findingFloor(inputPosition.ptr[stack.top[idx] - 2], nowTop.nowID - 1);
            if(nowTop.nowID < document.sentence.ptr[mySentenceID].nodeBegin){
                return;
            }
            push(stack, idx, nowTop);
        }else if(stack.top[idx] >= inputPosition.size){
            //return data
            printf("%d\n",idx);
        }else{
            nowTop.beginID = nowTop.nowID;
            nowTop.nowID = findingFloor(inputPosition.ptr[stack.top[idx]], nowTop.nowID + distance.ptr[stack.top[idx]] >= document.sentence.ptr[mySentenceID].nodeEnd ? document.sentence.ptr[mySentenceID].nodeEnd - 1 : nowTop.nowID + distance.ptr[stack.top[idx]]);
            if(nowTop.nowID == -1){
                return;
            }
            push(stack, idx, nowTop);
        }
    }
}
void __global__ dfsWalker(documentToken document, array<array<size_t>> inputPosition, array<size_t> distance, gpuStack<stackStatus> stack, array<size_t> ans){
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    dfsWalker(document, inputPosition,  distance,  stack,  ans,  idx);
}


auto dfsTraversalSearch (std::vector<array<size_t>> inputPosition){
    cudaError_t error;


    array<size_t> ans;
    ans.size = inputPosition[0].size * inputPosition.size();
    error = cudaMallocManaged(reinterpret_cast<void **>(&(ans.ptr)), sizeof(size_t) * ans.size);
    if(error != cudaSuccess){
        std::wcout << "alloc size" << ans.size << std::endl;
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }

    gpuStack<stackStatus> stack;
    stack.width = inputPosition[0].size;
    stack.high = inputPosition.size();
    error = cudaMallocManaged(reinterpret_cast<void **>(&(stack.top)), sizeof(size_t) * stack.width);
    if(error != cudaSuccess){
        std::wcout << "alloc size" << ans.size << std::endl;
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }
    error = cudaMallocManaged(reinterpret_cast<void **>(&(stack.ptr)), sizeof(stackStatus) * stack.width * stack.high);
    if(error != cudaSuccess){
        std::wcout << "alloc size" << ans.size << std::endl;
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }
    array<array<size_t>> devInputPosition;
    devInputPosition.size = inputPosition.size();
    error = cudaMallocManaged(reinterpret_cast<void **>(&(devInputPosition.ptr)), sizeof(array<size_t>) * devInputPosition.size);
    if(error != cudaSuccess){
        std::wcout << "alloc size" << ans.size << std::endl;
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }
    thrust::copy(inputPosition.begin(), inputPosition.end(), devInputPosition.ptr);
}