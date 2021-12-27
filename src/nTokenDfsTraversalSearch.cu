//
// Created by ascdc on 2021-12-17.
//

#include "nToken.cuh"

struct stackStatus{
    size_t beginData, endData; //token id
    size_t  nowID; //this level position id
    size_t level;
};


/** return id
    less than or equal
 */
//template<typename T>
//size_t __device__ __host__ findingFloor(array<T> input, T number){
//    if(number == input.ptr[0]){
//        return 0;
//    }
//    if(number >= input.ptr[input.size - 1]){
//        return input.size - 1;
//    }
//    if(number < input.ptr[0]){
//        return -1;
//    }
//
//
//    size_t begin, end, now;
//    begin = 0;
//    end = input.size;
//    while(end >= begin){
//        now = (end - begin) / 2 + begin;
//        if(input.ptr[now - 1] <= number && input.ptr[now ] >= number){
//            return now - 1;
//        }
//        if(input.ptr[now] > number){
//            end = now - 1;
//        }else{
//            begin = now + 1;
//        }
//
//    }
//}

template<typename T>
size_t __device__ __host__ findingFloor(array<T> input, T number) {
    for(size_t it = input.size - 1; it != -1 ; it --){
        if(input.ptr[it] <= number)
            return it;
    }
    return -1;

}
size_t __device__ __host__ findWordDeSentenceID(array<documentSentenceNode> sentence, size_t wordID){
    size_t begin, end, now;
    begin = 0;
    end = sentence.size;
    if(wordID < sentence.ptr[0].nodeBegin){
        return static_cast<size_t>(-1);
    }
    if(wordID > sentence.ptr[sentence.size - 1].nodeEnd){
        return static_cast<size_t>(-1);
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
        .beginData = inputPosition.ptr[0].ptr[idx],
        .endData = inputPosition.ptr[0].ptr[idx],
        .nowID = idx,
        .level = 0
    });


    while (!isEmpty(stack,idx)){
        auto level = stack.top[idx];
        if(level == inputPosition.size){
            break;
        }
        auto nowTop = top(stack, idx);
        if(inputPosition.ptr[nowTop.level].ptr[nowTop.nowID] >= nowTop.beginData){
            auto number = inputPosition.ptr[nowTop.level].ptr[nowTop.nowID] + distance.ptr[nowTop.level];
            if(number > document.sentence.ptr[mySentenceID].nodeEnd)
                number = document.sentence.ptr[mySentenceID].nodeEnd;
            auto next = findingFloor(inputPosition.ptr[nowTop.level + 1], number);
            if(next == -1 ||
            inputPosition.ptr[nowTop.level + 1].ptr[next] < document.sentence.ptr[mySentenceID].nodeBegin ||
            inputPosition.ptr[nowTop.level + 1].ptr[next] >= document.sentence.ptr[mySentenceID].nodeEnd ||
            inputPosition.ptr[nowTop.level + 1].ptr[next] < inputPosition.ptr[nowTop.level].ptr[nowTop.nowID]){
                auto temp = top(stack, idx);
                temp.nowID --;
                pop(stack, idx);
                push(stack, idx, temp);

            }else{
                stackStatus temp;
                temp.beginData = inputPosition.ptr[nowTop.level].ptr[nowTop.nowID] + 1;
                temp.endData = inputPosition.ptr[nowTop.level + 1].ptr[next];
                temp.nowID = next;
                temp.level = nowTop.level + 1;
                push(stack, idx, temp);
            }
        }else{
            if(isEmpty(stack, idx))
                break;
            while (inputPosition.ptr[nowTop.level].ptr[nowTop.nowID] < nowTop.beginData || nowTop.nowID == -1 ){
                pop(stack, idx);
                if(isEmpty(stack, idx))
                    break;
                auto temp = top(stack, idx);
                temp.nowID--;
                pop(stack, idx);
                push(stack, idx, temp);
                nowTop = top(stack, idx);
            }

        }

    }
    bool eq = false;
    for(size_t it = 0 ; it < inputPosition.size - 1; it++){
        if(inputPosition.ptr[it].ptr[get(stack, idx, it).nowID] == inputPosition.ptr[it+1].ptr[get(stack, idx, it + 1).nowID]){
            eq = true;
            break;
        }
    }
    if(!isEmpty(stack, idx) && !eq){
        printf("%zd\n", idx);
        for(size_t it = 0 ; it < inputPosition.size ; it++){
            ans.ptr[idx * inputPosition.size + it] = inputPosition.ptr[it].ptr[get(stack, idx, it).nowID];
        }
    }else{
        for(size_t it = 0 ; it < inputPosition.size ; it++){
            ans.ptr[idx * inputPosition.size + it] = -1;
        }
    }



}
void __global__ dfsWalker(documentToken document, array<array<size_t>> inputPosition, array<size_t> distance, gpuStack<stackStatus> stack, array<size_t> ans){
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= inputPosition.ptr[0].size)
        return;
    dfsWalker(document, inputPosition,  distance,  stack,  ans,  idx);
}

template <typename T>
struct notEqual{
        T val;
        notEqual(T input):val(input){}
        bool __device__ __host__ operator()(T input){
            return input != val;
        }
    };

array<size_t> nToken::dfsTraversalSearch (std::vector<array<size_t>> inputPosition){

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
    thrust::fill(thrust::device, stack.top, stack.top + stack.width, 0);
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

    for(size_t idx = 0 ; idx < devInputPosition.ptr[0].size ; idx++){
        dfsWalker(this->document, devInputPosition, array<size_t>{.ptr=this->expressionDistList, .size = devInputPosition.size}, stack, ans, idx);
    }

    //dfsWalker<<<devInputPosition.ptr[0].size / 512 + 1, 512>>>(this->document, devInputPosition, array<size_t>{.ptr=this->expressionDistList, .size = devInputPosition.size}, stack, ans);
    cudaDeviceSynchronize();
    error = cudaGetLastError();

    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }

    cudaFree(devInputPosition.ptr);
    cudaFree(stack.ptr);
    cudaFree(stack.top);

    array<size_t> compactAns;
    compactAns.size = thrust::count_if(thrust::host, ans.ptr, ans.ptr + ans.size, notEqual<size_t>{static_cast<size_t>(-1)});
    std::wcout <<compactAns.size << std::endl;
    error = cudaMallocManaged(reinterpret_cast<void **>(&(compactAns.ptr)), sizeof(size_t) * compactAns.size);
    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }
    thrust::copy_if(thrust::device, ans.ptr, ans.ptr + ans.size, compactAns.ptr, notEqual<size_t>{static_cast<size_t>(-1)});
//    for(size_t it = 0 ; it < compactAns.size ; it++){
//        std::wcout << it << ": " << compactAns.ptr[it] << std::endl;
//    }
//    std::wcout << compactAns.size / inputPosition.size() << std::endl;

    return ans;


}
