//
// Created by ascdc on 2021-12-17.
//
#include "nToken.cuh"

__host__ __device__ void walk(const parseTreeInArray parseTree, const documentToken document, gpuStack<parseTreeInArrayNode> stack, array<bool> answer, const size_t idx){
    if(idx > document.token.size){
        return;
    }
    parseTreeInArrayNode a, b;

    for(size_t it = parseTree.nodeListSize - 1 ; it != static_cast<size_t>(-1) ; it--){
        switch (parseTree.nodeList[it].type) {
            case symbolTable::dollarSign:
            {
                a = top(stack, idx);
                pop(stack, idx);
                bool equal = true;
                if(a.strInArrayEndId - a.strInArrayBeginId != document.token.ptr[idx].end - document.token.ptr[idx].begin){
                    equal = false;
                }
                for(size_t it2 = a.strInArrayBeginId, it3 = document.token.ptr[idx].begin ; it2 != a.strInArrayEndId && it3 != document.token.ptr[idx].end && equal == true; it2++, it3++){
                    if(parseTree.strArray[it2] != document.word.ptr[it3]){
                        equal = false;
                    }
                }
                a.type = symbolTable::boolean;
                a.data = equal;
                push(stack, idx, a);
            }
                break;
            case symbolTable::percentSign:
            {
                a = top(stack, idx);
                pop(stack, idx);
                uint16_t num = 0;
                for(size_t it2 = a.strInArrayBeginId ; it2 != a.strInArrayEndId ; it2++){
                    num *= 10;
                    num += parseTree.strArray[it2] - L'0';
                }
                a.type = symbolTable::boolean;
                a.data = (num == document.token.ptr[idx].partOfSpeech);
                push(stack, idx, a);
            }
                break;
            case symbolTable::verticalBar:
            {
                a = top(stack, idx);
                pop(stack, idx);
                b = top(stack, idx);
                pop(stack, idx);
                a.data |= b.data;
                push(stack, idx, a);
            }
                break;
            case symbolTable::exclamationMark:
            {
                a = top(stack, idx);
                pop(stack, idx);
                a.data = ~(a.data);
                push(stack, idx, a);
            }
                break;
            case symbolTable::caret:
            {
                a = top(stack, idx);
                pop(stack, idx);
                b = top(stack, idx);
                pop(stack, idx);
                a.data &= b.data;
                push(stack, idx, a);
            }
                break;
            case symbolTable::squareBracketLeft:
            case symbolTable::squareBracketRight:
            case symbolTable::curlyBracketLeft:
            case symbolTable::curlyBracketRight:
            case symbolTable::str:
            case symbolTable::boolean:
            case symbolTable::null:
                push(stack,idx, parseTree.nodeList[it]);
        }
    }
    answer.ptr[idx] = top(stack,idx).data;
}

__global__ void walk(parseTreeInArray parseTree, documentToken document, gpuStack<parseTreeInArrayNode> stack, array<bool> answer){
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    walk(parseTree, document, stack, answer, idx);

}

__global__ void compacter(array<bool> pred, array<size_t> scan, array<size_t> out){
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= pred.size){
        return;
    }
    if(!pred.ptr[idx]){
        return;
    }
    size_t newAddr = scan.ptr[idx];
    out.ptr[newAddr] = idx;
}


auto nToken::getPosition() -> std::vector<array<size_t>>{
    cudaError_t error;
    std::vector<array<size_t>> vecAns;
    for(size_t it = 0 ; it != expressionSize ; it++){
        array<bool> ans;
        ans.size = document.token.size;
        error = cudaMallocManaged(reinterpret_cast<void **>(&(ans.ptr)), sizeof(bool) * ans.size);
        if(error != cudaSuccess){
            throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
        }
        gpuStack<parseTreeInArrayNode> stack;
        stack.high = expression[it].nodeListSize + 1;
        stack.width = document.token.size;
        error = cudaMallocManaged(reinterpret_cast<void **>(&(stack.ptr)), sizeof (parseTreeInArrayNode) * stack.width * stack.high);
        if(error != cudaSuccess){
            std::wcout << "we need :" << sizeof (parseTreeInArrayNode) * stack.width * stack.high << std::endl;
            throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
        }
        error = cudaMallocManaged(reinterpret_cast<void **>(&(stack.top)), sizeof(size_t) * stack.width);
        cudaMemset(stack.top, 0, sizeof(size_t) * stack.width);
        if(error != cudaSuccess){
            throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
        }

        {



            for(size_t it2 = 0 ; it2 < document.token.size + 10 ; it2++){
                if(it2 == 327){
                    printf("\n");
                }
                walk(expression[it], document, stack, ans, it2);
            }


        }




        //walk<<<document.token.size / 512 + 1, 512>>>(expression[it], document, stack, ans);
        cudaDeviceSynchronize();
        error = cudaGetLastError();
        if(error != cudaSuccess){
            throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
        }

        cudaFree(stack.top);
        cudaFree(stack.ptr);

        array<size_t> scan;
        scan.size = ans.size;
        cudaMallocManaged(reinterpret_cast<void**>(&(scan.ptr)), sizeof(size_t) * scan.size);
        thrust::copy(thrust::device, ans.ptr, ans.ptr + ans.size, scan.ptr);
        thrust::exclusive_scan(thrust::device, scan.ptr, &(scan.ptr[scan.size ]), scan.ptr, static_cast<size_t>(0));
        array<size_t> compact;
        memcpy(&(compact.size), scan.ptr + scan.size - 1, sizeof(size_t));
        error = cudaMallocManaged(reinterpret_cast<void**>(&(compact.ptr)), sizeof(size_t) * compact.size);
        if(error != cudaSuccess){
            throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
        }
        compacter<<<ans.size / 1024 + 1, 1024>>>(ans, scan, compact);
        cudaDeviceSynchronize();
        error = cudaGetLastError();
        if(error != cudaSuccess){
            throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
        }
        vecAns.push_back(compact);
        std::wcout << "select id:" << it  << " size : " << compact.size <<std::endl;
        for(auto it2 = 0 ; it2 < compact.size ; it2++){
            std::wcout << compact.ptr[it2] << L" ";

        }
        std::wcout << std::endl;

        cudaFree(ans.ptr);
        cudaFree(scan.ptr);
    }
    return vecAns;
}