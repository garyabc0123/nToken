//
// Created by ascdc on 2021-11-22.
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

nToken::nToken(char * documentPath, char * searchQueryPath){

    //prepare data
    try{
        auto str = gpuUTF8FileReader(documentPath);
        document = getDocumentToken(str);

    }
    catch (std::string e){
        std::wcout << std::wstring(e.begin(), e.end()) << std::endl;
        exit(-1);
    }
    {
        auto str = fileReader(searchQueryPath);
        auto tuple = compiler(str);
        this->expression = std::get<0>(tuple);
        this->expressionDistList = std::get<1>(tuple);
        this->expressionSize = std::get<2>(tuple);
    }
    dumpInfo();
    //step 1.
    try{
        getPosition();
    }
    catch (std::string e){
        std::wcout << std::wstring(e.begin(), e.end()) << std::endl;
        exit(-1);
    }

    //step 2.

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



void nToken::dumpInfo(){

    std::wcout << L"dump operator\n";
    for(size_t it = 0 ; it < expressionSize ; it++){
        std::wcout << "word tuple id: " << it << std::endl;
        std::wcout << "word tuple size: " << expression[it].nodeListSize << std::endl;
        std::wcout << "dist: " << expressionDistList[it] << std::endl;
        parseTreeInArray hostExpress;
        hostExpress.nodeListSize = this->expression[it].nodeListSize;
        hostExpress.nodeList = new parseTreeInArrayNode[hostExpress.nodeListSize];
        hostExpress.strArraySize = this->expression[it].strArraySize;
        hostExpress.strArray = new charType[hostExpress.strArraySize];
        cudaMemcpy(hostExpress.nodeList, this->expression[it].nodeList, sizeof(parseTreeInArrayNode) * hostExpress.nodeListSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(hostExpress.strArray, this->expression[it].strArray, sizeof(charType) * hostExpress.strArraySize, cudaMemcpyDeviceToHost);
        for(auto itInside = 0 ; itInside < hostExpress.nodeListSize ;itInside++){
            std::wcout << L"id: " << hostExpress.nodeList[itInside].tokenId << std::endl;
            std::wcout << L"type: " << hostExpress.nodeList[itInside].type << std::endl;
            if(hostExpress.nodeList[itInside].type == symbolTable::str){
                std::wstring temp;
                for(auto strIt = hostExpress.nodeList[itInside].strInArrayBeginId ; strIt != hostExpress.nodeList[itInside].strInArrayEndId ; strIt++){
                    temp.push_back(hostExpress.strArray[strIt]);
                }
                std::wcout << L"str: " << temp << std::endl;
            }
            std::wcout << L"---------------------///-----------------------\n";
        }

    }
    std::wcout << "token size " << document.token.size << std::endl;
    std::wcout << "word size " << document.word.size << std::endl;
    std::wcout << "sentence size " << document.sentence.size << std::endl;

    std::wcout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    /*
    for(size_t it = 0 ; it < document.token.size ; it++){
        std::wcout << "id: " << document.token.ptr[it].id << " pos:" << document.token.ptr[it].partOfSpeech << std::endl;
        std::wcout << "begin: " << document.token.ptr[it].begin << " end: " << document.token.ptr[it].end << std::endl;
        std::wstring temp;
        for(auto itt = document.token.ptr[it].begin ; itt != document.token.ptr[it].end ; itt++){
            temp.push_back(document.word.ptr[itt]);
        }
        std::wcout << "str: " << temp << std::endl;
        std::wcout << L"---------------------///-----------------------\n";

    }*/
}

nToken::~nToken(){
    cudaError_t error;
    std::string errorInfo;
    documentToken document;
    for(size_t it = 0 ; it  < expressionSize ; it++){
        error = cudaFree(this->expression[it].nodeList);
        if(error != cudaSuccess){
            errorInfo = __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
            std::wcout <<  std::wstring (errorInfo.begin(), errorInfo.end());
        }
        error = cudaFree(this->expression[it].strArray);
        if(error != cudaSuccess){
            errorInfo = __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
            std::wcout <<  std::wstring (errorInfo.begin(), errorInfo.end());
        }
    }
    error = cudaFree(this->expressionDistList);
    if(error != cudaSuccess){
        errorInfo = __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
        std::wcout <<  std::wstring (errorInfo.begin(), errorInfo.end());
    }
    error = cudaFree(this->document.token.ptr);
    if(error != cudaSuccess){
        errorInfo = __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
        std::wcout <<  std::wstring (errorInfo.begin(), errorInfo.end());
    }
    error = cudaFree(this->document.word.ptr);
    if(error != cudaSuccess){
        errorInfo = __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
        std::wcout <<  std::wstring (errorInfo.begin(), errorInfo.end());
    }
    error = cudaFree(this->document.sentence.ptr);
    if(error != cudaSuccess){
        errorInfo = __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
        std::wcout <<  std::wstring (errorInfo.begin(), errorInfo.end());
    }

}