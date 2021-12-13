//
// Created by ascdc on 2021-11-22.
//

#include "nToken.cuh"

__host__ __device__ void walk(parseTreeInArray parseTree, documentToken document, gpuStack<parseTreeInArrayNode> stack, array<bool> answer, size_t idx){
    if(idx > document.word.size){
        return;
    }
    parseTreeInArrayNode a, b;

    for(size_t it = parseTree.nodeListSize - 1 ; it >= 0 ; it--){
        switch (parseTree.nodeList[it].type) {
            case symbolTable::dollarSign:
            {
                a = top(stack, idx);
                pop(stack, idx);
                bool equal = true;
                if(a.strInArrayEndId - a.strInArrayBeginId != document.token.ptr[idx].end - document.token.ptr[idx].begin){
                    equal = false;
                }
                for(size_t it = a.strInArrayBeginId, it2 = document.token.ptr[idx].begin ; it != a.strInArrayEndId && it2 != document.token.ptr[idx].end && equal == true; it++, it2++){
                    if(parseTree.strArray[it] != document.word.ptr[it2]){
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
                for(size_t it = a.strInArrayBeginId ; it != a.strInArrayEndId ; it++){
                    num *= 10;
                    num += parseTree.strArray[it] - L'0';
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
                a.data |= b.data;
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
                push(stack,idx, parseTree.nodeList[idx]);
        }
    }
    answer.ptr[idx] = top(stack,idx).data;
}

__global__ void walk(parseTreeInArray parseTree, documentToken document, gpuStack<parseTreeInArrayNode> stack, array<bool> answer){
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    walk(parseTree, document, stack, answer, idx);

}

nToken::nToken(char * documentPath, char * searchQueryPath){
    try{
        auto str = gpuUTF8FileReader(documentPath);
        document = getDocumentToken(str);

    }
    catch (std::string e){
        std::wcout << std::wstring(e.begin(), e.end()) << std::endl;
    }
    {
        auto str = fileReader(searchQueryPath);
        auto tuple = compiler(str);
        this->expression = std::get<0>(tuple);
        this->expressionDistList = std::get<1>(tuple);
        this->expressionSize = std::get<2>(tuple);
    }
    dumpInfo();
    try{
        getPosition();
    }
    catch (std::string e){
        std::wcout << std::wstring(e.begin(), e.end()) << std::endl;
    }

}

void nToken::getPosition() {
    cudaError_t error;

    for(size_t it = 0 ; it != expressionSize ; it++){
        array<bool> ans;
        error = cudaMalloc(reinterpret_cast<void **>(&(ans.ptr)), sizeof(bool) * document.word.size);
        if(error != cudaSuccess){
            throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
        }
        gpuStack<parseTreeInArrayNode> stack;
        stack.high = expression[it].nodeListSize;
        stack.width = document.word.size;
        error = cudaMalloc(reinterpret_cast<void **>(&(stack.ptr)), sizeof (parseTreeInArrayNode) * stack.width * stack.high);
        stack.high = expression[it].nodeListSize;
        stack.width = document.word.size;
        if(error != cudaSuccess){
            std::wcout << "we need :" << sizeof (parseTreeInArrayNode) * expression[it].nodeListSize * document.word.size << std::endl;
            throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
        }
        error = cudaMalloc(reinterpret_cast<void **>(&(stack.top)), sizeof(size_t) * stack.width);
        if(error != cudaSuccess){
            throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
        }

        {
            parseTreeInArray hostExpression;
            documentToken hostDocument;
            gpuStack<parseTreeInArrayNode> hostStack;
            array<bool> hostAns;

            hostExpression.nodeListSize = expression[it].nodeListSize;
            hostExpression.strArraySize = expression[it].strArraySize;
            hostExpression.nodeList = new parseTreeInArrayNode[hostExpression.nodeListSize];
            hostExpression.strArray = new charType [hostExpression.strArraySize];
            hostDocument.word.size = document.word.size;
            hostDocument.token.size = document.token.size;
            hostDocument.sentence.size = document.sentence.size;
            hostDocument.word.ptr = new charType [hostDocument.word.size];
            hostDocument.token.ptr = new wordAndPartOfSpeechPair[hostDocument.token.size];
            hostDocument.sentence.ptr = new documentSentenceNode[hostDocument.sentence.size];
            hostStack.top = stack.top;
            hostStack.width = stack.width;
            hostStack.ptr = new parseTreeInArrayNode[hostStack.high * hostStack.width];
            hostStack.top = new size_t [hostStack.width];
            hostAns.size = ans.size;
            hostAns.ptr = new bool [hostAns.size];

            cudaMemcpy(hostExpression.nodeList, expression[it].nodeList, sizeof(parseTreeInArrayNode) * hostExpression.nodeListSize, cudaMemcpyDeviceToHost);
            cudaMemcpy(hostExpression.strArray, expression->strArray, sizeof(charType) * expression->strArraySize, cudaMemcpyDeviceToHost);
            cudaMemcpy(hostDocument.word.ptr, document.word.ptr, sizeof(charType) * hostDocument.word.size, cudaMemcpyDeviceToHost);
            cudaMemcpy(hostDocument.token.ptr, document.token.ptr, sizeof(wordAndPartOfSpeechPair) * hostDocument.token.size, cudaMemcpyDeviceToHost);



        }




        walk<<<document.word.size / 512 +1, 512>>>(expression[it], document, stack, ans);
        cudaDeviceSynchronize();
        error = cudaGetLastError();
        if(error != cudaSuccess){
            throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
        }


        array<bool> vecAns;
        vecAns.size = ans.size;
        vecAns.ptr = new bool[vecAns.size];
        cudaMemcpy(vecAns.ptr, ans.ptr, ans.size * sizeof(bool), cudaMemcpyDeviceToHost);
        std::wcout << "select id: " << it << std::endl;
        for(auto i2t = 0 ; i2t < vecAns.size ; i2t++){
            if(vecAns.ptr[i2t]){
                std::wcout << i2t << " " ;
            }
        }
        std::wcout << std::endl;

        cudaFree(ans.ptr);
        cudaFree(stack.ptr);
    }
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
}