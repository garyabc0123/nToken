//
// Created by ascdc on 2021-11-22.
//

#include "nToken.cuh"

__global__ void walk(parseTreeInArray parseTree, documentTokenPtr document, gpuStack<parseTreeInArrayNode> stack, hyperCompressBoolArray answer){
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx > document.tokenSize){
        return;
    }

    for(size_t it = 0 ; it < parseTree.nodeListSize ; it++){
        push(stack, it, parseTree.nodeList[it]);
    }


}

nToken::nToken(char * documentPath, char * searchQueryPath){
    try{
        auto str = fileReader(documentPath);
        if(str.size() > 250000000){
            //limit : 0.25G char
            str.resize(250000000);
        }
        document = getDocumentToken(str);
    }
    catch (std::string e){
        std::cout << e << std::endl;
    }
    /*{
        auto str = fileReader(searchQueryPath);
        auto tuple = compiler(str);
        this->expression = std::get<0>(tuple);
        this->expressionDistList = std::get<1>(tuple);
        this->expressionSize = std::get<2>(tuple);
    }
    getPosition();*/

}
void nToken::getPosition() {
    std::vector<hyperCompressBoolArray> answer(expressionSize);
    try{
        for(size_t it = 0 ; it < expressionSize ; it++){
            auto error = cudaMallocManaged(reinterpret_cast<void **>(&(answer[it].ptr)), getHyperCompressBoolArrayNeedSpace(document.token.size));
            if(error != cudaSuccess){
                throw cudaGetErrorName(error);
            }
            answer[it].size = document.token.size;
        }
    }catch (const char * message){
        throw __FILE__ + std::to_string(__LINE__) + __func__ + "\n" + std::string(message);
    }



}