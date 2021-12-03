//
// Created by ascdc on 2021-11-22.
//

#include "nToken.cuh"

__global__ void walk(parseTreeInArray parseTree, documentTokenPtr document, gpuStack<size_t> stack, hyperCompressBoolArray answer){
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx > document.tokenSize){
        return;
    }

    for(size_t it = 0 ; it < parseTree.nodeListSize ; it++){

    }


}

nToken::nToken(char * documentPath, char * searchQueryPath){
    {
        auto str = fileReader(documentPath);
        document = getDocumentToken(str);
    }
    {
        auto str = fileReader(searchQueryPath);
        auto tuple = compiler(str);
        this->expression = std::get<0>(tuple);
        this->expressionDistList = std::get<1>(tuple);
        this->expressionSize = std::get<2>(tuple);
    }
    getPosition();

}
void nToken::getPosition() {
    std::vector<hyperCompressBoolArray> answer(expressionSize);
    try{
        for(size_t it = 0 ; it < expressionSize ; it++){
            auto error = cudaMallocManaged(reinterpret_cast<void **>(&(answer[it].ptr)), getHyperCompressBoolArrayNeedSpace(document.token.size()));
            if(error != cudaSuccess){
                throw cudaGetErrorName(error);
            }
            answer[it].boolSize = document.token.size();
        }
    }catch (const char * message){
        throw __FILE__ + std::to_string(__LINE__) + __func__ + "\n" + std::string(message);
    }



}