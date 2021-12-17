//
// Created by ascdc on 2021-11-22.
//

#include "nToken.cuh"



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