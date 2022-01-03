//
// Created by ascdc on 2021-11-22.
//

#include "nToken.cuh"




nToken::nToken(char * documentPath, char * searchQueryPath){
    cudaSetDevice(0);
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



}

nToken::nToken(std::vector<uint8_t> textBytes, std::wstring searchQuery){
    cudaSetDevice(0);
    try{
        auto str = gpuUTF8Loader(textBytes);
        document = getDocumentToken(str);

    }
    catch (std::string e){
        std::wcout << std::wstring(e.begin(), e.end()) << std::endl;
        exit(-1);
    }
    {
        auto str = searchQuery;
        auto tuple = compiler(str);
        this->expression = std::get<0>(tuple);
        this->expressionDistList = std::get<1>(tuple);
        this->expressionSize = std::get<2>(tuple);
    }
    dumpInfo();
}

void nToken::go(){
    //step 1.
    try{
        position = getPosition();
    }
    catch (std::string e){
        std::wcout << std::wstring(e.begin(), e.end()) << std::endl;
        exit(-1);
    }

    //step 2.

    try{
        local = dfsTraversalSearch(position);
    }catch (std::string e){
        std::wcout << std::wstring(e.begin(), e.end()) << std::endl;
        exit(-1);
    }

    dumpPosition("output.json");
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

void nToken::dumpPosition(std::string path){
    nlohmann::json json;

    {

        //json["position"] = nlohmann::j
        for(size_t oldIt = 0 ; oldIt != position.size() ; oldIt++){
            json["position"].push_back(nlohmann::json());
            json["position"].back()["phraseID"] = oldIt;
            size_t sentenceID = -1;
            for(size_t it = 0 ; it < position[oldIt].size ; it++){
                auto oldSentenceID = sentenceID;
                sentenceID = findWordDeSentenceID(document.sentence, position[oldIt].ptr[it]);
                if(sentenceID != oldSentenceID){
                    json["position"].back()["data"].push_back(nlohmann::json());
                    json["position"].back()["data"].back()["sentenceID"] = sentenceID;

                }
                json["position"].back()["data"].back()["data"].push_back(position[oldIt].ptr[it]);

            }
        }
    }
    {

        size_t sentenceID = -1;
        for(size_t it = 0 ; it < local.size ; it++){
            if(local.ptr[it] != -1){
                if(it % position.size() == 0){
                    auto oldSentenceID = sentenceID;
                    sentenceID = findWordDeSentenceID(document.sentence, local.ptr[it]);
                    if(oldSentenceID != sentenceID){
                        //json["answer"].push_back(ansType{.sentenceID = sentenceID, .ansList = std::vector<std::vector<size_t>>()});
                        json["answer"].push_back(nlohmann::json());
                        json["answer"].back()["sentenceID"] = sentenceID;
                    }
                    json["answer"].back()["data"].push_back(std::vector<size_t>());

                }
                //newAns.back().ansList.back().push_back(local.ptr[it]);
                json["answer"].back()["data"].back().push_back(local.ptr[it]);
                //newAns[std::to_string(sentenceID)].back().push_back(local.ptr[it]);
            }

        }



    }



    std::ofstream o(path);
    o << std::setw(4) << json << std::endl;
}

nToken::~nToken(){
    cudaError_t error;
    std::string errorInfo;

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

    for(auto it = position.begin() ; it != position.end() ; it++){
        cudaFree(it->ptr);
    }

}

std::string nToken::getJSON() {
    nlohmann::json json;

    {

        //json["position"] = nlohmann::j
        for(size_t oldIt = 0 ; oldIt != position.size() ; oldIt++){
            json["position"].push_back(nlohmann::json());
            json["position"].back()["phraseID"] = oldIt;
            size_t sentenceID = -1;
            for(size_t it = 0 ; it < position[oldIt].size ; it++){
                auto oldSentenceID = sentenceID;
                sentenceID = findWordDeSentenceID(document.sentence, position[oldIt].ptr[it]);
                if(sentenceID != oldSentenceID){
                    json["position"].back()["data"].push_back(nlohmann::json());
                    json["position"].back()["data"].back()["sentenceID"] = sentenceID;

                }
                json["position"].back()["data"].back()["data"].push_back(position[oldIt].ptr[it]);

            }
        }
    }
    {

        size_t sentenceID = -1;
        for(size_t it = 0 ; it < local.size ; it++){
            if(local.ptr[it] != -1){
                if(it % position.size() == 0){
                    auto oldSentenceID = sentenceID;
                    sentenceID = findWordDeSentenceID(document.sentence, local.ptr[it]);
                    if(oldSentenceID != sentenceID){
                        //json["answer"].push_back(ansType{.sentenceID = sentenceID, .ansList = std::vector<std::vector<size_t>>()});
                        json["answer"].push_back(nlohmann::json());
                        json["answer"].back()["sentenceID"] = sentenceID;
                    }
                    json["answer"].back()["data"].push_back(std::vector<size_t>());

                }
                //newAns.back().ansList.back().push_back(local.ptr[it]);
                json["answer"].back()["data"].back().push_back(local.ptr[it]);
                //newAns[std::to_string(sentenceID)].back().push_back(local.ptr[it]);
            }

        }



    }

    std::string temp = json.dump();

    return  temp;
}