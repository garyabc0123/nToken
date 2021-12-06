//
// Created by ascdc on 2021-11-29.
//



#include "documentParser.cuh"





/**
 * Check if every character in string is equal to ch
 * @param devInputStr
 * @return output
 * @param ch
 *
 */
__global__ void tagChar(array<charType> devInputStr, array<size_t> output, charType ch){
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= devInputStr.size)
        return;
    if(devInputStr.ptr[idx] == ch){
        output.ptr[idx] = true;
    }
}

/**
 * create document token list and line info
 * @param scanSpace
 * @param scanBreak
 * @param str
 * @return token
 * @return document
 * @param idx
 */
__host__ __device__ void  writeTokenData(array<size_t> scanSpace, array<size_t> scanBreak, array<charType > str, array<wordAndPartOfSpeechPair> token, array<documentSentenceNode> document, size_t idx){
    if(idx >= scanSpace.size)
        return;
    size_t myTokenId = scanSpace.ptr[idx] / 2;
    bool isPOS = scanSpace.ptr[idx] & 1;
    if(!isPOS){
        if(idx == 0){
            token.ptr[myTokenId].begin = 0;
            token.ptr[myTokenId].id = myTokenId;
        }else if(scanSpace.ptr[idx] != scanSpace.ptr[idx - 1]){
            //和左邊的不一樣
            token.ptr[myTokenId].begin = idx;
            token.ptr[myTokenId].id = myTokenId;

        }
        if(scanSpace.ptr[idx] != scanSpace.ptr[idx + 1]){
            //和右邊的不一樣
            token.ptr[myTokenId].end = idx;
            token.ptr[myTokenId].id = myTokenId;
        }


    }else if(str.ptr[idx] == L'('){
        uint16_t num = 0;
        auto it = idx + 1;
        while(str.ptr[it] != L')'){
            num *= 10;
            num += str.ptr[it] - L'0';
            it++;
        }
        token.ptr[myTokenId].partOfSpeech = num;
    }

    size_t mySentenceId = scanBreak.ptr[idx];
    if(idx == 0){
        document.ptr[mySentenceId].id = mySentenceId;
        document.ptr[mySentenceId].nodeBegin = myTokenId;
    }else if(idx == scanBreak.size - 1){
        document.ptr[mySentenceId].id = mySentenceId;
        document.ptr[mySentenceId].nodeEnd = myTokenId + 1;
    }else{
        if(scanBreak.ptr[idx] != scanBreak.ptr[idx - 1]){
            //和左邊的不一樣
            document.ptr[mySentenceId].nodeBegin = myTokenId;
            document.ptr[mySentenceId].id = mySentenceId;

        }

        if(scanBreak.ptr[idx] != scanBreak.ptr[idx + 1]){
            //和右邊的不一樣
            document.ptr[mySentenceId].nodeEnd = myTokenId + 1;
            document.ptr[mySentenceId].id = mySentenceId;
        }
    }

}

__global__ void writeTokenData(array<size_t> scanSpace, array<size_t> scanBreak, array<charType > str, array<wordAndPartOfSpeechPair> token, array<documentSentenceNode> document){
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    writeTokenData(scanSpace, scanBreak, str, token, document, idx);

}

/**
 * convert input string to  documentToken
 * @param input
 * @return documentToken(devive ptr)
 */
auto getDocumentToken(std::wstring &input) -> documentToken{
    array<charType> devInput;
    auto error = cudaMalloc(reinterpret_cast<void **>(&(devInput.ptr)), sizeof(charType) * input.size());
    devInput.size = input.size();
    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }
    error = cudaMemcpy(devInput.ptr, input.c_str(), sizeof(charType) * devInput.size, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }

    array<size_t> isSpace;
    array<size_t> isBreak;
    isSpace.size = devInput.size;
    isBreak.size = devInput.size;
    error = cudaMalloc(reinterpret_cast<void **>(&(isSpace.ptr)), isSpace.size * sizeof(size_t));
    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }
    error = cudaMalloc(reinterpret_cast<void **>(&(isBreak.ptr)), isBreak.size * sizeof(size_t));
    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }

    tagChar<<<devInput.size / 512 + 1, 512>>>(devInput, isSpace, L' ');
    cudaDeviceSynchronize();
    tagChar<<<devInput.size / 512 + 1, 512>>>(devInput, isBreak, L'\n');
    cudaDeviceSynchronize();
    tagChar<<<devInput.size / 512 + 1, 512>>>(devInput, isSpace, L'\n');
    cudaDeviceSynchronize();
    error = cudaGetLastError();

    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }

    array<size_t> scanSpace;
    array<size_t> scanBreak;
    scanSpace.size = devInput.size;
    scanBreak.size = devInput.size;

    error = cudaMalloc(reinterpret_cast<void **>(&(scanSpace.ptr)), scanSpace.size * sizeof(size_t));
    if(error != cudaSuccess){
        std::cout << scanSpace.size  << std::endl;
        std::cout << scanSpace.size * sizeof(size_t) << std::endl;
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }
    error = cudaMalloc(reinterpret_cast<void **>(&(scanBreak.ptr)), scanBreak.size * sizeof(size_t));
    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }
    thrust::exclusive_scan(thrust::device, isSpace.ptr, isSpace.ptr + isSpace.size, scanSpace.ptr);
    thrust::exclusive_scan(thrust::device, isBreak.ptr, isBreak.ptr + isBreak.size, scanBreak.ptr);



    array<wordAndPartOfSpeechPair> token;
    array<documentSentenceNode> document;
    error = cudaMemcpy(&(token.size), scanSpace.ptr + scanSpace.size - 1, sizeof(size_t), cudaMemcpyDeviceToHost);
    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }
    error = cudaMemcpy(&(document.size), scanBreak.ptr + scanBreak.size - 1, sizeof(size_t), cudaMemcpyDeviceToHost);
    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }
    token.size += 1;
    token.size /= 2;
    document.size++;
    error = cudaMalloc(reinterpret_cast<void **>(&(token.ptr)), sizeof(wordAndPartOfSpeechPair) * token.size);
    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }
    error = cudaMalloc(reinterpret_cast<void **>(&(document.ptr)), sizeof(documentSentenceNode) * document.size);
    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }

    writeTokenData<<<devInput.size / 1024 + 1, 1024>>>(scanSpace, scanBreak, devInput, token, document);
    /*
    {
        auto hostScanSpace = scanSpace;
        auto hostScanBreak = scanBreak;
        auto hostToken = token;
        auto hostDocument = document;
        auto hostInput = devInput;
        hostScanSpace.ptr = new size_t[hostScanSpace.size];
        hostScanBreak.ptr = new size_t[hostScanBreak.size];
        hostInput.ptr = new charType[hostInput.size];
        hostToken.ptr = new wordAndPartOfSpeechPair[hostToken.size];
        hostDocument.ptr = new documentSentenceNode[hostDocument.size];
        cudaMemcpy(hostScanSpace.ptr, scanSpace.ptr, scanSpace.size * sizeof(size_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostScanBreak.ptr, scanBreak.ptr, scanBreak.size * sizeof(size_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostInput.ptr, devInput.ptr, devInput.size * sizeof(charType), cudaMemcpyDeviceToHost);
        for(auto idx = 0 ; idx < devInput.size ; idx++){
            writeTokenData(hostScanSpace, hostScanBreak, hostInput, hostToken, hostDocument, idx);
        }

    }
    */
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }

    /*{
        array<wordAndPartOfSpeechPair> hostToken;
        array<documentSentenceNode> hostDoc;
        hostToken.size = token.size;
        hostDoc.size = document.size;
        hostToken.ptr = new wordAndPartOfSpeechPair[hostToken.size];
        hostDoc.ptr = new documentSentenceNode[hostDoc.size];
        cudaMemcpy(hostToken.ptr, token.ptr, token.size * sizeof(wordAndPartOfSpeechPair), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostDoc.ptr, document.ptr, document.size * sizeof(documentSentenceNode), cudaMemcpyDeviceToHost);

        for(auto it = 0 ; it < hostToken.size ; it++){
            std::wcout << "id: " << hostToken.ptr[it].id << std::endl;
            std::wcout << "pos: " << hostToken.ptr[it].partOfSpeech << std::endl;
            std::wcout << "begin: " << hostToken.ptr[it].begin <<" end: " << hostToken.ptr[it].end << std::endl;
            std::wcout << input.substr(hostToken.ptr[it].begin, hostToken.ptr[it].end - hostToken.ptr[it].begin) << std::endl;
            std::wcout << "@@@@@@@@@@@@\n";
        }
        for(auto it = 0 ; it < document.size ; it++){
            std::wcout << "docId: " << hostDoc.ptr[it].id << std::endl;
            std::wcout << "from: " << hostDoc.ptr[it].nodeBegin << " to: " << hostDoc.ptr[it].nodeEnd << std::endl;
            std::wcout << "@@@@@@@@@@@@\n";

        }
    }*/





    cudaFree(isSpace.ptr);
    cudaFree(isBreak.ptr);
    cudaFree(scanSpace.ptr);
    cudaFree(scanBreak.ptr);
    return {
        token,
        document,
        devInput
    };
}