//
// Created by ascdc on 2021-11-22.
//

#ifndef NTOKEN_NTOKEN_CUH
#define NTOKEN_NTOKEN_CUH

#include "basicDataTypeDefine.cuh"
#include "operatorParser.cuh"
#include "documentParser.cuh"
#include "device_launch_parameters.h"
#include "fileReader.cuh"
#include "gpuStack.cuh"
#include <set>

#include "json.hpp"

class nToken {
public:
    nToken(char * documentPath, char * searchQueryPath);
    nToken(std::vector<uint8_t> textBytes, std::wstring searchQuery);
    void go();
    auto getPosition() -> std::vector<array<size_t>>;
    void dumpInfo();
    ~nToken();
    array<size_t> dfsTraversalSearch (std::vector<array<size_t>> inputPosition);
    void dumpPosition(std::string path);
    std::string getJSON();

private:
    parseTreeInArray * expression;
    distList expressionDistList;
    size_t expressionSize;
    documentToken document;
    std::vector<array<size_t>> position;
    array<size_t> local;
};


size_t __device__ __host__ findWordDeSentenceID(array<documentSentenceNode> sentence, size_t wordID);








#endif //NTOKEN_NTOKEN_CUH
