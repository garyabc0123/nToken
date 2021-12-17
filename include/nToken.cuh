//
// Created by ascdc on 2021-11-22.
//

#ifndef NTOKEN_NTOKEN_CUH
#define NTOKEN_NTOKEN_CUH

#include "basicDataTypeDefine.cuh"
#include "operatorParser.cuh"
#include "fileReader.cuh"
#include "documentParser.cuh"
#include "gpuStack.cuh"
#include "device_launch_parameters.h"
#include <set>

class nToken {
public:
    nToken(char * documentPath, char * searchQueryPath);
    auto getPosition() -> std::vector<array<size_t>>;
    void dumpInfo();
    ~nToken();

private:
    parseTreeInArray * expression;
    distList expressionDistList;
    size_t expressionSize;
    documentToken document;
};

//struct documentTokenPtr{
//    wordAndPartOfSpeechPair* token;
//    size_t tokenSize;
//
//    documentSentenceNode*  sentence;
//    size_t sentenceSize;
//
//    charType* word;
//    size_t wordSize;
//};






#endif //NTOKEN_NTOKEN_CUH
