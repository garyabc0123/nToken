//
// Created by ascdc on 2021-11-29.
//

#ifndef NTOKEN_DOCUMENTPARSER_CUH
#define NTOKEN_DOCUMENTPARSER_CUH


#include "basicDataTypeDefine.cuh"
#include <string>
#include <sstream>
#include "thrust/universal_vector.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "device_launch_parameters.h"
/**
 *  document token stream structure
 */
struct wordAndPartOfSpeechPair{
    size_t id;
    size_t begin,  end;
    partWordOfSpeechType partOfSpeech;
};


struct documentSentenceNode{
    size_t id;
    size_t nodeBegin, nodeEnd;

};
struct documentToken{
#define DOCUMENTTYPE thrust::universal_vector
#define DOCUMENTTYPE array
    DOCUMENTTYPE<wordAndPartOfSpeechPair> token;
    DOCUMENTTYPE<documentSentenceNode>  sentence;
    DOCUMENTTYPE<charType> word;

};

//auto getDocumentToken(std::wstring &input) -> documentToken;
auto getDocumentToken(array<charType> devInput) -> documentToken;

#endif //NTOKEN_DOCUMENTPARSER_CUH
