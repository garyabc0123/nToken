//
// Created by ascdc on 2021-11-29.
//

#ifndef NTOKEN_DOCUMENTPARSER_CUH
#define NTOKEN_DOCUMENTPARSER_CUH


#include "basicDataTypeDefine.cuh"
#include <string>
#include <sstream>

#include "thrust/universal_vector.h"
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
    thrust::universal_vector<wordAndPartOfSpeechPair> token;
    thrust::universal_vector<documentSentenceNode>  sentence;
    thrust::universal_vector<charType> word;
};

auto getDocumentToken(std::wstring &input) -> documentToken;

#endif //NTOKEN_DOCUMENTPARSER_CUH
