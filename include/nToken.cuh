//
// Created by ascdc on 2021-11-22.
//

#ifndef NTOKEN_NTOKEN_CUH
#define NTOKEN_NTOKEN_CUH

#include "basicDataTypeDefine.cuh"



class nToken {

};




/**
 *  document token stream structure
 */
struct wordAndPartOfSpeechPair{
    int id;
    charType * word;
    partWordOfSpeechType partOfSpeech;
};



#endif //NTOKEN_NTOKEN_CUH
