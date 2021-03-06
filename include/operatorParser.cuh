//
// Created by ascdc on 2021-11-22.
//

#ifndef NTOKEN_OPERATORPARSER_CUH
#define NTOKEN_OPERATORPARSER_CUH

#include "basicDataTypeDefine.cuh"
#include <string>
#include <tuple>
#include <locale>
#include <codecvt>
#include <deque>
#include <stack>
#include <algorithm>
#include <iostream>
#include <map>
#include "device_launch_parameters.h"


enum struct symbolTable{
    null,
    dollarSign,
    percentSign,
    verticalBar,
    exclamationMark,
    caret,
    squareBracketLeft,
    squareBracketRight,
    curlyBracketLeft,
    curlyBracketRight,
    str,
    boolean

};
std::wostream& operator<<(std::wostream& out, const symbolTable value);


struct symbolTokenStream{
    size_t id;
    symbolTable type;
    std::wstring str;

};

using distList = size_t*;
struct parseTree{
    symbolTokenStream token;
    parseTree *left;
    parseTree *right;

    ~parseTree(){
        delete left;
        delete right;
        token.~symbolTokenStream();
    }
};

struct parseTreeInArrayNode{
    size_t tokenId;
    symbolTable type;
    size_t strInArrayBeginId, strInArrayEndId;
    bool data;

};
struct parseTreeInArray{
    parseTreeInArrayNode * nodeList;
    size_t nodeListSize;
    charType * strArray;
    size_t strArraySize;
};


/**
 * convert search query to parse tree
 * @param searchKey input search key
 * @return parse tree and dist list
 */
//auto compiler(std::wstring searchKey) -> std::tuple<parseTree **, distList, size_t >;
auto compiler(std::wstring searchKey) -> std::tuple<parseTreeInArray *, distList, size_t >;

auto __host__ __device__ operatorPriority(symbolTable in) -> int;




#endif //NTOKEN_OPERATORPARSER_CUH
