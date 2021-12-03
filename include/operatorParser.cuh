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

};

struct symbolTokenStream{
    size_t id;
    symbolTable type;
    std::wstring str;
    auto __host__ __device__ operator=(symbolTokenStream const &right) {
        id = right.id;
        type = right.type;
        if(type == symbolTable::str){
            str = right.str;
        }
        return ;
    }
//    ~symbolTokenStream(){
//        if(type == symbolTable::str){
//            str.~array();
//        }
//    }

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


};
struct parseTreeInArray{
    parseTreeInArrayNode * nodeList;
    size_t nodeListSize;
    charType * strArray;
    size_t strArraySize;
    size_t deep;
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
