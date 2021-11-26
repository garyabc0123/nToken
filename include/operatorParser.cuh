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

enum struct symbolTable{
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
    array<charType> str;
};

using distList = size_t[];
struct parseTree{
    symbolTokenStream token;
    parseTree *left;
    parseTree *right;
};

/**
 * convert search query to parse tree
 * @param searchKey input search key
 * @return parse tree and dist list
 */
auto compiler(std::string searchKey) -> std::tuple<parseTree*[], distList >;




#endif //NTOKEN_OPERATORPARSER_CUH
