//
// Created by ascdc on 2021-11-29.
//

#ifndef NTOKEN_FILEREADER_CUH
#define NTOKEN_FILEREADER_CUH

#include <string>
#include <locale>
#include <fstream>
#include <codecvt>
#include "basicDataTypeDefine.cuh"
#include "thrust/scan.h"
#include "thrust/execution_policy.h"

auto fileReader(char * path) -> std::wstring;

auto gpuUTF8FileReader(char *path)-> array<charType>;


#endif //NTOKEN_FILEREADER_CUH
