//
// Created by ascdc on 2021-11-29.
//

#ifndef NTOKEN_FILEREADER_CUH
#define NTOKEN_FILEREADER_CUH

#include <locale>
#include <fstream>
#include <sstream>
#include <codecvt>
#include "basicDataTypeDefine.cuh"
#include "thrust/universal_vector.h"
#include "thrust/execution_policy.h"
#include "device_launch_parameters.h"

auto fileReader(char * path) -> std::wstring;

auto gpuUTF8Loader(std::vector<uint8_t> buffer) -> array<charType >;

auto gpuUTF8FileReader(char *path)-> array<charType>;

auto gpuUTF16FileReader(char *path) -> array<charType >;

#endif //NTOKEN_FILEREADER_CUH
