//
// Created by ascdc on 2021-11-22.
//

#ifndef NTOKEN_BASICDATATYPEDEFINE_CUH
#define NTOKEN_BASICDATATYPEDEFINE_CUH

#include <string>
#include <cstdint>
#include "stdint.h"


using charType = uint32_t;
using partWordOfSpeechType = uint16_t;

#ifdef __WIN32 // I hope never use it;
#include <windows.h>
#define wchar_t uint32_t
#define size_t uint64_t

#endif

template <typename T>
struct array{
    T *ptr;
    size_t size;
};

#endif //NTOKEN_BASICDATATYPEDEFINE_CUH
