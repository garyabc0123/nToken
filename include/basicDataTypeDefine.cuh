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


template <typename T>
struct array{
    T *ptr;
    size_t size;
};

#endif //NTOKEN_BASICDATATYPEDEFINE_CUH
