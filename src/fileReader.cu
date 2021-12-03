//
// Created by ascdc on 2021-11-29.
//

#include <sstream>
#include "fileReader.cuh"
auto fileReader(char * path) -> std::wstring {
    std::wfstream wfs;
    std::wifstream wif(path);
    wif.imbue(std::locale(std::locale{"zh_TW.utf8"}, new std::codecvt_utf8<wchar_t>));
    std::wstringstream wss;
    wss << wif.rdbuf();
    return wss.str();
}