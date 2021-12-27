#include <iostream>
#include "nToken.cuh"
int main(int argc, char ** argv) {
    std::wcout.imbue(std::locale{"zh_TW.UTF8"});
    std::locale::global(std::locale{"zh_TW.UTF8"});
    std::wcout << __cplusplus << std::endl;
    std::wcout << (__cplusplus > 201703L) << std::endl;
    try{
        nToken n(argv[1], argv[2]);
        n.go();
    } catch (std::string e) {
        std::wcout << std::wstring(e.begin(), e.end()) << std::endl;
    }




    return 0;
}

void func(){
//    std::cout << __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error) + "\n";
}