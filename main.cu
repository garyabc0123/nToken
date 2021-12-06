#include <iostream>
#include "nToken.cuh"
int main(int argc, char ** argv) {
    std::cout << __cplusplus << std::endl;
    std::cout << (__cplusplus > 201703L) << std::endl;
    try{
        //nToken(argv[1], argv[2]);
        gpuUTF8FileReader(argv[1]);
    } catch (std::string e) {
        std::cout << e << std::endl;
    }




    return 0;
}

void func(){

}