//
// Created by ascdc on 2021-11-29.
//

#include "fileReader.cuh"
#include <sstream>
#define OUTPUTMEMALLOCSIZE

auto fileReader(char * path) -> std::wstring {
    std::wfstream wfs;
    std::wifstream wif(path);
    wif.imbue(std::locale(std::locale{"zh_TW.utf8"}, new std::codecvt_utf8<wchar_t>));
    std::wstringstream wss;
    wss << wif.rdbuf();
    return wss.str();
}

__global__ void isUTF8FirstByte(array<uint8_t> devBinFile, array<bool> isFirst){
    size_t idx = threadIdx.x + blockIdx.x  * blockDim.x;
    if (idx >= devBinFile.size)
        return;
    bool first = devBinFile.ptr[idx] >> 7;
    if(!first){
        isFirst.ptr[idx] = true;
    }else{
        if((devBinFile.ptr[idx] & 0xC0) == 0xC0){
            isFirst.ptr[idx] = true;
        }else{
            isFirst.ptr[idx] = false;

        }
    }

}
__global__ void convertUTF82Unicode(array<bool> isFirst, array<size_t> scanFirst, array<uint8_t> devBinFile, array<charType> out){
    size_t idx = threadIdx.x + blockIdx.x  * blockDim.x;
    if (idx >= devBinFile.size)
        return;
    if(isFirst.ptr[idx] == 0){
        return;
    }
    size_t saveAddr = scanFirst.ptr[idx];
    uint8_t firstByte = devBinFile.ptr[idx];
    if(firstByte < 128){
        out.ptr[saveAddr] = firstByte;
        return;
    }
    uint32_t high_bit_mask = (1 << 6) -1;
    uint32_t high_bit_shift = 0;
    uint32_t total_bits = 0;
    const uint32_t other_bits = 6;
    charType charcode = 0;
    uint8_t it = 1;
    while((firstByte & 0xC0) == 0xC0){
        firstByte <<= 1;
        firstByte &= 0xff;
        total_bits += 6;
        high_bit_mask >>= 1;
        high_bit_shift++;
        charcode <<= other_bits;
        charcode |= static_cast<uint32_t>(devBinFile.ptr[idx + it]) & ((1 <<other_bits) - 1);
        it++;
        if(idx + it > devBinFile.size)
            break;
    }
    charcode |= ((static_cast<uint32_t>(firstByte) >> high_bit_shift) & high_bit_mask) << total_bits;
    out.ptr[saveAddr] = charcode;
}

/**
 *
 * @param path = file path
 * @return device pointer to unicode array
 */
auto gpuUTF8FileReader(char * path) -> array<charType>{
    std::ifstream myFile;
    cudaSetDevice(0);
    myFile.open (path, std::ios::in | std::ios::binary);
    std::vector<uint8_t> buffer(std::istreambuf_iterator<char>(myFile), {});
    cudaError_t error;

    array<uint8_t> devBinFile;
    devBinFile.size = buffer.size();
    error = cudaMallocManaged(reinterpret_cast<void**>(&(devBinFile.ptr)), sizeof(uint8_t) * buffer.size());
#ifdef OUTPUTMEMALLOCSIZE
    std::wcout << "devBinFile" << sizeof(uint8_t) * buffer.size() << std::endl;
#endif
    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }
    memcpy(devBinFile.ptr, buffer.data(), sizeof(uint8_t) * buffer.size());
    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }


    array<bool> isFirst;
    isFirst.size = buffer.size();
    error = cudaMallocManaged(reinterpret_cast<void**>(&(isFirst.ptr)), sizeof(bool) * buffer.size());
#ifdef OUTPUTMEMALLOCSIZE
    std::wcout << "isFirst" << sizeof(bool) * buffer.size() << std::endl;
#endif
    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }


    array<size_t> scanFirst;
    scanFirst.size = buffer.size();
    error = cudaMallocManaged(reinterpret_cast<void**>(&(scanFirst.ptr)), sizeof(size_t) * buffer.size());
#ifdef OUTPUTMEMALLOCSIZE
    std::wcout << "scanFirst" << sizeof(size_t) * buffer.size() << std::endl;
#endif
    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }
    isUTF8FirstByte<<<devBinFile.size / 1024 + 1, 1024>>>(devBinFile, isFirst);
    cudaDeviceSynchronize();
    std::cout << __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error) + "\n";
    error = cudaGetLastError();
    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }
    thrust::exclusive_scan(thrust::device, isFirst.ptr, isFirst.ptr + isFirst.size, scanFirst.ptr, size_t(0));

    std::cout << __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error) + "\n";
    array<charType> devUnicodeFile;
    memcpy(&(devUnicodeFile.size), scanFirst.ptr + scanFirst.size - 1, sizeof(size_t));
    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }
    error =  cudaMallocManaged(reinterpret_cast<void**>(&(devUnicodeFile.ptr)), sizeof(charType) * devUnicodeFile.size);

    if(error != cudaSuccess){


        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }
    std::cout << __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error) + "\n";

    convertUTF82Unicode<<<devBinFile.size / 1024 + 1, 1024>>>(isFirst, scanFirst, devBinFile, devUnicodeFile);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }
    std::cout << __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error) + "\n";


    cudaFree(devBinFile.ptr);
    cudaFree(isFirst.ptr);
    cudaFree(scanFirst.ptr);



    return devUnicodeFile;
}