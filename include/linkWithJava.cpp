//
// Created by ascdc on 2021-12-27.
//


#include "linkWithJava.h"
JNIEXPORT jcharArray JNICALL Java_nToken_callNToken
        (JNIEnv *env , jobject obj, jbyteArray text, jbyteArray query){
    jsize textLen = env->GetArrayLength(text);
    jbyte * textPtr = env->GetByteArrayElements(text, 0);
    std::vector<uint8_t> textVec(textPtr, textPtr + textLen);

    jsize queryLen = env->GetArrayLength(query);
    jbyte * queryPtr = env->GetByteArrayElements(text, 0);
    std::string queryStr(queryPtr, queryPtr + queryLen);

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::wstring queryWstring = converter.from_bytes(reinterpret_cast<char *>(queryPtr), reinterpret_cast<char *>(queryPtr + queryLen));

   std::wcout << queryWstring << std::endl;

}