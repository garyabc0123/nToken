//
// Created by ascdc on 2021-12-27.
//
#include <vector>
#include <cstdint>
#include <iostream>
#include <locale>
#include <codecvt>
#include <string>
#include "linkWithJava.h"
#include "nToken.cuh"
/**
 *
 * @param env
 * @param obj
 * @param text
 * @param query
 * @return
 * input text, query are UTF-8 coding
 */
JNIEXPORT jbyteArray JNICALL Java_nToken_nToken_callNToken
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

   nToken temp(textVec, queryWstring);
   env->ReleaseByteArrayElements(text,textPtr, 0);
   env->ReleaseByteArrayElements(query, queryPtr, 0);
   temp.go();
   auto json = temp.getJSON();

    jbyteArray ret = env->NewByteArray(json.size());
   env->SetByteArrayRegion(ret, 0, json.size(), (jbyte *)(json.c_str()));

}