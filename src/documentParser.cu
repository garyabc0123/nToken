//
// Created by ascdc on 2021-11-29.
//



#include "documentParser.cuh"




auto getDocumentToken(std::wstring &input) -> documentToken{
    size_t first = 0;
    std::wstring charArray;
    std::vector<documentSentenceNode> document;
    std::vector<wordAndPartOfSpeechPair> wapos;
    document.reserve(100000);
    wapos.reserve(10000000);
    charArray.reserve(200000000);
    size_t counterSentence = 0;
    size_t counterToken = 0;

    std::wstring sentence;
    std::wstring tempStr;
    sentence.reserve(10000000);
    tempStr.reserve(1000);
    while (first < static_cast<size_t >(input.size())){
        size_t second =  input.find_first_of(L"\n", first);
        if(first != second){

            document.push_back(documentSentenceNode{
                    .id = counterSentence,
                    .nodeBegin =  counterToken
            });

            sentence = input.substr(first, second - first);
            size_t sentenceFirst = 0;
            while(sentenceFirst < sentence.size()){
                size_t sentenceSecond = sentence.find_first_of(L" ", sentenceFirst);
                if(sentenceFirst != sentenceSecond){
                    tempStr = sentence.substr(sentenceFirst, sentenceSecond - sentenceFirst);
                    if(tempStr.front() == L'(' && tempStr.back() == ')'){
                        wapos.back().partOfSpeech = std::stoi(tempStr.substr(1, tempStr.size() - 2));
                    }else{
                        wapos.push_back(wordAndPartOfSpeechPair{
                                .id = counterToken,
                                .begin = charArray.size()
                        });
                        charArray = charArray + tempStr;
                        wapos.back().end = charArray.size();
                        counterToken++;
                    }
                }

                if(sentenceSecond == std::wstring::npos)
                    break;
                sentenceFirst = sentenceSecond + 1;
            }

            counterSentence++;
            document.back().nodeEnd = counterToken;
        }
        if (second == std::wstring::npos)
            break;
        first = second + 1;
    }




    return documentToken{
        .token = thrust::universal_vector<wordAndPartOfSpeechPair>(wapos.begin(), wapos.end()),
        .sentence = thrust::universal_vector<documentSentenceNode>(document.begin(), document.end()),
        .word = thrust::universal_vector<charType>(charArray.begin(), charArray.end())
    };
}