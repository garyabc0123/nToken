//
// Created by ascdc on 2021-11-26.
//
#include "operatorParser.cuh"
auto lecicalAnalyzer(std::string input){
    try{
        std::vector<symbolTokenStream> ret;
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        std::wstring wStr = converter.from_bytes(input);
        std::deque<wchar_t> buffer;
        size_t id = 0;
        for (size_t it = 0; it < wStr.size() ; it++){
            switch (wStr[it]) {
                case L'\\':
                    buffer.push_back(wStr[it + 1]);
                    break;
                case L'$':
                case L'%':
                case L'|':
                case L'!':
                case L'^':
                case L'[':
                case L']':
                case L'{':
                case L'}':
                {
                    if(!buffer.empty()){
                        auto tempPtr = buffer.begin().operator*();
                        array<charType> tempArray(tempPtr, buffer.size());

                        ret.push_back(symbolTokenStream{id, symbolTable::str, tempArray});
                        id++;
                        buffer.clear();

                    }
                    array<charType> tempArray(1, 1);
                    tempArray[0] = static_cast<charType>(wStr[it]);
                    ret.push_back(symbolTokenStream{id, symbolTable::str, tempArray});
                    id++;
                    break;
                }

                case L' ':
                case L'\n':
                case L'\t':
                    if(!buffer.empty()){
                        array<charType> tempArray(buffer.begin().operator*(), buffer.size());
                        ret.push_back(symbolTokenStream{id, symbolTable::str, tempArray});
                        id++;
                        buffer.clear();

                    }
                    break;
                default:
                    buffer.push_back(wStr[it]);
            }
        };
        if(!buffer.empty()){
            array<charType> tempArray(buffer.begin().operator*(), buffer.size());
            ret.push_back(symbolTokenStream{id, symbolTable::str, tempArray});
        }

        //TODO Classification symbol
        for(auto it = ret.begin() ; it != ret.end() ; it++){
            if(it->type == symbolTable::str && it->str.size == 1){
                bool dist = false;
                switch (it->str[0]) {
                    case L'$':

                        break;
                    default:



                        break;
                }
                if(dist)
                    it->str.~array();


            }

        }

    }catch (const char * message){
        throw __FILE__ + std::to_string(__LINE__) + __func__ + "\n" + std::string(message);
    }

}


auto compiler(std::string searchKey) -> std::tuple<parseTree*[], distList >{
    return std::tuple<parseTree*[], distList>();
}
