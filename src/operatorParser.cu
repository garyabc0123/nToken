//
// Created by ascdc on 2021-11-26.
//
#include "operatorParser.cuh"


/**
 * 詞法分析（英語：lexical analysis）是計算機科學中將字符序列轉換為標記（token）序列的過程。進行詞法分析的程序或者函數叫作詞法分析器
 * （lexical analyzer，簡稱lexer），也叫掃描器（scanner）。詞法分析器一般以函數的形式存在，供語法分析器調用。
 * @param input
 * @return token stream
 */
auto lecicalAnalyzer(std::wstring input) -> std::vector<symbolTokenStream>{

    std::vector<symbolTokenStream> ret;
    std::deque<wchar_t> buffer;
    size_t id = 0;
    for (size_t it = 0; it < input.size() ; it++){
        switch (input[it]) {
            case L'\\':
                buffer.push_back(input[it + 1]);//TODO have bug ex: \% {}
                it++;
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
                    ret.push_back(symbolTokenStream{id, symbolTable::str, std::wstring(buffer.begin(), buffer.end())});
                    id++;
                    buffer.clear();

                }
                wchar_t ch = input.at(it);
                ch += 0x100000;
                ret.push_back(symbolTokenStream{id, symbolTable::str, std::wstring(1, ch)});
                //offset to unicode "Private Use Area-B"
                id++;
                break;
            }

            case L' ':
            case L'\n':
            case L'\t':
                if(!buffer.empty()){
                    ret.push_back(symbolTokenStream{id, symbolTable::str, std::wstring(buffer.begin(), buffer.end())});
                    id++;
                    buffer.clear();

                }
                break;
            default:
                buffer.push_back(input[it]);
        }
    };
    if(!buffer.empty()){
        ret.push_back(symbolTokenStream{id, symbolTable::str, std::wstring(buffer.begin(), buffer.end())});
    }

    //0x100000: offset symbol to Unicode Private Use Area-B
    for(auto it = ret.begin() ; it != ret.end() ; it++){
        if(it->type == symbolTable::str && it->str.size() == 1 && it->str.front() > 0x100000){
            switch (it->str.front() - 0x100000) {
                case L'$' :
                    it->type = symbolTable::dollarSign;
                    it->str.clear();
                    break;
                case L'%' :
                    it->type = symbolTable::percentSign;
                    it->str.clear();
                    break;
                case L'|' :
                    it->type = symbolTable::verticalBar;
                    it->str.clear();
                    break;
                case L'!' :
                    it->type = symbolTable::exclamationMark;
                    it->str.clear();
                    break;
                case L'^' :
                    it->type = symbolTable::caret;
                    it->str.clear();
                    break;
                case L'[' :
                    it->type = symbolTable::squareBracketLeft;
                    it->str.clear();
                    break;
                case L']' :
                    it->type = symbolTable::squareBracketRight;
                    it->str.clear();
                    break;
                case L'{' :
                    it->type = symbolTable::curlyBracketLeft;
                    it->str.clear();
                    break;
                case L'}' :
                    it->type = symbolTable::curlyBracketRight;
                    it->str.clear();
                    break;
                default:
                    //do nothing
                    break;
            }


        }

    }
    return ret;

}
#ifdef NOONOD
auto infixToPrefix(std::vector<symbolTokenStream> input) -> std::vector<symbolTokenStream>{
    std::stack<symbolTokenStream> stack;
    std::deque<symbolTokenStream> output;
    for(int64_t it = input.size() - 1 ; it >= 0 ; it--){
        switch (input[it].type) {
            case symbolTable::str:
                output.push_back(input[it]);
                break;
            case symbolTable::curlyBracketRight:
                stack.push(input[it]);
                break;
            case symbolTable::curlyBracketLeft:
                for (;!stack.empty();){
                    auto temp = stack.top();
                    stack.pop();
                    if(temp.type == symbolTable::curlyBracketRight){
                        break;
                    }else{
                        output.push_front(temp);
                    }
                }
                break;
            case symbolTable::verticalBar:
            case symbolTable::caret:
                if(stack.empty()){
                    stack.push(input[it]);
                }else{
                    for(;!stack.empty();){
                        auto temp = stack.top();
                        if(temp.type == symbolTable::curlyBracketRight){
                            break;
                        }else if(operatorPriority(temp.type) < operatorPriority(input[it].type)){
                            output.push_front(temp);
                            stack.pop();
                        }else{
                            break;
                        }
                    }
                    stack.push(input[it]);
                }
                break;
            case symbolTable::percentSign:
            case symbolTable::exclamationMark:
            case symbolTable::dollarSign:
            default:
                output.push_front(input[it]);
                break;
        }
    }
    for(;!stack.empty();){
        auto temp = stack.top();
        output.push_front(temp);
        stack.pop();
    }
    return std::vector<symbolTokenStream>(output.begin(), output.end());
}
#endif
auto infixToPrefix(std::vector<symbolTokenStream> input) -> std::vector<symbolTokenStream>{
    std::stack<symbolTokenStream> stack;
    std::deque<symbolTokenStream> output;
    for(size_t it = input.size() - 1;  static_cast<size_t>(it) !=  static_cast<size_t>(-1) ; it--){
        switch (input[it].type) {
            case symbolTable::curlyBracketRight:
                stack.push(input[it]);
                break;
            case symbolTable::curlyBracketLeft:
                while(!stack.empty()){
                    if(stack.top().type == symbolTable::curlyBracketRight){
                        stack.pop();
                        break;
                    }
                    output.push_back(stack.top());
                    stack.pop();
                }
            default:
            {
                if(operatorPriority(input[it].type) < 100){
                    while(!stack.empty()){
                        if(stack.top().type == symbolTable::curlyBracketRight || operatorPriority(input[it].type) <
                                                                                         operatorPriority(stack.top().type)){
                            break;
                        }
                        output.push_back(stack.top());
                        stack.pop();
                    }
                    stack.push(input[it]);
                }else{
                    output.push_back(input[it]);
                }
            }
        }
    }

    while(!stack.empty()){
        output.push_back(stack.top());
        stack.pop();
    }
    std::reverse(output.begin(), output.end());
    return std::vector(output.begin(), output.end());
}


auto prefixToParseTree(std::vector<symbolTokenStream> &input, size_t begin, size_t size, parseTree * me) -> size_t{
    size_t retNext;
    for(size_t it = begin ; it < begin + size && it < input.size() ; it++){
        switch (input[it].type) {
            case symbolTable::dollarSign:
            case symbolTable::percentSign:
            case symbolTable::exclamationMark:
                me->token = input[it];
                me->left = new parseTree;
                it = prefixToParseTree(input, it+1, 1, me->left);
                retNext = it;
                break;
            case symbolTable::verticalBar:
            case symbolTable::caret:
                me->token = input[it];
                me->left = new parseTree;
                me->right = new parseTree;
                it = prefixToParseTree(input, it+1, 1, me->left);
                it = prefixToParseTree(input, it+1, 1, me->right);
                retNext = it;
                break;
            default:
                me->token = input[it];
                retNext = it;
                break;
        }
    }
    return retNext;
}


auto tokenStream2Tree(std::vector<symbolTokenStream> token) -> parseTree *{
    token = infixToPrefix(token);
    parseTree * treeRoot = new parseTree;
    prefixToParseTree(token, 0, token.size(), treeRoot);
    return treeRoot;
}
auto tokenStream2TreeInArray(std::vector<symbolTokenStream> token) -> parseTreeInArray{
    token = infixToPrefix(token);

    std::wstring charArray;
    std::vector<parseTreeInArrayNode> tArray(token.size());

    for(size_t i = 0 ; i < token.size() ; i++){
        tArray[i].type = token[i].type;
        tArray[i].tokenId = token[i].id;
        tArray[i].strInArrayBeginId = charArray.size();
        if(token[i].type == symbolTable::str){
            charArray += token[i].str;
        }
        tArray[i].strInArrayEndId = charArray.size();
    }
    parseTreeInArray ret;
    cudaMallocManaged(reinterpret_cast<void **>(&(ret.nodeList)), sizeof(parseTreeInArrayNode) * token.size());
    cudaMallocManaged(reinterpret_cast<void **>(&(ret.strArray)), sizeof(charType) * charArray.size());
    memcpy(ret.nodeList, tArray.data(), sizeof(parseTreeInArrayNode) * tArray.size());
    memcpy(ret.strArray, charArray.c_str(), sizeof(charType) * charArray.size());
    ret.nodeListSize = tArray.size();
    ret.strArraySize = charArray.size();



    return ret;

}


/**
 * 語法分析（英語：syntactic analysis，也叫 parsing）是根據某種給定的形式文法對由單詞序列（如英語單詞序列）構成的輸入文字進行分析並確定其語
 * 法結構的一種過程。
 * @param token
 * @return parseTree
 * @return dist
 */
auto syntaxDirectedTranslator(std::vector<symbolTokenStream> token) -> std::tuple<parseTreeInArray *, distList, size_t >{
    int nowState = 0;
    int curlyBegin, curlyEnd = 0;
    int squarBegin, squareEnd = 0;
    std::stack<symbolTable> stack;
    std::vector<parseTreeInArray > computeTupleTree;
    std::vector<size_t> dist;

    for(size_t it = 0 ; it < token.size() ; it++){
        switch (nowState){
            case 0:
                if(token[it].type == symbolTable::curlyBracketLeft){
                    curlyBegin = it;
                    nowState++;
                    stack.push(symbolTable::curlyBracketLeft);
                }
                break;
            case 1:
                if(token[it].type == symbolTable::curlyBracketLeft){
                    stack.push(symbolTable::curlyBracketLeft);
                }else if(token[it].type == symbolTable::curlyBracketRight){
                    if(stack.empty()){
                        throw "Synatex Error, loss {";
                    }else{
                        stack.pop();
                        if(stack.empty()){
                            curlyEnd = it;
                            nowState++;
                        }
                    }
                }
                break;
            case 2:
                if(token[it].type == symbolTable::squareBracketLeft){
                    squarBegin = it;
                    nowState++;
                }
                break;
            case 3:
                if(token[it].type == symbolTable::squareBracketRight){
                    squareEnd = it;
                    nowState = 0;
                    auto computeTree = tokenStream2TreeInArray(std::vector<symbolTokenStream>(token.begin()+curlyBegin+1, token.begin()+curlyEnd));
                    std::wstring distStr(token[squarBegin + 1].str);
                    size_t distThis = std::stoi(distStr) + 1;
                    dist.push_back(distThis);
                    computeTupleTree.push_back(computeTree);
                }
        }


    }
    parseTreeInArray * tempComputeTupleTree ;//= new parseTree*[computeTupleTree.size()];
    cudaError_t error;
    error = cudaMallocManaged(reinterpret_cast<void **>(&tempComputeTupleTree), sizeof(parseTreeInArray) * computeTupleTree.size());
    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }
    distList tempDistList;// = new size_t[computeTupleTree.size()];
    error = cudaMallocManaged(reinterpret_cast<void **>(&tempDistList), sizeof(size_t) * computeTupleTree.size());
    if(error != cudaSuccess){
        throw __FILE__ + std::to_string(__LINE__) + __func__  + cudaGetErrorName(error)+ "\n";
    }
    std::copy(computeTupleTree.begin(), computeTupleTree.end(), tempComputeTupleTree);
    std::copy(dist.begin(), dist.end(), tempDistList);
    return std::tuple<parseTreeInArray *, distList, size_t >{tempComputeTupleTree, tempDistList, computeTupleTree.size()};

}


auto compiler(std::wstring searchKey) -> std::tuple<parseTreeInArray *, distList, size_t >{
    auto token = lecicalAnalyzer(searchKey);
    return syntaxDirectedTranslator(token);

}

auto __host__ __device__ operatorPriority(symbolTable in) -> int{
    switch (in) {
        case symbolTable::squareBracketLeft:
        case symbolTable::squareBracketRight:
        case symbolTable::curlyBracketLeft:
        case symbolTable::curlyBracketRight:
                //[ ] { }
                return 3;
        case symbolTable::dollarSign:
        case symbolTable::percentSign:
                //$ %
                return 4;
        case symbolTable::exclamationMark:
                //!
                return 5;
        case symbolTable::caret:
                //^
                return 6;
        case symbolTable::verticalBar:
                //|
                return 7;
        case symbolTable::str:
        case symbolTable::boolean:
            return 100;
        case symbolTable::null:
            return 0;
        default:
            return 0;
        }
}

std::wostream& operator<<(std::wostream& out, const symbolTable value){
    static std::map<symbolTable, std::string> strings;
    if (strings.size() == 0){

#define INSERT_ELEMENT(p) strings[p] = #p
        INSERT_ELEMENT(symbolTable::null);
        INSERT_ELEMENT(symbolTable::dollarSign);
        INSERT_ELEMENT(symbolTable::percentSign);
        INSERT_ELEMENT(symbolTable::verticalBar);
        INSERT_ELEMENT(symbolTable::exclamationMark);
        INSERT_ELEMENT(symbolTable::caret);
        INSERT_ELEMENT(symbolTable::squareBracketLeft);
        INSERT_ELEMENT(symbolTable::squareBracketRight);
        INSERT_ELEMENT(symbolTable::curlyBracketLeft);
        INSERT_ELEMENT(symbolTable::curlyBracketRight);
        INSERT_ELEMENT(symbolTable::str);
        INSERT_ELEMENT(symbolTable::boolean);

#undef INSERT_ELEMENT
    }
    std::wstring temp(strings[value].begin(), strings[value].end());
    return out << temp;
}