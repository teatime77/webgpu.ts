import { assert, msg  } from "@i18n";

let unknownToken  = new Set();

export enum TokenType{
    unknown,

    // 識別子
    identifier,

    // クラス
    Class,

    // 数値
    Number,

    // 記号
    symbol,

    // 予約語
    reservedWord,

    type,

    // 文字列
    String,

    // 改行
    newLine,

    // End Of Text
    eot,

    // 行コメント
    lineComment,

    // ブロックコメント
    blockComment,
}

var SymbolTable : Array<string> = new  Array<string> (
    ",",
    ";",
    "(",
    ")",
    "{",
    "}",
    ":",
    "<",
    ">",
    "[",
    "]",
    "_",
    "=",
    ".",
    "+",
    "-",
    "*",
    "%",
    "/",

    "++",
    "--",

    "+=",
    "-=",
    "*=",
    "/=",
    "%=",

    "==",
    "!=",
    "<=",
    ">=",

    "&&",
    "||",

    "//",
    "=>",
    "->"
);
    
var KeywordMap : string[] = [
    "struct",
    "let",
    "var",
    "const",
    "fn",
    "uniform",
    "read",
    "read_write",
    "storage",
    "return",
    "if",
    "else",
    "while",
    "for",
    "of",
    "parallel",

    "@group",
    "@binding",
    "@location",
    "@compute",
    "@workgroup_size",
    "@builtin",
    "@vertex",
    "@compute",
    "@fragment",
];

const TypeName : string[] = [
    "mat4x4",
    "mat3x3",
    "vec4",
    "vec3",
    "vec2",
    "texture_2d",
    "array",

    "vec3u",
    "f32",
    "i32",
    "u32",
    "sampler"
]

export function isLetter(s : string) : boolean {
    return s.length == 1 && ("a" <= s && s <= "z" || "A" <= s && s <= "Z");
}

function isDigit(s : string) : boolean {
    return s.length == 1 && "0123456789".indexOf(s) != -1;
}

function isHexDigit(s : string) : boolean {
    return s.length == 1 && "0123456789ABCDEF".indexOf(s) != -1;
}

function isLetterOrDigit(s : string) : boolean {
    return isLetter(s) || isDigit(s) || s == "_";
}
    
export enum TokenSubType {
    unknown,
    integer,
    float,
    double,
}

export class Token{
    typeTkn:TokenType;
    subType:TokenSubType;
    text:string;
    charPos:number;

    public constructor(type : TokenType, sub_type : TokenSubType, text : string, char_pos : number){
        //msg("" + TokenType[type] + " " + TokenSubType[sub_type] + " " + text + " " + char_pos);
        this.typeTkn = type;
        this.subType = sub_type;
        this.text = text;
        this.charPos = char_pos;
    }
}

export function lexicalAnalysis(text : string) : Token[] {
    const tokens : Token[] = [];

    // 現在の文字位置
    let pos : number = 0;

    let line_num : number = 0;

    while(pos < text.length){
        
        // 空白をスキップします。
        for ( ; pos < text.length && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\r'); pos++);

        if (text.length <= pos) {
            // テキストの終わりの場合

            break;
        }

        const start_pos = pos;

        var token_type = TokenType.unknown;
        var sub_type : TokenSubType = TokenSubType.unknown;

        // 現在位置の文字
        var ch1 : string = text[pos];

        // 次の文字の位置。行末の場合は'\0'
        var ch2 : string;

        if (pos + 1 < text.length) {
            // 行末でない場合

            ch2 = text[pos + 1];
        }
        else {
            // 行末の場合

            ch2 = '\0';
        }

        if(ch1 == '\n'){

            token_type = TokenType.newLine;
            line_num++;
            pos++;
        }
        else if(ch1 + ch2 == "//"){
            for (pos += 2 ; pos < text.length && text[pos] != "\n"; pos++);
            continue;
        }
        else if (isLetter(ch1) || ch1 == '@' && isLetter(ch2)) {
            // 識別子の最初の文字の場合

            // 識別子の文字の最後を探します。識別子の文字はユニコードカテゴリーの文字か数字か'_'。
            for (pos++; pos < text.length && isLetterOrDigit(text[pos]); pos++);

            // 識別子の文字列
            var name : string = text.substring(start_pos, pos);

            if (KeywordMap.indexOf(name) != -1) {
                // 名前がキーワード辞書にある場合

                token_type = TokenType.reservedWord;
            }
            else if (TypeName.indexOf(name) != -1) {
                // 名前が型の辞書にある場合

                token_type = TokenType.type;
            }
            else {
                // 名前がキーワード辞書にない場合

                token_type = TokenType.identifier;
            }
        }
        else if (isDigit(ch1) || ch1 == "-" && isDigit(ch2)) {
            // 数字の場合

            token_type = TokenType.Number;

            // 10進数の終わりを探します。
            for (pos++; pos < text.length && isDigit(text[pos]); pos++);

            if (pos < text.length && text[pos] == '.') {
                // 小数点の場合

                pos++;

                // 10進数の終わりを探します。
                for (; pos < text.length && isDigit(text[pos]); pos++);

                if(text[pos] == "e"){
                    pos++;
                    if(text[pos] == "-"){
                        pos++;
                    }

                    // 10進数の終わりを探します。
                    for (; pos < text.length && isDigit(text[pos]); pos++);
                }

                sub_type = TokenSubType.float;
            }
            else {

                if (pos < text.length && text[pos] == 'u'){
                    pos++;
                }

                sub_type = TokenSubType.integer;
            }
        }
        else if(ch1 == '#'){
            token_type = TokenType.Number;
            sub_type = TokenSubType.integer;

            // 10進数の終わりを探します。
            for (pos++; pos < text.length && isHexDigit(text[pos]); pos++);

            assert(pos - start_pos == 7);
        }
        else if(ch1 == '"'){
            token_type = TokenType.String;
            pos = text.indexOf('"', pos + 1);
            assert(pos != -1);
            pos++;
        }
        else if (SymbolTable.indexOf("" + ch1 + ch2) != -1) {
            // 2文字の記号の表にある場合

            token_type = TokenType.symbol;
            pos += 2;
        }
        else if (SymbolTable.indexOf("" + ch1) != -1) {
            // 1文字の記号の表にある場合

            token_type = TokenType.symbol;
            pos++;
        }
        else {
            // 不明の文字の場合

            token_type = TokenType.unknown;
            pos++;

            const s = text.substring(start_pos, pos);
            if(! unknownToken.has(s)){
                unknownToken.add(s);
                msg(`不明 [${s}]`);
            }
        }

        // 字句の文字列を得ます。
        var word : string;
        if(token_type == TokenType.String){
            word = text.substring(start_pos + 1, pos - 1);
        }
        else{
            word = text.substring(start_pos, pos);
        }

        const token = new Token(token_type, sub_type, word, start_pos);

        tokens.push(token);
    }

    tokens.push(eotToken);

    return tokens;
}

const eotToken : Token = new Token(TokenType.eot, TokenSubType.unknown, "", -1);
