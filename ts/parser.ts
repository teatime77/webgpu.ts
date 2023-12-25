namespace webgputs {

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
    "/",

    "//",
    "->"
);
    
var KeywordMap : string[] = [
    "struct",
    "var",
    "fn",
    "uniform",
    "read",
    "read_write",
    "storage",

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
    "u32",
    "sampler"
]

function isLetter(s : string) : boolean {
    return s.length === 1 && ("a" <= s && s <= "z" || "A" <= s && s <= "Z");
}

function isDigit(s : string) : boolean {
    return s.length == 1 && "0123456789".indexOf(s) != -1;
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

// export enum Builtin {
//     global_invocation_id,
//     position
// }

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

function makeGPUVertexAttributes(vars : Variable[]) : GPUVertexAttribute[] {
    const attributes : GPUVertexAttribute[] = [];
    let offset : number = 0;
    for(const va of vars){
        const fmt = va.type.format() as GPUVertexFormat;

        const attr : GPUVertexAttribute = {
            shaderLocation: va.mod.location!,
            offset: offset,
            format: fmt
        };
        attributes.push(attr);

        offset += va.type.size();
    }

    return attributes;
}

export class Module {
    module : GPUShaderModule;
    structs : Struct[] = [];
    vars : Variable[] = [];
    fns : Function[] = [];
    instances : number[] = [];

    constructor(text : string){
        this.module = g_device.createShaderModule({ code: text });

        const tokens = lexicalAnalysis(text);

        const parser = new Parser(tokens);
        parser.parse(this);
    }

    dump(){
        this.structs.forEach(x => x.dump());
        this.vars.forEach(x => msg(`${x.str()};`))
        this.fns.forEach(x => msg(`${x.str()}`))
    }


    makeVertexBufferLayouts(is_instance : boolean) : GPUVertexBufferLayout[] {
        console.assert(this.fns.length == 1);
        const main = this.fns[0];

        const instance_vars = main.args.filter(x => x.name == "pos" );
        const vertex_vars   = main.args.filter(x => ! instance_vars.includes(x) );
        
        const vertex_buffer_layouts : GPUVertexBufferLayout[] = [
            {
                arrayStride: sum( vertex_vars.map(x => x.type.size()) ),
                stepMode: 'vertex',
                attributes: makeGPUVertexAttributes(vertex_vars)
            }            
        ];
    
        if(is_instance){
    
            vertex_buffer_layouts.push({
                arrayStride: sum( instance_vars.map(x => x.type.size()) ),    
                stepMode: 'instance',    
                attributes: makeGPUVertexAttributes(instance_vars)
            });
        }
    
        return vertex_buffer_layouts;
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

            line_num++;
            pos++;
            continue;
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
        else if (isDigit(ch1)) {
            // 数字の場合

            token_type = TokenType.Number;

            // 10進数の終わりを探します。
            for (; pos < text.length && isDigit(text[pos]); pos++);

            sub_type = TokenSubType.integer;
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
        var word : string = text.substring(start_pos, pos);

        const token = new Token(token_type, sub_type, word, start_pos);

        tokens.push(token);
    }

    tokens.push(eotToken);

    return tokens;
}


class Modifier {
    group : number | undefined;
    binding : number | undefined;
    location : number | undefined;
    workgroup_size : number[] | undefined;
    builtin : string | undefined;
    fnType : string | undefined; 

    str() : string {
        let s = "";

        if(this.group != undefined){

            s += ` group(${this.group})`
        }

        if(this.binding != undefined){

            s += ` binding(${this.binding})`
        }

        if(this.location != undefined){

            s += ` location(${this.location})`
        }

        if(this.workgroup_size != undefined){

            s += ` workgroup_size(${this.workgroup_size})`
        }

        if(this.builtin != undefined){

            s += ` builtin(${this.builtin})`
        }

        if(this.fnType != undefined){

            s += ` ${this.fnType}`
        }

        return s;
    }
}

function primitiveTypeSize(primitive : string) : number{
    switch(primitive){
    case "vec3u": return 3 * 4;
    case "f32"  : return 4;
    case "u32"  : return 4;
    case "sampler":
    default:
        error(`unknown type size ${primitive}`);
        return NaN;
    }
}

class Type {
    mod : Modifier;
    aggregate : string | undefined;
    primitive : string;

    constructor(mod : Modifier, aggregate : string | undefined, primitive : string){
        this.mod = mod;
        this.aggregate = aggregate;
        this.primitive = primitive;
    }

    str() : string {
        if(this.aggregate != undefined){
            return `${this.mod.str()} ${this.aggregate}<${this.primitive}>`;
        }
        else{
            return `${this.mod.str()} ${this.primitive}`;
        }
    }

    size() : number {
        const primitive_size = primitiveTypeSize(this.primitive);
        if(this.aggregate == undefined){
            return primitive_size;
        }
        else{
            switch(this.aggregate){
            case "mat4x4" : return 4 * 4 * primitive_size;
            case "mat3x3" : return 3 * 3 * primitive_size;
            case "vec4"   : return     4 * primitive_size;
            case "vec3"   : return     3 * primitive_size;
            case "vec2"   : return     2 * primitive_size;

            case "texture_2d":
            case "array" :
            default:
                error(`unknown aggregate size ${this.aggregate}`);
                return NaN;
            }
        }
    }

    format() : string {
        let fmt : string;

        switch(this.primitive){
            case "f32"  : fmt = "float32"; break;
            case "u32"  : fmt = "uint32"; break;
            default:
                error(`unknown type format ${this.primitive}`);
                return "";
        }

        switch(this.aggregate){
        case undefined: return fmt;
        case "vec2"   : return fmt + "x2";
        case "vec3"   : return fmt + "x3";
        case "vec4"   : return fmt + "x4";
        }
        
        error(`unknown type format ${this.aggregate}`);
        return "";
    }
}

class Struct {
    mod : Modifier;
    name : string;
    members : Variable[] = [];

    constructor(mod : Modifier, name : string){
        this.mod = mod;
        this.name = name;
    }

    dump(){
        msg(`${this.mod.str()} struct ${this.name}{`);
        for(const va of this.members){
            msg(`    ${va.str()};`);
        }
        msg("}")
    }
}

class Variable {
    mod : Modifier;
    name : string;
    type : Type;

    constructor(mod : Modifier, name : string, type : Type){
        this.mod = mod;
        this.name = name;
        this.type = type;
    }

    str() : string {
        return `${this.mod.str()} ${this.name} : ${this.type.str()}`
    }
}

class Function {
    mod : Modifier;
    name : string;
    args : Variable[] = [];
    type : Type | undefined;

    constructor(mod : Modifier, name : string){
        this.mod = mod;
        this.name = name;
    }

    str() : string {
        const vars_s = this.args.map(x => x.str()).join(", ");
        const type_s = this.type == undefined ? "" : `-> ${this.type.str()}`;
        
        return `${this.mod.str()} fn ${this.name}(${vars_s}) ${type_s}`;
    }
}

export class Parser {
    tokenPos:number = 0;
    inFn : boolean = false;

    tokenList : Token[];

    constructor(tokens : Token[]){
        this.tokenList = tokens;
        this.tokenPos = 0;
    }

    get currentToken() : Token {
        if(this.tokenPos < this.tokenList.length){
            return this.tokenList[this.tokenPos];
        }
        else{
            return eotToken;
        }
    }

    currentText() : string {
        return this.currentToken.text;
    }

    advance(){
        this.tokenPos++;
    }

    readText(text : string){
        if(this.currentToken.text != text){
            error(`read token : ${this.currentToken.text} : ${text} is expected.`)
        }

        this.advance();
    }

    readReserved(){
        if(this.currentToken.typeTkn != TokenType.reservedWord){
            error(`reserved word is expected.`)
        }

        const id = this.currentToken.text;
        this.advance();

        return id;
    }

    readToken(type : TokenType){
        if(this.currentToken.typeTkn != type){
            error(`identifier is expected.`)
        }

        const text = this.currentToken.text;
        this.advance();

        return text;
    }

    readId(){
        return this.readToken(TokenType.identifier);
    }

    readModifiers() : Modifier {
        const mod = new Modifier();

        while(true){
            switch(this.currentToken.text){
                case "@group":
                    mod.group = this.readAttribute();
                    break;
    
                case "@binding":
                    mod.binding = this.readAttribute();
                    break;
    
                case "@location":
                    mod.location = this.readAttribute();
                    break;
    
                case "@workgroup_size":
                    mod.workgroup_size = this.readAttributeList();
                    break;

                case "@compute":
                case "@vertex":
                case "@fragment":
                    mod.fnType = this.currentToken.text;
                    this.advance();
                    break;

                case "@builtin":
                    this.advance();
                    this.readText("(");
                    mod.builtin = this.readId();
                    this.readText(")");
                    break;

                default:
                    return mod;
            }
        }
    }

    readType() : Type {
        const mod = this.readModifiers();

        switch(this.currentToken.text){
            case "mat4x4":
            case "mat3x3":
            case "vec4":
            case "vec3":
            case "vec2":
            case "array":
            case "texture_2d":
                const aggregate = this.readToken(TokenType.type);
                this.readText("<");
                const primitive = this.readToken(TokenType.type);
                this.readText(">");
                return new Type(mod, aggregate, primitive);
        }

        const primitive = this.readToken(TokenType.type);
        return new Type(mod, undefined, primitive);

    }

    readVariable() : Variable {
        const mod = this.readModifiers();
        const name = this.readId();
        this.readText(":");
        const type = this.readType();
        
        return new Variable(mod, name, type);

    }

    structDeclaration(mod : Modifier) : Struct {
        this.advance();

        const name = this.readId();
        const struct = new Struct(mod, name);

        // change typeTkn
        this.tokenList.filter(x => x.text == name).forEach(x => x.typeTkn = TokenType.type);

        this.readText("{");

        while(true){
            if(this.currentToken.text == "}"){
                break;
            }

            const variable  = this.readVariable();
            struct.members.push(variable);

            if(this.currentToken.text == ","){
                this.readText(",");
            }
            else{
                break;
            }
        }
        this.readText("}");

        return struct;
    }

    readBufferVar(mod : Modifier) : Variable {
        this.readText("var");

        if(this.currentText() == "<"){

            this.readText("<");
            while(true){
                const buf_attr = this.readReserved();

                if(this.currentText() == ","){
                    this.readText(",");
                }
                else{
                    break;
                }
            }
            this.readText(">");
        }

        const name = this.readId();
        this.readText(":");
        const type = this.readType();
        
        const buf_var = new Variable(mod, name, type);

        this.readText(";");

        return buf_var;
    }

    readInt(){
        if(this.currentToken.typeTkn != TokenType.Number){
            error("number is missing.")
        }
        const n = parseInt(this.currentToken.text);

        this.advance();

        return n;
    }

    readAttribute() : number {
        this.advance();

        this.readText("(");
        const value = this.readInt();
        this.readText(")");

        return value;
    }

    readAttributeList() : number[] {
        this.advance();

        this.readText("(");

        const values : number[] = [];

        while(true){
            const n = this.readInt();
            values.push( n );

            if(this.currentToken.text == ","){

                this.readText(",");
            }
            else{
                break;
            }
        }

        this.readText(")");

        return values;
    }

    readArgs(){

    }

    readFn(mod : Modifier) : Function {
        this.readText("fn");

        const name = this.readId();

        const fn = new Function(mod, name);

        this.readText("(");

        while(true){
            if(this.currentToken.text == ")"){
                break;
            }

            const variable  = this.readVariable();
            fn.args.push(variable);

            if(this.currentToken.text == ","){
                this.readText(",");

            }
            else{
                break;
            }
        }
        this.readText(")");

        if(this.currentToken.text == "->"){

            this.readText("->");

            fn.type = this.readType();
        }

        this.readText("{");
        let nest = 1;
        while(true){
            if(this.currentToken.text == "{"){
                nest++;
            }
            else if(this.currentToken.text == "}"){
                nest--;
                if(nest == 0){
                    this.readText("}");
                    break;
                }
            }
            this.advance();
        }

        return fn;
    }

    parse(module : Module){
        while(this.currentToken != eotToken){
            const mod = this.readModifiers();

            switch(this.currentToken.text){
                case "struct":
                    const struct = this.structDeclaration(mod);
                    module.structs.push(struct);
                    break;
    
                case "var":
                    const buf_var = this.readBufferVar(mod);
                    module.vars.push(buf_var);
                    break;

                case "fn":
                    const fn = this.readFn(mod);
                    module.fns.push(fn);
                    break;

                default:
                    error(`parse error [${this.currentToken.text}]`)
            }
        }
    }


    isEoT() : boolean {
        return this.currentToken == eotToken;
    }


    appended(){
        if(this.tokenPos == 0){
            return false;
        }

        const prev_token = this.tokenList[this.tokenPos - 1];

        return prev_token.charPos + prev_token.text.length == this.currentToken.charPos;
    }

    isArgs(){
        if(! this.appended()){
            return false;
        }

        return this.currentToken.text == '[' || this.currentToken.text == '{';
    }
}

const eotToken : Token = new Token(TokenType.eot, TokenSubType.unknown, "", -1);

export async function parseAll(){
    const shader_names = [
        "compute",
        "demo",
        "depth-frag",
        "instance-vert",
        "shape-vert",
        "sprite",
        "texture-frag",
        "texture-vert",
        "updateSprites",
    ];

    for(const shader_name of shader_names){
        msg(`\n------------------------------ ${shader_name}`)
        const text = await fetchText(`../wgsl/${shader_name}.wgsl`);
        const mod = new Module(text);
        mod.dump();
    }
}

}
