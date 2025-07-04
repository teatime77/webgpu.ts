namespace webgpu_ts {

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
    "%",
    "/",
    "&&",
    "||",

    "//",
    "->"
);
    
var KeywordMap : string[] = [
    "struct",
    "var",
    "const",
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
    "i32",
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

function makeGPUVertexVarsAttributes(vars : Variable[]) : GPUVertexAttribute[] {
    const attributes : GPUVertexAttribute[] = [];
    let offset : number = 0;
    for(const va of vars){
        const attr : GPUVertexAttribute = {
            shaderLocation: va.mod.location!,
            offset: offset,
            format: va.type.format() as GPUVertexFormat
        };
        attributes.push(attr);

        offset += va.type.size();
    }

    return attributes;
}

function makeGPUInstanceVarsAttributes(vars : Variable[], map : Map<Variable, Field>) : GPUVertexAttribute[] {
    const attributes : GPUVertexAttribute[] = [];
    for(const va of vars){
        const field = map.get(va)!;
        assert(field != undefined);

        const attr : GPUVertexAttribute = {
            shaderLocation: va.mod.location!,
            offset: field.offset(),
            format: va.type.format() as GPUVertexFormat
        };

        attributes.push(attr);
    }

    return attributes;
}

export class Module {
    module : GPUShaderModule;
    structs : Struct[] = [];
    vars : Variable[] = [];
    fns : Function[] = [];
    instances : number[] = [];
    vertexSlot : number = -1;
    instanceSlot : number = -1;

    constructor(text : string){
        this.module = makeShaderModule(text);

        const tokens = lexicalAnalysis(text);

        const parser = new Parser(tokens);
        parser.parse(this);
    }

    dump(){
        this.structs.forEach(x => x.dump());
        this.vars.forEach(x => msg(`${x.str()};`))
        this.fns.forEach(x => msg(`${x.str()}`))
    }

    getUniformVar() : Variable {
        const uniform_var = this.vars.find(x => x.mod.usage == BufferUsage.uniform);
        if(uniform_var == undefined){
            throw new Error("no uniform var");
        }

        return uniform_var;;
    }

    uniformSize() : number {
        const uniform_var = this.getUniformVar();

        if(uniform_var.type instanceof Struct){

            return uniform_var.type.size();
        }
        else{

            throw new Error("no uniform type");
        }
    }
    

    makeVertexBufferLayouts(compute : ComputePipeline | null) : GPUVertexBufferLayout[] {
        
        const main = this.fns.find(x => x.mod.fnType == "@vertex");
        if(main == undefined){
            throw new Error(`no vert main `);
        }
        const map = new Map<Variable, Field>();

        let struct: Struct | undefined;
        let instance_var_names : string[];
        if(compute == null){
            instance_var_names = [];
        }
        else{
            instance_var_names = compute.varNames;
            const output_vars = compute.compModule.vars.filter(x => x.mod.usage == BufferUsage.storage_read_write);
            assert(output_vars.length == 1);
            const output_var  = output_vars[0];

            struct = compute.compModule.structs.find(x => x.typeName == output_var.type.typeName);
            if(struct == undefined){
                throw new MyError();
            }

            for(const arg of main.args){
                const member = struct.members.find(x => x.name == arg.name);
                if(member != undefined){
                    msg(`arg-mem:${arg.name}`);
                    map.set(arg, member);
                }
            }
        }

        const builtin_var_names = [ "vertex_index" ];
        const instance_vars = main.args.filter(x => instance_var_names.includes(x.name) );
        assert(instance_vars.length == map.size && instance_vars.every(x => map.has(x)));

        const vertex_vars   = main.args.filter(x => (! instance_vars.includes(x) && ! builtin_var_names.includes(x.name)) );
        
        const vertex_step_layout : GPUVertexBufferLayout = {
            arrayStride: sum( vertex_vars.map(x => x.type.size()) ),
            stepMode: 'vertex',
            attributes: makeGPUVertexVarsAttributes(vertex_vars)
        };
    
        assert((compute == null) == (instance_var_names.length == 0));
        if(compute == null){
            this.vertexSlot = 0;
            return [ vertex_step_layout ];
        }
        else{
    
            const instance_step_layout  : GPUVertexBufferLayout = {
                arrayStride: sum( instance_vars.map(x => x.type.size()) ),    
                stepMode: 'instance',    
                attributes: makeGPUInstanceVarsAttributes(instance_vars, map)
            };
            assert(instance_step_layout.arrayStride == struct!.size());


            this.instanceSlot = 0;
            this.vertexSlot = 1;
            return [ instance_step_layout, vertex_step_layout ];
        }
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

            if (pos < text.length && text[pos] == '.') {
                // 小数点の場合

                pos++;

                // 10進数の終わりを探します。
                for (; pos < text.length && isDigit(text[pos]); pos++);

                sub_type = TokenSubType.float;
            }
            else {

                sub_type = TokenSubType.integer;
            }
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

enum BufferUsage {
    unknown,
    uniform,
    storage_read,
    storage_read_write
}

class Modifier {
    group : number | undefined;
    binding : number | undefined;
    location : number | undefined;
    workgroup_size : number[] | undefined;
    builtin : string | undefined;
    fnType : string | undefined;
    usage : BufferUsage = BufferUsage.unknown;

    empty() : boolean {
        const v = [
            this.group,
            this.binding,
            this.location,
            this.workgroup_size,
            this.builtin,
            this.fnType
        ];
        return v.every(x => x == undefined) && this.usage == BufferUsage.unknown;
    }

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
    case "i32"  : return 4;
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
    typeName : string;

    constructor(mod : Modifier, aggregate : string | undefined, type_name : string){
        this.mod = mod;
        this.aggregate = aggregate;
        this.typeName = type_name;
    }

    name() : string {
        if(this.aggregate != undefined){
            return `${this.aggregate}<${this.typeName}>`;
        }
        else{
            return this.typeName;
        }
    }

    str() : string {
        if(this.aggregate != undefined){
            return `${this.mod.str()} ${this.aggregate}<${this.typeName}>`;
        }
        else{
            return `${this.mod.str()} ${this.typeName}`;
        }
    }

    size() : number {
        const primitive_size = primitiveTypeSize(this.typeName);
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

        switch(this.typeName){
            case "f32"  : fmt = "float32"; break;
            case "i32"  : fmt = "int32"; break;
            case "u32"  : fmt = "uint32"; break;
            default:
                error(`unknown type format ${this.typeName}`);
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

export class Struct extends Type {
    members : Field[] = [];

    constructor(mod : Modifier, type_name : string){
        super(mod, undefined, type_name);
        this.mod = mod;
        this.typeName = type_name;
    }

    dump(){
        msg(`${this.mod.str()} struct ${this.typeName}{`);
        for(const va of this.members){
            msg(`    ${va.str()};`);
        }
        msg("}")
    }

    size() : number {
        return sum( this.members.map(x => x.type.size()) );
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

class Field extends Variable {
    parent : Struct;

    constructor(mod : Modifier, name : string, type : Type, parent : Struct){
        super(mod, name, type);
        this.parent = parent;
    }

    offset() : number {
        const idx = this.parent.members.indexOf(this);
        assert(idx != -1);

        const prev_siblings = this.parent.members.slice(0, idx);
        return sum(prev_siblings.map(x => x.type.size()));
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
    structs : Struct[] = [];

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

        const type_name = this.readToken(TokenType.type);

        const struct = this.structs.find(x => x.typeName == type_name);
        if(struct == undefined){

            return new Type(mod, undefined, type_name);
        }
        else{
            assert(mod.empty());
            return struct;
        }
    }

    readVariable(parent : Struct | undefined) : Variable {
        const mod = this.readModifiers();
        const name = this.readId();
        this.readText(":");
        const type = this.readType();
        
        if(parent == undefined){
            return new Variable(mod, name, type);
        }
        else{
            return new Field(mod, name, type, parent);
        }
    }

    structDeclaration(mod : Modifier) : Struct {
        this.advance();

        const name = this.readId();
        assert(mod.empty());
        const struct = new Struct(mod, name);
        this.structs.push(struct);

        // change typeTkn
        this.tokenList.filter(x => x.text == name).forEach(x => x.typeTkn = TokenType.type);

        this.readText("{");

        while(true){
            if(this.currentToken.text == "}"){
                break;
            }

            const field  = this.readVariable(struct) as Field;
            struct.members.push(field);

            if(this.currentToken.text == ","){
                this.readText(",");
            }
            else{
                break;
            }
        }
        this.readText("}");

        if(this.currentToken.text == ";"){
            this.readText(";");
        }

        return struct;
    }

    readBufferVar(mod : Modifier) : Variable {
        this.readText("var");

        if(this.currentText() == "<"){

            this.readText("<");
            let is_storage = false;
            while(true){
                const buf_attr = this.readReserved();
                switch(buf_attr){
                case "uniform":
                    mod.usage = BufferUsage.uniform;
                    break;
                case "storage":
                    is_storage = true;
                    break;
                case "read":
                    mod.usage = BufferUsage.storage_read;
                    break;
                case "read_write":
                    mod.usage = BufferUsage.storage_read_write;
                    break;
                default:
                    throw new MyError();
                }

                if(this.currentText() == ","){
                    this.readText(",");
                }
                else{
                    break;
                }
            }
            this.readText(">");

            if(is_storage){
                assert(mod.usage == BufferUsage.storage_read || mod.usage == BufferUsage.storage_read_write);
            }
            else{
                assert(mod.usage == BufferUsage.uniform);
            }
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

    readConst(){
        this.readText("const");
        while(this.currentToken.text != ";"){
            this.advance();
        }
        this.readText(";");
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

            const variable  = this.readVariable(undefined);
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

                case "const":
                    this.readConst();
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

export class ParamsParser extends Parser {
    vars = new Map<string, number | number[]>();

    constructor(tokens : Token[]){
        super(tokens);
    }

    PrimaryExpression() : number {
        if(this.currentToken.typeTkn == TokenType.identifier){
            const var_name = this.currentText();
            assert(this.vars.has(var_name));
            this.advance();

            return this.vars.get(var_name)! as number;
        }
        else if(this.currentToken.typeTkn == TokenType.Number){
            let n = parseFloat(this.currentToken.text);
            if(isNaN(n)){
                throw new Error();
            }

            this.advance();
            return n;
        }
        else if(this.currentToken.text == '('){

            this.readText("(");
            const n = this.AdditiveExpression();

            this.readText(")");

            return n;
        }
        else{
            throw new Error();
        }
    }    

    PowerExpression() : number {
        const trm1 = this.PrimaryExpression();
        if(this.currentToken.text == "^"){

            this.readText("^");

            const trm2 = this.PowerExpression();

            return trm1 ** trm2;
        }

        return trm1;
    }

    UnaryExpression() : number {
        if (this.currentToken.text == "-") {
            // 負号の場合

            this.readText("-");

            // 基本の式を読みます。
            const t1 = this.PowerExpression();

            // 符号を反転します。
            return - t1;
        }
        else {

            // 基本の式を読みます。
            return this.PowerExpression();
        }
    }

    DivExpression() : number {
        let trm1 = this.UnaryExpression();

        while(this.currentToken.text == "/"){
            this.readText("/");

            let trm2 = this.UnaryExpression();

            trm1 /= trm2;
        }
    
        return trm1;
    }

    
    MultiplicativeExpression() : number {
        let trm1 = this.DivExpression();

        while(this.currentToken.text == "*"){
            this.readText("*");

            let trm2 = this.DivExpression();

            trm1 *= trm2;
        }
    
        return trm1;
    }
    
    AdditiveExpression() : number {
        let trm1 = this.MultiplicativeExpression();

        while(this.currentToken.text == "+" || this.currentToken.text == "-"){
            const opr = this.currentToken.text;
            this.advance();

            const trm2 = this.MultiplicativeExpression();
            if(opr == "+"){
                trm1 += trm2;
            }
            else{
                trm1 -= trm2;
            }
        }

        return trm1;
    }

    listExpression() : number[] {
        const nums : number[] = [];

        this.readText("[");
        while(true){
            const n = this.AdditiveExpression();
            nums.push(n);

            if(this.currentText() != ","){
                break;
            }
            this.readText(",");
        }

        this.readText("]");

        return nums;
    }

    parseParams(){
        while(true){
            assert(this.currentToken.typeTkn == TokenType.identifier);
            const var_name = this.currentText();
            this.advance();

            this.readText("=");

            let value : number | number[];
            if(this.currentText() == "["){

                value = this.listExpression();
            }
            else{

                value = this.AdditiveExpression();
            }

            this.vars.set(var_name, value);

            if(this.currentText() != ","){
                break;
            }    
            this.readText(",");
        }

        assert(this.currentToken == eotToken);

        assert(this.vars.has("@instance_count") && this.vars.has("@instance_size"));

        const s = Array.from(this.vars.keys()).map(x => `${x}:${this.vars.get(x)}`).join(", ");
        msg(`vars: ${s}`);
    }

    get(var_name : string) : number{
        assert(this.vars.has(var_name));
        return this.vars.get(var_name) as number;
    }
}

export function parseParams(text : string) : ParamsParser {
    const tokens = lexicalAnalysis(text);

    const parser = new ParamsParser(tokens);
    parser.parseParams();

    return parser;
}

const eotToken : Token = new Token(TokenType.eot, TokenSubType.unknown, "", -1);

export async function parseAll(){
    const shader_names = [
        "line-comp",
        "line-vert",
        "line-vert-fix",
        "maxwell",
        "compute",
        "demo",
        "depth-frag",
        "function-comp",
        "function-vert",
        "instance-vert",
        "shape-vert",
        "texture-frag",
        "texture-vert",
        "updateSprites",
        "electric-field"
    ];

    for(const shader_name of shader_names){
        msg(`\n------------------------------ ${shader_name}`)
        const text = await fetchText(`../wgsl/${shader_name}.wgsl`);
        const mod = new Module(text);
        mod.dump();
    }
}

}
