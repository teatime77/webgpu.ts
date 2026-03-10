import { assert, msg, MyError, sum, fetchText  } from "@i18n";
import { AbstractSyntaxNode, BlockStatement, FncParser, getAllSub, setParentSub, Statement, Term } from "./parser-new.js";
import { makeShaderModule, error } from "./util.js";
import { Script } from "./script.js";

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

export class Module {
    name : string;
    module : GPUShaderModule;
    structs : Struct[] = [];
    vars : Variable[] = [];
    fns : Fn[] = [];

    constructor(name : string, text : string){
        this.name = name;

        this.module = makeShaderModule(text);

        const tokens = lexicalAnalysis(text);

        const parser = new FncParser(tokens, 0);
        parser.parseSource();
        [this.structs, this.vars, this.fns] = [parser.structs, parser.vars, parser.fns];

        assert(this.fns.length == 1);
        const main = this.fns[0];
        switch(main.mod.fnType){
        case "@compute":
            break;
        case "@vertex":
            break;
        case "@fragment":
            break;
        default:
            throw new MyError();
        }
    }

    uniformVars() : Variable[] {
        return this.vars.filter(x => x.mod.usage == BufferUsage.uniform);
    }

    readStorageVars() : Variable[]{
        return this.vars.filter(x => x.mod.usage == BufferUsage.storage && x.mod.readWrite == BufferReadWrite.storage_read);
    }

    writeStorageVars() : Variable[]{
        return this.vars.filter(x => x.mod.usage == BufferUsage.storage && x.mod.readWrite == BufferReadWrite.storage_read_write);
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

export enum BufferUsage {
    unknown,
    uniform,
    storage
}

export enum BufferReadWrite {
    unknown,
    storage_read,
    storage_read_write
}

export class Modifier {
    group : number | undefined;
    binding : number | undefined;
    location : number | undefined;
    workgroup_size : number[] | undefined;
    builtin : string | undefined;
    fnType : string | undefined;
    usage : BufferUsage = BufferUsage.unknown;
    readWrite : BufferReadWrite = BufferReadWrite.unknown;

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
        return this.toString();
    }

    toString() : string {
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

export abstract class Type extends AbstractSyntaxNode {
    mod : Modifier;
    typeName : string;

    constructor(mod : Modifier, type_name : string){
        super();
        this.mod = mod;
        this.typeName = type_name;
    }

    getAll(alls: AbstractSyntaxNode[]) : void{ 
        alls.push(this);
    }

    toString() : string {
        return `${this.mod.str()} ${this.typeName}`;
    }

    size() : number {
        return primitiveTypeSize(this.typeName);
    }

    format() : string {
        switch(this.typeName){
            case "f32"  : return "float32";
            case "i32"  : return "int32";
            case "u32"  : return "uint32";
            default     : return this.typeName;
        }
    }
}

export class SimpleType extends Type {
    constructor(mod : Modifier, type_name : string){
        super(mod, type_name);
    }
}

export class ArrayType extends Type {
    elementType : Type;
    dimensions : Term[];

    constructor(mod : Modifier, elementType : Type, dimensions : Term[]){
        super(mod, "array");
        this.elementType = elementType;
        this.dimensions  = dimensions.slice();
    }

    setParent(parent : AbstractSyntaxNode){
        super.setParent(parent);
        setParentSub(this, this.elementType, this.dimensions);
    }

    getAll(alls: AbstractSyntaxNode[]) : void{ 
        alls.push(this);
        getAllSub(alls, this.elementType, this.dimensions);
    }
}

export class ShaderType extends Type {
    elementType : Type;

    constructor(mod : Modifier, type_name : string, elementType : Type){
        super(mod, type_name);
        this.elementType = elementType;
    }

    getAll(alls: AbstractSyntaxNode[]) : void{ 
        alls.push(this);
        getAllSub(alls, this.elementType);
    }

    name() : string {
        return `${this.typeName}<${this.typeName}>`;
    }

    toString() : string {
        return `${this.mod.str()} ${this.typeName}<${this.typeName}>`;
    }

    size() : number {
        const elementTypeSize = primitiveTypeSize(this.elementType.typeName);

        switch(this.typeName){
        case "mat4x4" : return 4 * 4 * elementTypeSize;
        case "mat3x3" : return 3 * 3 * elementTypeSize;
        case "vec4"   : return     4 * elementTypeSize;
        case "vec3"   : return     3 * elementTypeSize;
        case "vec2"   : return     2 * elementTypeSize;

        case "texture_2d":
        case "array" :
        default:
            error(`unknown typeName size ${this.typeName}`);
            return NaN;
        }
    }

    format() : string {
        const elementTypeName = this.elementType.format();

        switch(this.typeName){
        case undefined: return elementTypeName;
        case "vec2"   : return elementTypeName + "x2";
        case "vec3"   : return elementTypeName + "x3";
        case "vec4"   : return elementTypeName + "x4";
        default       : return `${this.typeName}<${elementTypeName}>`;
        }
    }
}

export class Struct extends Type {
    members : Field[] = [];

    constructor(mod : Modifier, type_name : string){
        super(mod, type_name);
        this.mod = mod;
        this.typeName = type_name;
    }

    getAll(alls: AbstractSyntaxNode[]) : void{ 
        alls.push(this);
        getAllSub(alls, this.members);
    }

    setParent(parent : AbstractSyntaxNode){
        super.setParent(parent);
        setParentSub(this, this.members);
    }

    dump(){
        msg(`${this.mod.str()} struct ${this.typeName}{`);
        for(const va of this.members){
            msg(`    ${va.str()};`);
        }
        msg("}")
    }

    size() : number {
        assert(this.members.every(x => x.type != undefined));
        return sum( this.members.map(x => x.type!.size()) );
    }
}

export class Variable extends AbstractSyntaxNode {
    mod : Modifier;
    name : string;
    type : Type | undefined;
    initializer : Term | undefined;

    constructor(mod : Modifier, name : string, type : Type | undefined, initializer : Term | undefined){
        super();
        this.mod = mod;
        this.name = name;
        this.type = type;
        this.initializer = initializer;
    }

    setParent(parent : AbstractSyntaxNode){
        super.setParent(parent);
        setParentSub(this, this.type, this.initializer);
    }

    getAll(alls: AbstractSyntaxNode[]) : void{ 
        alls.push(this);
        getAllSub(alls, this.type, this.initializer);
    }

    toString() : string {
        assert(this.type != undefined);
        return `${this.mod.str()} ${this.name} : ${this.type!.str()}`
    }
}

export class Field extends Variable {
    constructor(mod : Modifier, name : string, type : Type | undefined, initializer : Term | undefined, parent : Struct){
        super(mod, name, type, initializer);
        this.parent = parent;
    }

    get struct() : Struct {
        return this.parent as Struct;
    }

    offset() : number {
        const idx = this.struct.members.indexOf(this);
        assert(idx != -1);

        const prev_siblings = this.struct.members.slice(0, idx);
        assert(prev_siblings.every(x => x.type != undefined))
        return sum(prev_siblings.map(x => x.type!.size()));
    }
}

export class Fn extends AbstractSyntaxNode {
    mod : Modifier;
    name : string;
    args : Variable[] = [];
    type : Type | undefined;
    block! : BlockStatement;

    constructor(mod : Modifier, name : string){
        super();
        this.mod = mod;
        this.name = name;
    }

    setParent(parent : AbstractSyntaxNode){
        super.setParent(parent);
        setParentSub(this, this.args, this.type, this.block);
    }

    getAll(alls: AbstractSyntaxNode[]) : void{ 
        alls.push(this);
        getAllSub(alls, this.args, this.type, this.block);
    }

    toString() : string {
        const vars_s = this.args.map(x => x.str()).join(", ");
        const type_s = this.type == undefined ? "" : `-> ${this.type.str()}`;
        
        return `${this.mod.str()} fn ${this.name}(${vars_s}) ${type_s}`;
    }
}

export class Domain extends AbstractSyntaxNode {
    vars : Variable[] = [];
    fns : Fn[] = [];

    getAll(alls: AbstractSyntaxNode[]) : void{
        alls.push(this);
        getAllSub(alls, this.vars, this.fns);
    }

    setParent(){
        setParentSub(this, this.vars, this.fns);
    }

    getAllInDomain() : (Statement | Term)[]{
        const alls: (Statement | Term)[] = [];
        this.vars.forEach(x => x.getAll(alls));
        this.fns.forEach(x => x.getAll(alls));
        return alls;
    }
}

const eotToken : Token = new Token(TokenType.eot, TokenSubType.unknown, "", -1);

export async function parseAll(){
    const shader_names = [
        "arrow-comp",
        "arrow-instance-vert",
        "mesh-comp",
        "mesh-instance-vert",
        "phong-frag",
        "line-vert",
        "maxwell",
        "compute",
        "demo",
        "depth-frag",
        "point-vert",
        "surface-vert",
        "texture-frag",
        "texture-vert",
        "electric-field"
    ];

    for(const shader_name of shader_names){
        msg(`\n------------------------------ ${shader_name}`);
        const text = await fetchText(`./wgsl/${shader_name}.wgsl`);
        const module = new Module(shader_name, text);
        // mod.dump();
    }

    const script = new Script();
    await script.init();
}
