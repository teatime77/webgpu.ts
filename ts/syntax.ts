import { assert, msg, MyError, sum } from "@i18n";
import { error, formatCode } from "./util";
import { isLetter, lexicalAnalysis } from "./lex";
import { Parser } from "./parser";

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

export const indexOpr = "t[i]";

const predifined = new Set<string>([
    "true", "dispatch", "drawTubes", "i32", "f32", "cos", "sin", "vec4", "t[i]", "swap"
]);

export function isAssignmentToken(text : string){
    return ["=", "+=", "-=", "*=", "/=", "%="].includes(text);
}

export function setParentSub(parent : AbstractSyntaxNode, ...args:(AbstractSyntaxNode | AbstractSyntaxNode[] | undefined)[]){
    for(const arg of args){
        if(arg instanceof AbstractSyntaxNode){
            arg.setParent(parent);
        }
        else if(Array.isArray(arg)){
            arg.forEach(x => x.setParent(parent));
        }
    }

}

export function getAllSub(alls: AbstractSyntaxNode[], ...args:(AbstractSyntaxNode | AbstractSyntaxNode[] | undefined)[]){
    for(const arg of args){
        if(arg instanceof AbstractSyntaxNode){
            arg.getAll(alls);
        }
        else if(Array.isArray(arg)){
            arg.forEach(x => x.getAll(alls));
        }
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

export function getUniformVars(domain: IDomain): Variable[] {
    return domain.vars.filter(x => x.mod.usage == BufferUsage.uniform);
}

export function getStorageVars(domain: IDomain): Variable[] {
    return domain.vars.filter(x => x.mod.usage == BufferUsage.storage);
}

export function getReadStorageVars(domain: IDomain): Variable[] {
    return domain.vars.filter(x => x.mod.usage == BufferUsage.storage && x.mod.readWrite == BufferReadWrite.storage_read);
}

export function getWriteStorageVars(domain: IDomain): Variable[] {
    return domain.vars.filter(x => x.mod.usage == BufferUsage.storage && x.mod.readWrite == BufferReadWrite.storage_read_write);
}

export interface IDomain {
    structs : Struct[];
    vars : Variable[];
    fns : Fn[];
}

export class Modifier {
    variable : Variable | undefined;
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

    toString() : string {
        let s = "";

        if(this.group != undefined){

            s += ` @group(${this.group})`
        }

        if(this.binding != undefined){

            s += ` @binding(${this.binding})`
        }

        if(this.usage == BufferUsage.uniform){
            s += " var<uniform>";
        }
        else if(this.usage == BufferUsage.storage){
            if(this.readWrite == BufferReadWrite.storage_read){

                s += " var<storage, read>";
            }
            else if(this.readWrite == BufferReadWrite.storage_read_write){
                
                s += " var<storage, read_write>";
            }
            else{
                
                s += " var<storage>";
            }
        }
        else{
            if(this.variable != undefined && this.variable.varKind != ""){
                s += ` ${this.variable.varKind}`;
            }
            else{
                s += " var"
            }
        }

        if(this.location != undefined){

            s += ` location(${this.location})`
        }

        if(this.workgroup_size != undefined){

            s += ` workgroup_size(${this.workgroup_size})`
        }

        if(this.builtin != undefined){

            s += ` @builtin(${this.builtin})`
        }

        if(this.fnType != undefined){

            s += ` ${this.fnType}`
        }

        return s;
    }
}

export abstract class AbstractSyntaxNode {
    static objCount = 0;
    objIdx : number;
    parent : AbstractSyntaxNode | undefined;

    constructor(){
        this.objIdx = AbstractSyntaxNode.objCount++;
    }

    abstract getAll(alls: AbstractSyntaxNode[]) : void;

    setParent(parentNode : AbstractSyntaxNode){
        this.parent = parentNode;
    }

    ancestors() : (App | Statement | Fn)[] {
        const objs : (App | Statement | Fn)[] = [];
        for(let obj : any = this; obj != undefined; obj = obj.parent){
            objs.push(obj);
        }

        return objs;
    }

    ancestorBlocks() : BlockStatement[] {
        return this.ancestors().filter(x => x instanceof BlockStatement);
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

    name() : string {
        return this.typeName;
    }

    toString() : string {
        return `${this.mod} ${this.typeName}`;
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

    name() : string {
        return `${this.typeName}<${this.elementType.typeName}>`;
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
        return `${this.typeName}<${this.elementType.typeName}>`;
    }

    toString() : string {
        return `${this.mod} ${this.typeName}<${this.typeName}>`;
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
        msg(`${this.mod} struct ${this.typeName}{`);
        for(const va of this.members){
            msg(`    ${va};`);
        }
        msg("}")
    }

    size() : number {
        assert(this.members.every(x => x.type != undefined));
        return sum( this.members.map(x => x.type!.size()) );
    }

    toString() : string{
        const s1 = this.members.map(x => `${x}`).join(",\n");
        return `struct ${this.typeName} {\n${s1}\n}\n`;
    }
}

export class Variable extends AbstractSyntaxNode {
    varKind : string;
    mod : Modifier;
    name : string;
    type : Type | undefined;
    initializer : Term | undefined;

    constructor(varKind: string, mod : Modifier, name : string, type : Type | undefined, initializer : Term | undefined){
        super();
        this.varKind = varKind;
        this.mod = mod;
        this.name = name;
        this.type = type;
        this.initializer = initializer;

        this.mod.variable = this;
    }

    setParent(parent : AbstractSyntaxNode){
        super.setParent(parent);
        setParentSub(this, this.type, this.initializer);
    }

    getAll(alls: AbstractSyntaxNode[]) : void{ 
        alls.push(this);
        getAllSub(alls, this.type, this.initializer);
    }

    argStr() : string {
        if(this.mod.builtin != undefined){
            return `@builtin(${this.mod.builtin}) ${this}`;
        }
        else{
            return this.toString();
        }
    }

    toString() : string {
        const s1 = this.type        == undefined ? "" : ` : ${this.type.name()}`;
        const s2 = this.initializer == undefined ? "" : ` = ${this.initializer}`;

        return `${this.name}${s1}${s2}`;
    }
}

export class Field extends Variable {
    constructor(mod : Modifier, name : string, type : Type | undefined, initializer : Term | undefined, parent : Struct){
        super("", mod, name, type, initializer);
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
        let modStr = "";

        if(this.mod.fnType != undefined){
            modStr += `${this.mod.fnType} @workgroup_size(64) `;
        }

        const vars_s = this.args.map(x => x.argStr()).join(", ");
        const type_s = this.type == undefined ? "" : `-> ${this.type.name()}`;
        
        return `${modStr}fn ${this.name}(${vars_s}) ${type_s}${this.block}`;
    }
}

export class Domain extends AbstractSyntaxNode implements IDomain {
    structs : Struct[] = [];
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

    toString() : string {
        const s1 = this.structs.map(x => `${x}\n`).join("");
        const s2 = this.vars.map(x => `${x.mod} ${x};`).join("\n") + "\n";
        const s3 = this.fns.map(x => `${x}\n`).join("");

        return formatCode(s1 + s2 + s3);
    }
}

export abstract class Term extends AbstractSyntaxNode {
    sign : number = 1;

    isOperator() : boolean {
        return this instanceof App && this.precedence() != -1;
    }

    isDot() : this is App & { fncName: "." }{
        return this instanceof App && this.fncName == ".";
    }

    isAssignmentApp() : boolean {
        return this instanceof App && isAssignmentToken(this.fncName);
    }

    isApp(fncName : string) : this is App {
        return this instanceof App && this.fncName == fncName;
    }

    getAll(alls: AbstractSyntaxNode[]) : void {
        alls.push(this);
    }

    int() : number {
        if(this instanceof ConstNum){
            return this.int()
        }
        throw new MyError();
    }


    abstract str() : string;

    toString() : string {
        const s = this.str();
        if(this.sign == -1){

            return `- ${s}`;
        }
        else{
            return s;
        }
    }

}

export class Str extends Term{
    text : string;

    constructor(text : string){
        super();
        this.text = text;
    }

    str() : string {
        return `"${this.text}"`;
    }
}

export class RefVar extends Term{
    name: string;
    refVar : Variable | undefined;
    refType : Type | undefined;

    constructor(target: string | Type){
        super();
        if(typeof target == "string"){
            this.name = target;
        }
        else{
            this.name    = target.name();
            this.refType = target;
        }
    }

    resolveVariable(domains:Domain[]){
        if(this.refVar != undefined || this.refType != undefined){
            return;
        }

        const blocks = this.ancestorBlocks();
        this.refVar = blocks.map(x => x.findVariable(this.name)).find(x => x != undefined);
        if(this.refVar == undefined){
            const vars = domains.map(x => x.vars).flat();
            this.refVar = vars.find(x => x.name == this.name);

            if(this.refVar == undefined && !predifined.has(this.name)){

                msg(`ref no var:${this.name} parent:[${this.parent!}]`);
            }
        }
    }

    str() : string {
        return this.name;
    }
}


export class ConstNum extends Term{
    text : string;
    value : number;

    static zero() : ConstNum {
        return new ConstNum("0");
    }

    constructor(text : string){
        super();
        this.text = text;

        if(text[0] == "#"){
            this.value = parseInt(text.substring(1), 16);
        }
        else{
            this.value = parseFloat(text);
        }

        if(isNaN(this.value)){
            throw new MyError();
        }
    }

    int() : number {
        if(Math.floor(this.value) == this.value){
            return this.value;
        }
        throw new MyError();
    }

    str() : string {
        return this.text;
    }
}

export class App extends Term{
    fnc : RefVar;
    args: Term[];
    remParentheses : boolean = false;

    static startEnd : { [start : string] : string } = {
        "(" : ")",
        "[" : "]",
        "{" : "}",
    }

    get refVar() : RefVar | null {
        if(this.fnc != null && this.fnc instanceof RefVar){
            return this.fnc;
        }
        else{
            return null;
        }
    }

    get fncName() : string {
        return this.fnc.name;
    }

    constructor(fnc: RefVar, args: Term[]){
        super();
        this.fnc    = fnc;
        this.fnc.parent = this;

        this.args   = args.slice();

        this.args.forEach(x => x.parent = this);
    }

    setParent(parent : AbstractSyntaxNode){
        super.setParent(parent);
        setParentSub(this, this.fnc, this.args);
    }
    
    addArgs(...trms : Term[]){
        for(const trm of trms){
            this.args.push(trm);
            trm.parent = this;
        }
    }

    precedence() : number {
        switch(this.fncName){
        case "^": 
            return 0;

        case "*": 
        case "/": 
            return 1;

        case "+": 
        case "-": 
            return 2;
        }

        return -1;
    }

    str() : string {
        const args = this.args.map(x => x.toString());
        
        let text : string;

        if(this.fncName == indexOpr){

            return `${args[0]}[${args.slice(1).join(", ")}]`
        }
        else if(isLetter(this.fncName[0]) || this.fnc.refType != undefined){
            const args_s = args.join(", ");
            text = `${this.fncName}(${args_s})`;
        }
        else if(this.fncName == "+"){
            text = "";
            for(const [i, s] of args.entries()){
                if(this.args[i].sign == -1){
                    text += ` ${s}`;
                }
                else if(i == 0){

                    text += `${s}`;
                }
                else{

                    text += ` + ${s}`;
                }
            }
        }
        else{
            switch(this.fncName){
                case "/":
                    if(this.args.length != 2){
                        throw new MyError();
                    }
                    break;

                case ".":
                    assert(args.length == 2);
                    return `${args[0]}.${args[1]}`;
            }

            text = args.join(` ${this.fncName} `);
        }

        if(this.isOperator() && this.parent instanceof App && this.parent.isOperator()){
            if(this.parent.precedence() <= this.precedence()){
                return `(${text})`;
            }            
        }

        return text;
    }

    getAll(alls: AbstractSyntaxNode[]) : void{ 
        alls.push(this);
        if(isLetter(this.fncName[0])){
            alls.push(this.fnc);
        }
        getAllSub(alls, this.args);
    }
}


export abstract class Statement extends AbstractSyntaxNode {
    isAssignmentCall() : boolean {
        return this instanceof CallStatement && this.app.isAssignmentApp();
    }
}


export class VariableDeclaration extends Statement {
    variable: Variable;

    constructor(variable: Variable) {
        super();
        this.variable = variable;
    }

    setParent(parent : AbstractSyntaxNode){
        super.setParent(parent);
        setParentSub(this, this.variable);
    }

    getAll(alls: AbstractSyntaxNode[]) : void{ 
        alls.push(this);
        getAllSub(alls, this.variable.initializer);
    }

    toString() : string {
        return `${this.variable.mod} ${this.variable};\n`;
    }
}

export class CallStatement extends Statement {
    app : App;

    constructor(app : App){
        super();
        this.app = app;
    }

    setParent(parent : AbstractSyntaxNode){
        super.setParent(parent);
        setParentSub(this, this.app);
    }

    getAll(alls: AbstractSyntaxNode[]) : void{  
        alls.push(this);
        getAllSub(alls, this.app);
    }

    toString() : string {
        return `${this.app};`;
    }
}

export class ReturnStatement extends Statement {
    value : Term | undefined;

    constructor(value : Term | undefined){
        super();
        this.value = value;
    }

    setParent(parent : AbstractSyntaxNode){
        super.setParent(parent);
        setParentSub(this, this.value);
    }

    getAll(alls: AbstractSyntaxNode[]) : void{     
        alls.push(this);
        getAllSub(alls, this.value);
    }

    toString() : string {
        if(this.value == undefined){
            return `return;`;
        }
        else{
            return `return ${this.value};`;
        }
    }
}

export class BlockStatement extends Statement {
    statements : Statement[];

    constructor(statements : Statement[]){
        super();
        this.statements = statements.slice();
    }

    addStatements(...statements : Statement[]){
        for(const statement of statements){
            this.statements.push(statement);
            statement.parent = this;
        }
    }

    setParent(parent : AbstractSyntaxNode){
        super.setParent(parent);
        setParentSub(this, this.statements);
    }

    findVariable(name : string) : Variable | undefined {
        const decl = this.statements.find(x => x instanceof VariableDeclaration && x.variable.name == name) as VariableDeclaration;
        return decl == undefined ? undefined : decl.variable;
    }

    getAll(alls: AbstractSyntaxNode[]) : void{  
        alls.push(this);
        getAllSub(alls, this.statements);
    }

    toString() : string {
        const s1 = this.statements.map(x => `${x}`).join("\n") + "\n";

        return `{\n${s1}}\n`;
    }
}

export class IfStatement extends Statement {
    condition: Term; // Changed
    ifBlock: BlockStatement;
    elseIf : IfStatement | undefined;
    elseBlock: BlockStatement | undefined;

    constructor(condition: Term, ifBlock: BlockStatement, elseIf : IfStatement | undefined, elseBlock: BlockStatement | undefined) {
        super();
        this.condition = condition;
        this.ifBlock   = ifBlock;
        this.elseIf    = elseIf;
        this.elseBlock = elseBlock;
    }

    setParent(parent : AbstractSyntaxNode){
        super.setParent(parent);
        setParentSub(this, this.condition, this.ifBlock, this.elseIf, this.elseBlock);
    }

    getAll(alls: AbstractSyntaxNode[]) : void{        
        alls.push(this);
        getAllSub(alls, this.condition, this.ifBlock, this.elseIf, this.elseBlock);
    }

    toString() : string {
        var s1 = `if( ${this.condition})${this.ifBlock}`;
        if(this.elseIf != undefined){
            s1 += `else ${this.elseIf}`;
        }
        if(this.elseBlock != undefined){
            s1 += `else${this.elseBlock}`;
        }

        return s1;
    }
}

export class WhileStatement extends Statement {
    condition: Term;
    block: BlockStatement;

    constructor(condition: Term, body: BlockStatement) {
        super();
        this.condition = condition;
        this.block = body;
    }

    setParent(parent : AbstractSyntaxNode){
        super.setParent(parent);
        setParentSub(this, this.condition, this.block);
    }

    getAll(alls: AbstractSyntaxNode[]) : void{        
        alls.push(this);
        getAllSub(alls, this.condition, this.block);
    }

    toString() : string {
        return `while(${this.condition})${this.block}\n`;
    }
}

export class ForStatement extends Statement {
    initializer: Variable;
    condition: Term | undefined;
    update: Term | undefined;
    body: BlockStatement;

    constructor(initializer: Variable, condition: Term | undefined, update: Term | undefined, body: BlockStatement) {
        super();
        this.initializer = initializer;
        this.condition = condition;
        this.update = update;
        this.body = body;
    }

    setParent(parent : AbstractSyntaxNode){
        super.setParent(parent);
        setParentSub(this, this.initializer, this.condition, this.update, this.body);
    }

    getAll(alls: AbstractSyntaxNode[]) : void{        
        alls.push(this);
        getAllSub(alls, this.initializer.initializer, this.condition, this.update, this.body);
    }
}

export class ParallelStatement extends Statement {
    blocks : BlockStatement[];

    constructor(blocks : BlockStatement[]){
        super();
        this.blocks = blocks;
    }

    getAll(alls: AbstractSyntaxNode[]) : void{        
        alls.push(this);
        getAllSub(alls, this.blocks);
    }
}


export class Module implements IDomain {
    name : string;
    text : string;
    structs : Struct[] = [];
    vars : Variable[] = [];
    fns : Fn[] = [];

    constructor(name : string, text : string){
        this.name = name;
        this.text = text;

        const tokens = lexicalAnalysis(text);

        const parser = new Parser(tokens, 0);
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

    dump(){
        this.structs.forEach(x => x.dump());
        this.vars.forEach(x => msg(`${x};`))
        this.fns.forEach(x => msg(`${x}`))
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
