import { assert, msg, MyError  } from "@i18n";
import { Modifier, Struct, Token, TokenSubType, TokenType, Type, Variable, Fn, Field, BufferUsage, Module, ShaderType, isLetter, BufferReadWrite, SimpleType, ArrayType, Domain } from "./parser.js";
import { error } from "./util.js";

export const indexOpr = "t[i]";

const predifined = new Set<string>([
    "true", "dispatch", "drawTubes", "i32", "f32", "cos", "sin", "vec4", "t[i]", "swap"
]);



function isRelationToken(text : string){
    return [ "==", "=", "!=", "<", ">", "<=", ">=", "in", "notin", "subset" ].includes(text);
}

function isAssignmentToken(text : string){
    return ["=", "+=", "-=", "*=", "/=", "%="].includes(text);
}

function operator(opr : string) : RefVar {
    return new RefVar(opr);
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

export class Rational{
    numerator : number = 1;
    denominator : number = 1;
    parent : Term | null = null;

    constructor(numerator : number, denominator : number = 1){
        this.numerator = numerator;
        this.denominator = denominator;
    }


    int() : number {
        assert(this.denominator == 1);
        return this.numerator;
    }

    toString() : string {
        if(this.denominator == 1){

            return `${this.numerator}`;
        }
        else{

            return `${this.numerator} / ${this.denominator}`;
        }
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

export abstract class Term extends AbstractSyntaxNode {
    // 係数
    value : Rational = new Rational(1);

    dumpTerm() : void {
        throw new MyError();
    }

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
            return this.value.int()
        }
        throw new MyError();
    }
}

export class Str extends Term{
    text : string;

    constructor(text : string){
        super();
        this.text = text;
    }

    dumpTerm() : void {
        msg(`pre-pare:${this.constructor.name}:${this.text}`);
    }

    toString() : string {
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

    dumpTerm() : void {
        msg(`pre-pare:${this.constructor.name}:${this.name}`);
    }

    toString() : string {
        return this.name;
    }
}


export class ConstNum extends Term{
    text : string;

    static zero() : ConstNum {
        return new ConstNum("0");
    }

    constructor(text : string){
        super();
        this.text = text;

        let n : number;

        if(text[0] == "#"){
            n = parseInt(text.substring(1), 16);
        }
        else{
            n = parseFloat(text);
        }

        if(isNaN(n)){
            throw new MyError();
        }

        this.value = new Rational(n, 1);
    }

    dumpTerm() : void {
        msg(`pre-pare:${this.constructor.name}:${this.value}`);
    }

    toString() : string {
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

    dumpTerm() : void {
        msg(`pre-pare:${this.constructor.name}:${this.value}`);
        this.fnc.dumpTerm();
        this.args.forEach(x => x.dumpTerm());
    }


    toString() : string {
        const args = this.args.map(x => x.toString());
        
        let text : string;

        if(this.fncName == indexOpr){

            return `${args[0]}[${args.slice(1).join(", ")}]`
        }
        else if(isLetter(this.fncName[0]) || this.fnc.refType != undefined){
            const args_s = args.join(", ");
            text = `${this.fncName}(${args_s})`;
        }
        else{

            switch(this.fncName){
                case "+":
                    switch(args.length){
                    case 0: return " +[] ";
                    case 1: return ` +[${args[0]}] `;
                    }
                    break
    
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
    dumpStatement() : void {
        throw new MyError();
    }

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

    dumpStatement() : void {
        msg(`pre-pare:${this.constructor.name}`);
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

    dumpStatement() : void {
        msg(`pre-pare:${this.constructor.name}:${this.app}`);
        this.app.dumpTerm();
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

    dumpStatement() : void {
        msg(`pre-pare:${this.constructor.name}`);
        this.statements.forEach(x => x.dumpStatement());
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

    dumpStatement() : void {
        msg(`pre-pare:${this.constructor.name}`);
        this.condition.dumpTerm();
        this.block.dumpStatement();
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

export enum Context {
    unknown,
    Object
}

export class FncParser {
    tokenPos:number;
    tokens: Token[];
    token: Token;
    lineStart : number = 0;

    structs : Struct[] = [];
    vars : Variable[] = [];
    fns : Fn[] = [];

    constructor(tokens: Token[], tokenPos : number){
        this.tokenPos = tokenPos;
        this.tokens = tokens;

        this.skipNewLine();
        this.token = this.tokens[this.tokenPos];
    }

    skipNewLine(){
        while(this.tokenPos < this.tokens.length && this.tokens[this.tokenPos].typeTkn == TokenType.newLine){
            this.tokenPos++;
        }
    }

    lineText() : string {
        let i = this.tokens.findIndex((t, i)=>this.tokenPos < i && t.typeTkn == TokenType.newLine);
        const lineWords = this.tokens.slice(this.tokenPos + 1, i != -1 ? i : this.tokens.length).map(x => x.text);
        
        return lineWords.join(" ");
    }

    next(){
        if(this.tokens.length == 0){

            this.token = new Token(TokenType.eot, TokenSubType.unknown, "", 0);
        }
        else{

            this.tokenPos++;
            this.skipNewLine();
            this.lineStart = this.tokenPos;

            this.token = this.tokens[this.tokenPos];
        }
    }

    isEoT() : boolean {
        return this.token.typeTkn == TokenType.eot;
    }

    showError(text : string){
        // const i = this.tokens_cp.length - this.tokens.length;
        // const words = this.tokens_cp.map(x => x.text);

        // words.splice(i, 0, `<<${text}>>`);
        // msg(`token err:${words.join(" ")}`);
    }

    nextToken(text : string){
        if(this.token.text != text){
            this.showError(text);
            throw new MyError();
        }

        this.next();
    }

    readToken(type : TokenType) : string {
        if(this.token.typeTkn != type){
            error(`identifier is expected.`)
        }

        return this.readText();
    }

    current(){
        return this.token.text;
    }

    peek() : string {
        return this.tokenPos + 1 < this.tokens.length ? this.tokens[this.tokenPos + 1].text : ""; 
    }

    readInt(){
        if(this.token.typeTkn != TokenType.Number){
            error("number is missing.")
        }
        const n = parseInt(this.token.text);

        this.next();

        return n;
    }

    readAttribute() : number {
        this.next();

        this.nextToken("(");
        const value = this.readInt();
        this.nextToken(")");

        return value;
    }

    readAttributeList() : number[] {
        this.next();

        this.nextToken("(");

        const values : number[] = [];

        while(true){
            const n = this.readInt();
            values.push( n );

            if(this.current() == ","){

                this.nextToken(",");
            }
            else{
                break;
            }
        }

        this.nextToken(")");

        return values;
    }

    readModifiers() : Modifier {
        const mod = new Modifier();

        while(true){
            switch(this.current()){
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
                    mod.fnType = this.current();
                    this.next();
                    break;

                case "@builtin":
                    this.next();
                    this.nextToken("(");
                    mod.builtin = this.readName();
                    this.nextToken(")");
                    break;

                default:
                    return mod;
            }
        }
    }

    readType(ctx : Context) : Type {
        const mod = this.readModifiers();

        switch(this.current()){
            case "mat4x4":
            case "mat3x3":
            case "vec4":
            case "vec3":
            case "vec2":
            case "array":
            case "texture_2d":
                const aggregate = this.readToken(TokenType.type);
                this.nextToken("<");
                const elementType = this.readType(ctx);
                this.nextToken(">");
                return new ShaderType(mod, aggregate, elementType);
        }

        const type_name = this.readToken(TokenType.type);

        let elementType : Type;
        const struct = this.structs.find(x => x.typeName == type_name);
        if(struct == undefined){

            elementType = new SimpleType(mod, type_name);
        }
        else{
            assert(mod.empty());
            elementType = struct;
        }

        if(this.current() == "["){
            const dimensions = this.readComma(ctx, "[", "]");
            return new ArrayType(mod, elementType, dimensions);
        }
        else{
            return elementType;
        }
    }

    structDeclaration(ctx : Context) : Struct {
        this.nextToken("struct");

        const name = this.readName();
        const struct = new Struct(new Modifier(), name);
        this.structs.push(struct);

        // change typeTkn
        this.tokens.filter(x => x.text == name).forEach(x => x.typeTkn = TokenType.type);

        this.nextToken("{");

        while(true){
            if(this.current() == "}"){
                break;
            }

            const mod = this.readModifiers();
            const field  = this.readVariable(ctx, "", mod, struct) as Field;
            struct.members.push(field);

            if(this.current() == ","){
                this.nextToken(",");
            }
            else{
                break;
            }
        }
        this.nextToken("}");

        if(this.current() == ";"){
            this.nextToken(";");
        }

        return struct;
    }


    readFn(ctx : Context) : Fn {
        const modFn = this.readModifiers();

        this.nextToken("fn");

        const name = this.readName();

        const fn = new Fn(modFn, name);

        this.nextToken("(");

        while(true){
            if(this.current() == ")"){
                break;
            }

            const modVar = this.readModifiers();
            const variable  = this.readVariable(ctx, "", modVar, undefined);
            fn.args.push(variable);

            if(this.current() == ","){
                this.nextToken(",");

            }
            else{
                break;
            }
        }
        this.nextToken(")");

        if(this.current() == "->"){

            this.nextToken("->");

            fn.type = this.readType(ctx);
        }

        fn.block = this.parseBlock(ctx, );

        return fn;
    }

    readFncArgs(ctx : Context, app : App){
        this.nextToken("(");

        while(this.current() != ")"){
            const trm = this.RelationalExpression(ctx);

            if(this.token.text == ":"){
                this.nextToken(":");
                const value = this.LogicalExpression(ctx);
                if(trm instanceof RefVar){
                    // msg(`fnc arg[${trm.name}] with init`);
                }
                else{
                    msg(`NG fnc arg`);
                }
            }
            
            app.args.push(trm);

            if(this.token.text == ","){
                this.nextToken(",");
            }
        }

        this.nextToken(")");
    }

    readComma(ctx : Context, start: string, end : string) : Term[]{
        this.nextToken(start);

        const terms : Term[] = [];
        while(this.current() != end){
            const trm = this.RelationalExpression(ctx);
            terms.push(trm);

            if(this.token.text == ","){
                this.nextToken(",");
            }
        }

        this.nextToken(end);

        return terms;
    }

    readArgs(ctx : Context, start: string, end : string, app : App){
        const terms = this.readComma(ctx, start, end);

        app.args.push(...terms);
    }

    readText() : string {
        const text = this.current();
        this.next();

        return text;
    }

    readName() : string {
        const idType = this.token.typeTkn;
        assert(idType == TokenType.identifier || idType == TokenType.type);

        return this.readText();
    }

    readId(ctx : Context) : RefVar | App {
        const idType = this.token.typeTkn;

        let term : RefVar | App;

        if(idType == TokenType.type && this.peek() == "<"){
            const tp = this.readType(ctx);
            term = new RefVar(tp);
        }
        else{
            const name = this.readName();
            term = new RefVar(name);
        }

        while(true){
            if(this.token.text == '('){

                if(term instanceof RefVar){
                    term = new App(term, []);
                }
                else{
                    term = new App(operator("()"), [term]);
                }
                this.readFncArgs(ctx, term);
            }
            else if(this.token.text == '['){

                term = new App(operator(indexOpr), [term]);
                this.readArgs(ctx, "[", "]", term);
            }
            else if(this.token.text == "."){
                this.nextToken(".");
                const id = this.readId(ctx);
                term = new App(operator("."), [term, id]);
            }
            else{

                return term;
            }
        }
    }

    PrimaryExpression(ctx : Context) : Term {
        let trm : Term;

        if(this.token.typeTkn == TokenType.identifier || this.token.typeTkn == TokenType.type){
            const id = this.readId(ctx);
            if(ctx == Context.Object && this.current() == ":"){
                this.nextToken(":");
                const value = this.LogicalExpression(Context.unknown);
            }

            return id;
        }
        else if(this.token.typeTkn == TokenType.Number){

            trm = new ConstNum(this.token.text);
            this.next();
        }
        else if(this.token.typeTkn == TokenType.String){
            trm = new Str(this.token.text);
            this.next();
        }
        else if(this.token.text == '('){

            const terms = this.readComma(ctx, "(", ")");

            if(this.current() == "=>"){
                this.nextToken("=>");
                if(this.current() == "{"){

                    const block = this.parseBlock(ctx);
                }
                else{
                    this.ArithmeticExpression(ctx);
                }

                return new RefVar("arrow");
            }
            else{
                const cmTerm = new App(operator(","), terms);

                if(this.current() == "."){
                    this.nextToken(".");
                    const id = this.readId(ctx);
                    return new App(operator("."), [cmTerm, id]);
                }
                else{
                    return cmTerm;
                }
            }
        }
        else if(this.token.text == '{'){
            const terms = this.readComma(Context.Object, "{", "}");
            return new App(operator("{}"), terms);
        }
        else if(this.token.text == '['){
            const terms = this.readComma(ctx, "[", "]");
            return new App(operator("[]"), terms);
        }
        else{
            throw new MyError();
        }

        return trm;
    }

    PowerExpression(ctx : Context) : Term {
        const trm1 = this.PrimaryExpression(ctx);
        if(this.token.text == "^"){

            this.nextToken("^");

            const trm2 = this.PowerExpression(ctx);

            return new App(operator("^"), [trm1, trm2]);
        }

        return trm1;
    }

    UnaryExpression(ctx : Context) : Term {
        if (this.token.text == "-") {
            // 負号の場合

            this.nextToken("-");

            // 基本の式を読みます。
            const t1 = this.PowerExpression(ctx);

            // 符号を反転します。
            t1.value.numerator *= -1;

            return t1;
        }
        else {

            // 基本の式を読みます。
            return this.PowerExpression(ctx);
        }
    }

    
    DivExpression(ctx : Context) : Term {
        let trm1 = this.UnaryExpression(ctx);
        while(this.token.text == "/" || this.token.text == "%"){
            let app = new App(operator(this.token.text), [trm1]);
            this.next();

            while(true){
                let trm2 = this.
                UnaryExpression(ctx);
                app.args.push(trm2);
                
                if(this.token.text == app.fncName){
                    this.next();
                }
                else{
                    trm1 = app;
                    break;
                }
            }
        }
    
        return trm1;
    }

    
    MultiplicativeExpression(ctx : Context) : Term {
        let trm1 = this.DivExpression(ctx);
        if(this.current() != "*"){
            return trm1;
        }

        while(this.current() == "*"){
            let app = new App(operator(this.token.text), [trm1]);
            this.next();

            while(true){
                let trm2 = this.DivExpression(ctx);
                app.args.push(trm2);
                
                if(this.token.text == app.fncName){
                    this.next();
                }
                else{
                    trm1 = app;
                    break;
                }
            }
        }
    
        return trm1;
    }
    
    AdditiveExpression(ctx : Context) : Term {
        const trm1 = this.MultiplicativeExpression(ctx);

        if(this.token.text == "+" || this.token.text == "-"){
            let app = new App(operator("+"), [trm1]);

            while(this.token.text == "+" || this.token.text == "-"){
                const opr = this.token.text;
                this.next();

                const trm2 = this.MultiplicativeExpression(ctx);
                if(opr == "-"){
                    trm2.value.numerator *= -1;
                }

                app.addArgs(trm2);
            }

            return app;
        }

        return trm1;
    }

    ArithmeticExpression(ctx : Context) : Term {
        return this.AdditiveExpression(ctx);
    }

    RelationalExpression(ctx : Context) : Term {
        let trm1 = this.ArithmeticExpression(ctx);

        while(isRelationToken(this.token.text)){
            let app = new App(operator(this.token.text), [trm1]);
            this.next();

            while(true){
                let trm2 = this.ArithmeticExpression(ctx);
                app.args.push(trm2);
                
                if(this.token.text == app.fncName){
                    this.next();
                }
                else{
                    trm1 = app;
                    break;
                }
            }
        }

        return trm1;
    }

    AndExpression(ctx : Context) : Term {
        const trm1 = this.RelationalExpression(ctx);

        if(this.token.text != "&&"){

            return trm1;
        }

        const app = new App(operator("&&"), [trm1]);

        while( [";", "&&"].includes(this.token.text) ){
            this.next();

            const trm2 = this.RelationalExpression(ctx);
            app.addArgs(trm2);
        }

        return app;
    }

    OrExpression(ctx : Context) : Term {
        const trm1 = this.AndExpression(ctx);

        if(this.current() != "||"){

            return trm1;
        }

        const app = new App(operator("||"), [trm1]);

        while( this.current() == "||" ){
            this.next();

            const trm2 = this.AndExpression(ctx);
            app.addArgs(trm2);
        }

        return app;
    }

    LogicalExpression(ctx : Context){
        const trm1 = this.OrExpression(ctx);

        if([ "=>", "⇔" ].includes(this.token.text)){
            const opr = this.token.text;

            this.next();

            let trm2 = this.OrExpression(ctx);
            return new App(operator(opr), [trm1, trm2]);    
        }
        else{
            
            return trm1;
        }
    }

    readVariable(ctx : Context, varKind:string, mod : Modifier, parent : Struct | undefined) : Variable {
        const name = this.readToken(TokenType.identifier);
        
        let type : Type | undefined;
        if(this.current() == ":"){
            this.nextToken(":");
            type = this.readType(ctx);
        }

        let initializer: Term | undefined;
        if (this.current() === '=') {
            this.nextToken("=");
            initializer = this.ArithmeticExpression(ctx);
        }
        
        if(parent == undefined){

            return  new Variable(varKind, mod, name, type, initializer);
        }
        else{

            return  new Field(mod, name, type, initializer, parent);
        }

    }

    parseVariableDeclaration(ctx : Context) : Variable {
        const mod = this.readModifiers();

        assert(["let", "var", "const"].includes(this.token.text));
        const varKind = this.token.text;
        this.next();

        if(this.current() == "<"){

            this.nextToken("<");
            while(true){
                const buf_attr = this.readToken(TokenType.reservedWord);
                switch(buf_attr){
                case "uniform":
                    mod.usage = BufferUsage.uniform;
                    break;
                case "storage":
                    mod.usage = BufferUsage.storage;
                    break;
                case "read":
                    mod.readWrite = BufferReadWrite.storage_read;
                    break;
                case "read_write":
                    mod.readWrite = BufferReadWrite.storage_read_write;
                    break;
                default:
                    throw new MyError();
                }

                if(this.current() == ","){
                    this.nextToken(",");
                }
                else{
                    break;
                }
            }
            this.nextToken(">");
        }

        return this.readVariable(ctx, varKind, mod, undefined);
    }

    parseReturn(ctx : Context) : ReturnStatement {
        this.nextToken("return");
        let value : Term | undefined;
        if(this.current() != ";"){
            value = this.LogicalExpression(ctx);
        }
        this.nextToken(";");

        return new ReturnStatement(value);
    }

    parseAssignment(ctx : Context) : App {
        const trm1 = this.PrimaryExpression(ctx);

        if(isAssignmentToken(this.current())){
            const opr = this.current();
            this.next();
            const trm2 = this.ArithmeticExpression(ctx);

            this.nextToken(";");
            return new App(operator(opr), [trm1, trm2]);
        }
        else if(trm1 instanceof App){
            this.nextToken(";");
            return trm1;
        }
        else{
            throw new MyError();
        }
    }

    parseBlock(ctx : Context): BlockStatement {
        this.nextToken("{");

        const statements: Statement[] = [];
        while (this.current() !== '}' && !this.isEoT()) {
            statements.push(this.parseStatement(ctx));
        }

        this.nextToken("}");

        return new BlockStatement(statements);
    }

    parseIf(ctx : Context): IfStatement {
        this.nextToken("if");

        this.nextToken("(");
        const condition = this.OrExpression(ctx);
        this.nextToken(")");

        const ifBlock = this.parseBlock(ctx);

        let elseIf : IfStatement | undefined;
        let elseBlock: BlockStatement | undefined;
        if (this.current() === 'else') {
            this.nextToken('else');

            if (this.current() === 'if') {
                elseIf = this.parseIf(ctx);
            } else {
                elseBlock = this.parseBlock(ctx);
            }
        }

        return new IfStatement(condition, ifBlock, elseIf, elseBlock);
    }

    parseWhile(ctx : Context): WhileStatement {
        this.nextToken("while");
        this.nextToken("(");
        const condition = this.LogicalExpression(ctx);
        this.nextToken(')');        

        const body = this.parseBlock(ctx);

        return new WhileStatement(condition, body);
    }

    parseFor(ctx : Context): ForStatement {
        this.nextToken("for");
        this.nextToken("(");

        const initializer = this.parseVariableDeclaration(ctx);

        let condition : Term | undefined;
        let update    : Term | undefined;

        if(this.current() == "of"){

            this.nextToken('of');

            this.ArithmeticExpression(ctx);
        }
        else{

            this.nextToken(';');

            condition = this.LogicalExpression(ctx);
            this.nextToken(';');

            update = this.parseAssignment(ctx);
        }

        this.nextToken(')');        

        const body = this.parseBlock(ctx);

        return new ForStatement(initializer, condition, update, body);
    }

    parseParallel(ctx : Context): ParallelStatement {
        this.nextToken("parallel");
        this.nextToken("{");

        const blocks : BlockStatement[] = [];
        while(this.current() != "}"){
            const block = this.parseBlock(ctx);
            blocks.push(block);
        }

        this.nextToken("}");

        return new ParallelStatement(blocks);
    }

    parseStatement(ctx : Context) : Statement{
        if(["let", "var", "const"].includes(this.token.text)){
            const variable = this.parseVariableDeclaration(ctx);
            this.nextToken(";");

            return new VariableDeclaration(variable);
        }
        else if(this.token.text == "return"){
            return this.parseReturn(ctx);
        }
        else if(this.token.text == "if"){
            return this.parseIf(ctx);
        }
        else if(this.token.text == "while"){
            return this.parseWhile(ctx);
        }
        else if(this.token.text == "for"){
            return this.parseFor(ctx);
        }
        else if(this.token.text == "parallel"){
            return this.parseParallel(ctx);
        }
        else{
    
            const app = this.parseAssignment(ctx);
            return new CallStatement(app);
        }
    }

    parseSource() : void {
        const ctx = Context.unknown;

        while(!this.isEoT()){
            switch(this.current()){
            case "[":{
                this.nextToken("[");
                const name = this.readName();
                assert(name == "gpu" || name == "cpu");
                this.nextToken("]");
                break;
                }
                
            case "struct":
                this.structDeclaration(ctx);
                break;

            case "@group":
            case "var":
            case "let":
            case "const":{
                const isGroup = this.current() == "@group";
                const lineText = this.lineText();
                const variable = this.parseVariableDeclaration(ctx);
                if(isGroup){
                    this.vars.push(variable);
                }
                this.nextToken(";");
                break;
            }
                
            case "@compute":
            case "@vertex":
            case "@fragment":
            case "fn":{
                const isEntry = this.current() != "fn";
                const fn = this.readFn(ctx);
                if(isEntry){
                    this.fns.push(fn);
                }
                break;
            }

            default:
                throw new MyError();
            }
        }
    }
}
