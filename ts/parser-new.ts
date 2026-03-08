import { assert, msg, MyError  } from "@i18n";
import { Modifier, Struct, Token, TokenSubType, TokenType, Type, Variable, Fn, Field, BufferUsage, Module, ShaderType, isLetter, BufferReadWrite } from "./parser.js";
import { error } from "./util.js";

export const indexOpr = "t[i]";

function isRelationToken(text : string){
    return [ "==", "=", "!=", "<", ">", "<=", ">=", "in", "notin", "subset" ].includes(text);
}

function isAssignmentToken(text : string){
    return ["=", "+=", "-=", "*=", "/=", "%="].includes(text);
}

function operator(opr : string) : RefVar {
    return new RefVar(opr);
}

function getAllSub(alls: (Statement | Term)[], ...args:(Term | Statement | Term[] | Statement[] | undefined)[]){
    for(const arg of args){
        if(arg instanceof Term){
            alls.push(arg);
            arg.getAll(alls);
        }
        else if(arg instanceof Statement){
            alls.push(arg);
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


    toString() : string {
        if(this.denominator == 1){

            return `${this.numerator}`;
        }
        else{

            return `${this.numerator} / ${this.denominator}`;
        }
    }
}

export abstract class Term {
    parent : App | null = null;
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

    getAll(alls: (Statement | Term)[]) : void {
        alls.push(this);
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
    refVar! : Variable | undefined;

    constructor(name: string){
        super();
        this.name = name;
    }

    dumpTerm() : void {
        msg(`pre-pare:${this.constructor.name}:${this.name}`);
    }

    toString() : string {
        return this.name;
    }
}


export class ConstNum extends Term{
    static zero() : ConstNum {
        return new ConstNum(0);
    }

    constructor(numerator : number, denominator : number = 1){
        super();
        this.value = new Rational(numerator, denominator);
    }

    dumpTerm() : void {
        msg(`pre-pare:${this.constructor.name}:${this.value}`);
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
    
    addArg(trm : Term){
        this.args.push(trm);
        trm.parent = this;
    }

    precedence() : number {
        switch(this.fncName){
        case "^": 
            return 0;

        case "/": 
            return 1;

        case "*": 
            return 2;

        case "+": 
        case "-": 
            return 3;
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
        if(isLetter(this.fncName)){
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

                case indexOpr:
                    return `${args[0]}[${args.slice(1).join(", ")}]`
            }

            text = args.join(` ${this.fncName} `);
        }

        if(this.isOperator() && this.parent != null && this.parent.isOperator()){
            if(this.parent.precedence() <= this.precedence()){
                return `(${text})`;
            }            
        }

        return text;
    }

    getAll(alls: (Statement | Term)[]) : void{ 
        alls.push(this);
        if(isLetter(this.fncName[0])){
            alls.push(this.fnc);
        }
        getAllSub(alls, this.args);
    }
}


export abstract class Statement {
    dumpStatement() : void {
        throw new MyError();
    }

    isAssignmentCall() : boolean {
        return this instanceof CallStatement && this.app.isAssignmentApp();
    }

    abstract getAll(alls: (Statement | Term)[]) : void;
}


export class VariableDeclaration extends Statement {
    variable: Variable;

    constructor(variable: Variable) {
        super();
        this.variable = variable;
    }

    dumpStatement() : void {
        msg(`pre-pare:${this.constructor.name}`);
    }

    getAll(alls: (Statement | Term)[]) : void{ 
        alls.push(this);
        getAllSub(alls, this.variable.initializer);
    }
}

export class CallStatement extends Statement {
    app : App;

    constructor(app : App){
        super();
        this.app = app;
    }

    dumpStatement() : void {
        msg(`pre-pare:${this.constructor.name}:${this.app}`);
        this.app.dumpTerm();
    }

    getAll(alls: (Statement | Term)[]) : void{  
        alls.push(this);
        getAllSub(alls, this.app);
    }
}

export class ReturnStatement extends Statement {
    value : Term | undefined;

    constructor(value : Term | undefined){
        super();
        this.value = value;
    }

    getAll(alls: (Statement | Term)[]) : void{     
        alls.push(this);
        getAllSub(alls, this.value);
    }
}

export class BlockStatement extends Statement {
    statements : Statement[];

    constructor(statements : Statement[]){
        super();
        this.statements = statements.slice();
    }

    dumpStatement() : void {
        msg(`pre-pare:${this.constructor.name}`);
        this.statements.forEach(x => x.dumpStatement());
    }

    getAll(alls: (Statement | Term)[]) : void{  
        alls.push(this);
        getAllSub(alls, this.statements);
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

    getAll(alls: (Statement | Term)[]) : void{        
        alls.push(this);
        getAllSub(alls, this.condition, this.ifBlock, this.elseIf, this.elseBlock);
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

    dumpStatement() : void {
        msg(`pre-pare:${this.constructor.name}`);
        this.condition.dumpTerm();
        this.block.dumpStatement();
    }

    getAll(alls: (Statement | Term)[]) : void{        
        alls.push(this);
        getAllSub(alls, this.condition, this.block);
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

    getAll(alls: (Statement | Term)[]) : void{        
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

    getAll(alls: (Statement | Term)[]) : void{        
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

    readType() : Type {
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
                const elementType = this.readType();
                this.nextToken(">");
                return new ShaderType(mod, aggregate, elementType);
        }

        const type_name = this.readToken(TokenType.type);

        const struct = this.structs.find(x => x.typeName == type_name);
        if(struct == undefined){

            return new Type(mod, type_name);
        }
        else{
            assert(mod.empty());
            return struct;
        }
    }

    readGenericType(app : App){
        this.nextToken("<");

        while(true){
            const typeName = this.readToken(TokenType.type);
            const refVar = new RefVar(typeName);
            app.args.push(refVar);

            if(this.token.text == ","){
                this.nextToken(",");
            }
            else{
                break;
            }
        }

        this.nextToken(">");
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
            const field  = this.readVariable(ctx, mod, struct) as Field;
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
            const variable  = this.readVariable(ctx, modVar, undefined);
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

            fn.type = this.readType();
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
        const name = this.readName();

        let term : RefVar | App = new RefVar(name);

        if(idType == TokenType.type && this.token.text == '<'){
            term = new App(operator("<>"), [term]);
            this.readGenericType(term);
        }

        while(true){
            if(this.token.text == '('){

                term = new App(operator("()"), [term]);
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
            let n : number;

            if(this.token.text[0] == "#"){
                n = parseInt(this.token.text.substring(1), 16);
            }
            else{
                n = parseFloat(this.token.text);
            }

            if(isNaN(n)){
                throw new MyError();
            }

            trm = new ConstNum(n);
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
        let nagative : boolean = false;
        if(this.token.text == "-"){
            nagative = true;
            this.next();
        }

        const trm1 = this.MultiplicativeExpression(ctx);
        if(nagative){
            trm1.value.numerator *= -1;
        }

        if(this.token.text == "+" || this.token.text == "-"){
            let app = new App(operator("+"), [trm1]);

            while(this.token.text == "+" || this.token.text == "-"){
                const opr = this.token.text;
                this.next();

                const trm2 = this.MultiplicativeExpression(ctx);
                if(opr == "-"){
                    trm2.value.numerator *= -1;
                }

                app.addArg(trm2);
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
            app.addArg(trm2);
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
            app.addArg(trm2);
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

    readVariable(ctx : Context, mod : Modifier, parent : Struct | undefined) : Variable {
        const name = this.readToken(TokenType.identifier);
        
        let type : Type | undefined;
        if(this.current() == ":"){
            this.nextToken(":");
            type = this.readType();
        }

        let initializer: Term | undefined;
        if (this.current() === '=') {
            this.nextToken("=");
            initializer = this.ArithmeticExpression(ctx);
        }
        
        if(parent == undefined){

            return  new Variable(mod, name, type, initializer);
        }
        else{

            return  new Field(mod, name, type, initializer, parent);
        }

    }

    parseVariableDeclaration(ctx : Context) : Variable {
        const mod = this.readModifiers();

        assert(["let", "var", "const"].includes(this.token.text));
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

        return this.readVariable(ctx, mod, undefined);
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
            this.next();
            const trm2 = this.ArithmeticExpression(ctx);

            this.nextToken(";");
            return new App(operator("="), [trm1, trm2]);
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
