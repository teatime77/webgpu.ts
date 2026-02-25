import { ComputePipeline } from "./compute.js";
import { lexicalAnalysis, Modifier, Struct, Token, TokenSubType, TokenType, Type, Variable, Function, Field } from "./parser.js";
import { assert, makeShaderModule, msg, MyError, sum, error, fetchText } from "./util.js";

type StatementApp = Statement | App;

function isRelationToken(text : string){
    return [ "==", "=", "!=", "<", ">", "<=", ">=", "in", "notin", "subset" ].includes(text);
}

function operator(opr : string) : RefVar {
    return new RefVar(opr);
}

export class Rational{
    numerator : number = 1;
    denominator : number = 1;
    parent : Term | null = null;

    constructor(numerator : number, denominator : number = 1){
        this.numerator = numerator;
        this.denominator = denominator;
    }
}

export abstract class Term {
    parent : App | null = null;
    // 係数
    value : Rational = new Rational(1);
}

class Str extends Term{
    text : string;

    constructor(text : string){
        super();
        this.text = text;
    }
}

export class RefVar extends Term{
    name: string;
    refVar! : Variable | undefined;

    constructor(name: string){
        super();
        this.name = name;
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
}

export class App extends Term{
    fnc : Term;
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
        if(this.fnc instanceof RefVar){
            return this.fnc.name;
        }
        else{
            return `no-fnc-name`;
        }
    }


    constructor(fnc: Term, args: Term[]){
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
}


export abstract class Statement {
}


export class VariableDeclaration extends Statement {
    variable: Variable;

    constructor(variable: Variable) {
        super();
        this.variable = variable;
    }
}

export class ReturnStatement extends Statement {
    value : Term | undefined;

    constructor(value : Term | undefined){
        super();
        this.value = value;
    }
}


export class BlockStatement extends Statement {
    statements : StatementApp[];

    constructor(statements : StatementApp[]){
        super();
        this.statements = statements.slice();
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
        this.ifBlock = ifBlock;
    }
}

export class WhileStatement extends Statement {
    condition: Term;
    body: BlockStatement;

    constructor(condition: Term, body: BlockStatement) {
        super();
        this.condition = condition;
        this.body = body;
    }
}

export class ForStatement extends Statement {
    initializer: VariableDeclaration;
    condition: Term;
    update: Term;
    body: BlockStatement;

    constructor(initializer: VariableDeclaration, condition: Term, update: Term, body: BlockStatement) {
        super();
        this.initializer = initializer;
        this.condition = condition;
        this.update = update;
        this.body = body;
    }
}

export class ParallelStatement extends Statement {
    blocks : BlockStatement[];

    constructor(blocks : BlockStatement[]){
        super();
        this.blocks = blocks;
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
    structs : Struct[] = [];

    constructor(tokens: Token[], tokenPos : number){
        this.tokenPos = tokenPos;
        this.tokens = tokens;
        this.token = this.tokens[this.tokenPos];
    }

    next(){
        if(this.tokens.length == 0){

            this.token = new Token(TokenType.eot, TokenSubType.unknown, "", 0);
        }
        else{

            this.tokenPos++;
            this.token = this.tokens[this.tokenPos];

            if(this.token.text == ";" || this.token.text == "{"){
                const i = this.tokens.findIndex((x, idx, arr) => this.tokenPos < idx && (x.text == ";" || x.text == "{"));
                if(i != -1){
                    msg(`line :${this.tokens.map(x => x.text).slice(this.tokenPos + 1, i + 1).join(" ")}`);
                }
            }
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

    readType() : Type {
        const mod = new Modifier();

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
                const primitive = this.readToken(TokenType.type);
                this.nextToken(">");
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

            const field  = this.readVariable(ctx, struct) as Field;
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


    readFn(ctx : Context) : Function {
        this.nextToken("fn");

        const name = this.readName();

        const fn = new Function(new Modifier(), name);

        this.nextToken("(");

        while(true){
            if(this.current() == ")"){
                break;
            }

            const variable  = this.readVariable(ctx, undefined);
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

        this.parseBlock(ctx, );

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
                    msg(`fnc arg[${trm.name}] with init`);
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

                term = new App(operator("[]"), [term]);
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
                return new App(operator(","), terms);
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

    readVariable(ctx : Context, parent : Struct | undefined) : Variable {
        const name = this.readToken(TokenType.identifier);
        
        let type: Type | undefined;
        if (this.current() === ':') {
            this.nextToken(":");
            type = this.readType();
        }

        let initializer: Term | undefined;
        if (this.current() === '=') {
            this.nextToken("=");
            initializer = this.ArithmeticExpression(ctx);
        }
        
        if(parent == undefined){

            return  new Variable(new Modifier(), name, new Type(new Modifier(), undefined, "inferred"), initializer);
        }
        else{

            return  new Field(new Modifier(), name, new Type(new Modifier(), undefined, "inferred"), initializer, parent);
        }

    }

    parseVariableDeclaration(ctx : Context) : VariableDeclaration {
        assert(this.token.text == "var" || this.token.text == "const");
        this.next();

        if(this.current() == "<"){
            this.nextToken("<");
            const varKind = this.readToken(TokenType.reservedWord);
            assert(varKind == "storage");
            this.nextToken(">");
        }

        const variable = this.readVariable(ctx, undefined);

        this.nextToken(";");
        return new VariableDeclaration(variable);
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

        if(["=", "+=", "-=", "*=", "/=", "%="].includes(this.current())){
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

        const statements: StatementApp[] = [];
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
        this.nextToken(';');

        const condition = this.LogicalExpression(ctx);
        this.nextToken(';');

        const update = this.parseAssignment(ctx);

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

    parseStatement(ctx : Context) : Statement | App{
        if(this.token.text == "var" || this.token.text == "const"){
            return this.parseVariableDeclaration(ctx);
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
    
            return this.parseAssignment(ctx);
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

            case "var":
                this.parseVariableDeclaration(ctx);
                break;
                
            case "fn":
                this.readFn(ctx);
                break;

            default:
                throw new MyError();
            }
        }
    }
}
