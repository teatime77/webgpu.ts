import { ComputePipeline } from "./compute.js";
import { lexicalAnalysis, Modifier, Struct, Token, TokenSubType, TokenType, Type, Variable } from "./parser.js";
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
    initializer: Term | undefined;
    constructor(variable: Variable, initializer: Term | undefined) {
        super();
        this.variable = variable;
        this.initializer = initializer;
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

        const text = this.current();
        this.next();

        return text;
    }

    current(){
        return this.token.text;
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

    readArgs(start: string, end : string, app : App){
        this.nextToken(start);

        while(true){
            const trm = this.RelationalExpression();
            app.args.push(trm);

            if(this.token.text == ","){
                this.nextToken(",");
            }
            else{
                break;
            }
        }

        this.nextToken(end);
    }

    readId() : RefVar | App {
        const idType = this.token.typeTkn;
        assert(idType == TokenType.identifier || idType == TokenType.type);

        let term : RefVar | App = new RefVar(this.token.text);
        this.next();

        if(idType == TokenType.type && this.token.text == '<'){
            term = new App(operator("<>"), [term]);
            this.readGenericType(term);
        }

        while(true){
            if(this.token.text == '('){

                term = new App(operator("()"), [term]);
                this.readArgs("(", ")", term);
            }
            else if(this.token.text == '['){

                term = new App(operator("[]"), [term]);
                this.readArgs("[", "]", term);
            }
            else if(this.token.text == "."){
                this.nextToken(".");
                const id = this.readId();
                term = new App(operator("."), [term, id]);
            }
            else{

                return term;
            }
        }
    }

    PrimaryExpression() : Term {
        let trm : Term;

        if(this.token.typeTkn == TokenType.identifier || this.token.typeTkn == TokenType.type){
            return this.readId();
        }
        else if(this.token.typeTkn == TokenType.Number){
            let n = parseFloat(this.token.text);
            if(isNaN(n)){
                throw new MyError();
            }

            trm = new ConstNum(n);
            this.next();
        }
        else if(this.token.text == '('){

            this.next();
            trm = this.RelationalExpression();

            if(this.current() != ')'){
                throw new MyError();
            }
            this.next();

            if(this.token.text == '('){

                let app = new App(trm, []);
                this.readArgs("(", ")", app);

                return app;
            }

            return trm;
        }
        else if(this.token.text == '{'){

            this.next();
            const element = this.RelationalExpression();

            this.nextToken('|');

            const logic = this.LogicalExpression();

            this.nextToken('}');

            trm = new App(operator("{|}"), [element, logic]);
            return trm;
        }
        else{
            throw new MyError();
        }

        return trm;
    }

    PowerExpression() : Term {
        const trm1 = this.PrimaryExpression();
        if(this.token.text == "^"){

            this.nextToken("^");

            const trm2 = this.PowerExpression();

            return new App(operator("^"), [trm1, trm2]);
        }

        return trm1;
    }

    UnaryExpression() : Term {
        if (this.token.text == "-") {
            // 負号の場合

            this.nextToken("-");

            // 基本の式を読みます。
            const t1 = this.PowerExpression();

            // 符号を反転します。
            t1.value.numerator *= -1;

            return t1;
        }
        else {

            // 基本の式を読みます。
            return this.PowerExpression();
        }
    }

    
    DivExpression() : Term {
        let trm1 = this.UnaryExpression();
        while(this.token.text == "/" || this.token.text == "%"){
            let app = new App(operator(this.token.text), [trm1]);
            this.next();

            while(true){
                let trm2 = this.UnaryExpression();
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

    
    MultiplicativeExpression() : Term {
        let trm1 = this.DivExpression();
        if(this.current() != "*"){
            return trm1;
        }

        while(this.current() == "*"){
            let app = new App(operator(this.token.text), [trm1]);
            this.next();

            while(true){
                let trm2 = this.DivExpression();
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
    
    AdditiveExpression() : Term {
        let nagative : boolean = false;
        if(this.token.text == "-"){
            nagative = true;
            this.next();
        }

        const trm1 = this.MultiplicativeExpression();
        if(nagative){
            trm1.value.numerator *= -1;
        }

        if(this.token.text == "+" || this.token.text == "-"){
            let app = new App(operator("+"), [trm1]);

            while(this.token.text == "+" || this.token.text == "-"){
                const opr = this.token.text;
                this.next();

                const trm2 = this.MultiplicativeExpression();
                if(opr == "-"){
                    trm2.value.numerator *= -1;
                }

                app.addArg(trm2);
            }

            return app;
        }

        return trm1;
    }

    ArithmeticExpression() : Term {
        return this.AdditiveExpression();
    }

    RelationalExpression() : Term {
        let trm1 : Term;
        if(this.token.text == "["){

            const ref = new RefVar("[]");
            trm1 = new App(ref, []);
            this.readArgs("[", "]", trm1 as App);
        }
        else{

            trm1 = this.ArithmeticExpression();
        }

        while(isRelationToken(this.token.text)){
            let app = new App(operator(this.token.text), [trm1]);
            this.next();

            while(true){
                let trm2 = this.ArithmeticExpression();
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

    AndExpression() : Term {
        const trm1 = this.RelationalExpression();

        if(this.token.text != "&&"){

            return trm1;
        }

        const app = new App(operator("&&"), [trm1]);

        while( [";", "&&"].includes(this.token.text) ){
            this.next();

            const trm2 = this.RelationalExpression();
            app.addArg(trm2);
        }

        return app;
    }

    OrExpression() : Term {
        const trm1 = this.AndExpression();

        if(this.current() != "||"){

            return trm1;
        }

        const app = new App(operator("||"), [trm1]);

        while( this.current() == "||" ){
            this.next();

            const trm2 = this.AndExpression();
            app.addArg(trm2);
        }

        return app;
    }

    LogicalExpression(){
        const trm1 = this.OrExpression();

        if([ "=>", "⇔" ].includes(this.token.text)){
            const opr = this.token.text;

            this.next();

            let trm2 = this.OrExpression();
            return new App(operator(opr), [trm1, trm2]);    
        }
        else{
            
            return trm1;
        }
    }

    parseVariableDeclaration() : VariableDeclaration {
        assert(this.token.text == "var" || this.token.text == "const");
        this.next();
        const name = this.readToken(TokenType.identifier);
        
        let type: Type | undefined;
        if (this.current() === ':') {
            this.nextToken(":");
            type = this.readType();
        }
        
        const variable = new Variable(new Modifier(), name, type ?? new Type(new Modifier(), undefined, "inferred"));

        let initializer: Term | undefined;
        if (this.current() === '=') {
            this.nextToken("=");
            initializer = this.ArithmeticExpression();
        }
        this.nextToken(";");

        return new VariableDeclaration(variable, initializer);
    }

    parseReturn() : ReturnStatement {
        this.nextToken("return");
        let value : Term | undefined;
        if(this.current() != ";"){
            value = this.LogicalExpression();
        }
        this.nextToken(";");

        return new ReturnStatement(value);
    }

    parseAssignment() : App {
        const trm1 = this.PrimaryExpression();

        if(["=", "+=", "-=", "*=", "/=", "%="].includes(this.current())){
            this.next();
            const trm2 = this.ArithmeticExpression();

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

    parseBlock(): BlockStatement {
        this.nextToken("{");

        const statements: StatementApp[] = [];
        while (this.current() !== '}' && !this.isEoT()) {
            statements.push(this.parseStatement());
        }

        this.nextToken("}");

        return new BlockStatement(statements);
    }

    parseIf(): IfStatement {
        this.nextToken("if");

        this.nextToken("(");
        const condition = this.OrExpression();
        this.nextToken(")");

        const ifBlock = this.parseBlock();

        let elseIf : IfStatement | undefined;
        let elseBlock: BlockStatement | undefined;
        if (this.current() === 'else') {
            this.nextToken('else');

            if (this.current() === 'if') {
                elseIf = this.parseIf();
            } else {
                elseBlock = this.parseBlock();
            }
        }

        return new IfStatement(condition, ifBlock, elseIf, elseBlock);
    }

    parseFor(): ForStatement {
        this.nextToken("for");
        this.nextToken("(");

        const initializer = this.parseVariableDeclaration();
        this.nextToken(';');

        const condition = this.LogicalExpression();
        this.nextToken(';');

        const update = this.parseAssignment();

        this.nextToken(')');        

        const body = this.parseBlock();

        return new ForStatement(initializer, condition, update, body);
    }

    parseStatement() : Statement | App{
        if(this.token.text == "var" || this.token.text == "const"){
            return this.parseVariableDeclaration();
        }
        else if(this.token.text == "return"){
            return this.parseReturn();
        }
        else if(this.token.text == "if"){
            return this.parseIf();
        }
        else if(this.token.text == "for"){
            return this.parseFor();
        }
        else{
    
            return this.parseAssignment();
        }
    
    }
}
