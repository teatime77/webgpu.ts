import { assert, fetchText, msg, MyError, unique } from "@i18n";
import { lexicalAnalysis, Fn as Fn, Variable, BufferUsage } from "./parser";
import { App, BlockStatement, CallStatement, ConstNum, Context, FncParser, indexOpr, RefVar, Statement, Str, Term, VariableDeclaration, WhileStatement } from "./parser-new";

const common = "@common";
const cpu    = "@cpu";
const gpu    = "@gpu";
type ScriptMode = "@common" | "@cpu" | "@gpu";

function getAssignmentTarget(app : App) : RefVar {
    assert(app.isAssignmentApp());
    const trm1 = app.args[0];
    if(trm1 instanceof RefVar){

        return trm1;
    }
    if(trm1.isDot()){
        const trm2 = trm1.args[0];
        if(trm2 instanceof RefVar){
            return trm2;
        }
        else if(trm2.isApp(indexOpr)){
            if(trm2.args[0] instanceof RefVar){
                return trm2.args[0];
            }
        }
    }
    else if(trm1.isApp(indexOpr)){
        const trm2 = trm1.args[0];
        if(trm2 instanceof RefVar){
            return trm2;
        }
    }

    msg(`assign ERR: ${trm1}`);
    throw new MyError();
}

export class Script {
    mode : ScriptMode = common;
    commonVariables: Variable[] = [];
    cpuVariables: Variable[] = [];
    gpuVariables: Variable[] = [];

    commonStatements: Statement[] = [];
    cpuStatements: Statement[] = [];
    gpuStatements: Statement[] = [];

    gpuFns : Fn[] = [];

    async init(){
        msg(`\n------------------------------ test`);
        const text = await fetchText(`./script/test.wgsl`);
        const tokens = lexicalAnalysis(text);

        const parser = new FncParser(tokens, 0);
        this.parseScript(parser);

        const statements = [this.commonStatements, this.cpuStatements, this.gpuStatements].flat();
        for(const statement of statements){
            if(statement instanceof VariableDeclaration){
                const variable = statement.variable;

                msg(`var:${variable.name}`);
            }
            else if(statement instanceof WhileStatement){
                const condition = statement.condition;
                const body      = statement.block;

            }
            else{
                throw new MyError();
            }

            msg(`statement:${statement.constructor.name}`);
        }

        for(const fn of this.gpuFns){
            const alls = fn.getAll();
            const refs = alls.filter(x => x instanceof RefVar);
            msg(`refs:${refs.map(x => x.name).join(", ")}`);
        }



        statements.forEach(x => this.prepareStatement(x));
    }

    addVariable(variable: Variable){
        switch(this.mode){
        case common: this.commonVariables.push(variable); break;
        case cpu   : this.cpuVariables.push(variable); break;
        case gpu   : this.gpuVariables.push(variable); break;
        }
    }

    addStatement(stmt: Statement){
        switch(this.mode){
        case common: this.commonStatements.push(stmt); break;
        case cpu   : this.cpuStatements.push(stmt); break;
        case gpu   : this.gpuStatements.push(stmt); break;
        }
    }

    parseScript(parser : FncParser){
        const ctx = Context.unknown;

        while (!parser.isEoT()) {
            if([common, cpu, gpu].includes(parser.current())){
                this.mode = parser.current() as ScriptMode;
                parser.next();
            }

            if(parser.current() == "struct"){

                parser.structDeclaration(ctx);
            }
            else if(parser.current() == "fn"){

                const fn = parser.readFn(ctx);
                if(this.mode == gpu){
                    this.gpuFns.push(fn);
                    const alls = fn.getAll();
                    const asns = alls.filter(x => x instanceof CallStatement && x.isAssignmentCall()) as CallStatement[];
                    for(const asn of asns){
                        const ref = getAssignmentTarget(asn.app);
                        msg(`target:${ref}`);
                    }

                    const commonVariableNames = this.commonVariables.map(x => x.name);
                    let   refs = alls.filter(x => x instanceof RefVar && commonVariableNames.includes(x.name)) as RefVar[];
                    const usedcommonVariableNames = unique(refs).map(x => x.name);
                    const usedcommonVariables = this.commonVariables.filter(x => usedcommonVariableNames.includes(x.name));
                    msg(`com vars:${usedcommonVariables.map(x => x.name + ":" + BufferUsage[x.mod.usage]).join(", ")}`);
                }
            }
            else{

                const stmt = parser.parseStatement(ctx);
                if(stmt instanceof VariableDeclaration){
                    this.addVariable(stmt.variable);
                }
                else{
                    this.addStatement(stmt);
                }
            }
        }
    }


    prepareTerm(term : Term){
        if(term instanceof Str){
            
        }
        else if(term instanceof ConstNum){

        }
        else if(term instanceof RefVar){

        }
        else if(term instanceof App){
            this.prepareTerm(term.fnc);
            term.args.forEach(x => this.prepareTerm(x));
        }
        else{
            throw new MyError();
        }
    }

    prepareStatement(stmt : Statement){
        if(stmt instanceof VariableDeclaration){

        }
        else if(stmt instanceof BlockStatement){
            stmt.statements.forEach(x => this.prepareStatement(x));
        }
        else if(stmt instanceof WhileStatement){
            this.prepareTerm(stmt.condition);
            this.prepareStatement(stmt.block);
        }
        else if(stmt instanceof CallStatement){

        }
    }
}
