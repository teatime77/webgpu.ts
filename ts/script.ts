import { assert, fetchText, msg, MyError, unique } from "@i18n";
import { lexicalAnalysis, Fn as Fn, Variable, BufferUsage, Domain } from "./parser";
import { App, BlockStatement, CallStatement, ConstNum, Context, FncParser, indexOpr, RefVar, Statement, Str, Term, VariableDeclaration, WhileStatement } from "./parser-new";
import { ComputePipeline } from "./compute";
import { RenderPipeline } from "./primitive";

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
    commonDomain : Domain;
    cpuDomain    : Domain;
    gpuDomain    : Domain;

    comps : ComputePipeline[] = [];
    meshes: RenderPipeline[] = [];

    constructor(){
        this.commonDomain = new Domain();
        this.cpuDomain    = new Domain();
        this.gpuDomain    = new Domain();
    }

    async init(){
        msg(`\n------------------------------ test`);
        const text = await fetchText(`./script/test.wgsl`);
        const tokens = lexicalAnalysis(text);

        const parser = new FncParser(tokens, 0);
        this.parseScript(parser);

        for(const domain of [this.commonDomain, this.cpuDomain, this.gpuDomain]){            
            domain.setParent();
            const alls = domain.getAllInDomain();
            const refs = alls.filter(x => x instanceof RefVar);
            const domains = domain == this.commonDomain ? [domain] : [this.commonDomain, domain];
            refs.forEach(x => x.resolveVariable(domains));
        }
    }

    parseScript(parser : FncParser){
        const ctx = Context.unknown;

        let domain = this.commonDomain;
        while (!parser.isEoT()) {
            const idx = [common, cpu, gpu].indexOf(parser.current());
            if(idx != -1){
                domain = [this.commonDomain, this.cpuDomain, this.gpuDomain][idx];
                parser.next();
            }

            if(parser.current() == "struct"){

                parser.structDeclaration(ctx);
            }
            else if(parser.current() == "fn"){

                const fn = parser.readFn(ctx);
                domain.fns.push(fn);
            }
            else{

                const stmt = parser.parseStatement(ctx);
                if(stmt instanceof VariableDeclaration){
                    domain.vars.push(stmt.variable);
                }
                else{
                    throw new MyError();
                }
            }
        }

        const alls = this.gpuDomain.getAllInDomain();

        const asns = alls.filter(x => x instanceof CallStatement && x.isAssignmentCall()) as CallStatement[];
        for(const asn of asns){
            const ref = getAssignmentTarget(asn.app);
            msg(`target:${ref}`);
        }

        const commonVariableNames = this.commonDomain.vars.map(x => x.name);
        let   refs = alls.filter(x => x instanceof RefVar && commonVariableNames.includes(x.name)) as RefVar[];
        const usedcommonVariableNames = unique(refs).map(x => x.name);
        const usedcommonVariables = this.commonDomain.vars.filter(x => usedcommonVariableNames.includes(x.name));
        msg(`com vars:${usedcommonVariables.map(x => x.name + ":" + BufferUsage[x.mod.usage]).join(", ")}`);
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
