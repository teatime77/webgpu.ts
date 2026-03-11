import { assert, fetchText, msg, MyError, unique } from "@i18n";
import { Domain, BufferReadWrite, getUniformVars, getReadStorageVars, getWriteStorageVars, getStorageVars, App, CallStatement, indexOpr, RefVar, VariableDeclaration } from "./syntax"
import { lexicalAnalysis } from "./parser";
import { Context, FncParser } from "./parser-new";
import { ComputePipeline } from "./compute";
import { RenderPipeline } from "./primitive";
import { makeShaderModule } from "./util";

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

                const struct = parser.structDeclaration(ctx);
                domain.structs.push(struct);
            }
            else if(["fn", "@compute"].includes(parser.current())){

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

        for(const domain of [this.commonDomain, this.cpuDomain, this.gpuDomain]){            
            domain.setParent();
            const alls = domain.getAllInDomain();
            const refs = alls.filter(x => x instanceof RefVar);
            const domains = domain == this.commonDomain ? [domain] : [this.commonDomain, domain];
            refs.forEach(x => x.resolveVariable(domains));
        }

        const alls = this.gpuDomain.getAllInDomain();

        const asns = alls.filter(x => x instanceof CallStatement && x.isAssignmentCall()) as CallStatement[];
        for(const asn of asns){
            const ref = getAssignmentTarget(asn.app);
            assert(ref.refVar != undefined);
            ref.refVar!.mod.readWrite = BufferReadWrite.storage_read_write;
            msg(`target:${ref}`);
        }

        getStorageVars(this.commonDomain)
            .filter( x => x.mod.readWrite == BufferReadWrite.unknown)
            .forEach(x => x.mod.readWrite = BufferReadWrite.storage_read);

        const uniformVars      = getUniformVars(this.commonDomain);
        const readStorageVars  = getReadStorageVars(this.commonDomain);
        const writeStorageVars = getWriteStorageVars(this.commonDomain);

        const uniformStorageVars = uniformVars.concat(readStorageVars, writeStorageVars);
        let binding = 0;
        for(const va of uniformStorageVars){
            va.mod.group = 0;
            va.mod.binding = binding;
            binding++;
        }

        msg("---------------------------------------- CPU");
        msg(`cpu domain:${this.cpuDomain}`);
        msg("---------------------------------------- Common");

        const compText = 
              `${this.commonDomain}` 
            + "\n//" + "-".repeat(50) + " GPU\n"
            + `${this.gpuDomain}`;

        msg(`${compText}`);

        const module = makeShaderModule(compText);
    }
}
