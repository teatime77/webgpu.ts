import { assert, fetchText, msg, MyError, unique } from "@i18n";
import { makeShaderModule } from "./util";
import { lexicalAnalysis } from "./lex";
import { Domain, BufferReadWrite, getUniformVars, getReadStorageVars, getWriteStorageVars, getStorageVars, App, CallStatement, indexOpr, RefVar, VariableDeclaration, Module, ConstNum } from "./syntax"
import { Context, Parser } from "./parser";
import { ComputePipeline } from "./compute";
import { RenderPipeline } from "./primitive";
import { makeComputeRenderPipelines, startAnimation } from "./instance";
import { ShapeInfo } from "./package";

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
    compShaderText! : string;

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

        const parser = new Parser(tokens, 0);
        this.parseScript(parser);
        this.prepare();
        await this.startScript();
    }

    parseScript(parser : Parser){
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
    }

    prepare(){
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

        this.compShaderText = 
              `${this.commonDomain}` 
            + "\n//" + "-".repeat(50) + " GPU\n"
            + `${this.gpuDomain}`;

        msg(`${this.compShaderText}`);
    }

    async startScript(){
        const gridSizeVar = this.commonDomain.vars.find(x => x.name == "gridSize");
        if(gridSizeVar == undefined || !(gridSizeVar.initializer instanceof ConstNum)) throw new MyError();
        const gridSize = gridSizeVar.initializer.int();
        const globalGrid = [gridSize];
        msg(`grid-size:${gridSize}`);

        const shapes : ShapeInfo[] = [
            {
                "type" : "Tube",
                "vertName" : "mesh-instance-vert",
                "fragName" : "phong-frag"
            }
        ]

        const [comp, comp_meshes] = makeComputeRenderPipelines(this.compShaderText, globalGrid, shapes);
        const comps : ComputePipeline[] = [comp];
        await startAnimation(comps, comp_meshes);
    }
}


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
        const module = new Module(text);
        const shaderModule = makeShaderModule(module.text);
        // mod.dump();
    }

    const script = new Script();
    await script.init();
}
