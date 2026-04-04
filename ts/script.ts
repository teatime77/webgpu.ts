import { $, $div, assert, fetchText, msg, MyError, parseURL } from "@i18n";
import { asyncBodyOnLoad, g_device, makeShaderModule } from "./util";
import { lexicalAnalysis } from "./lex";
import { Domain, BufferReadWrite, getUniformVars, getReadStorageVars, getWriteStorageVars, getStorageVars, App, CallStatement, indexOpr, RefVar, VariableDeclaration, Module, ConstNum } from "./syntax"
import { Context, Parser } from "./parser";
import { asyncBodyOnLoadCom, ComputePipeline } from "./compute";
import { RenderPipeline } from "./primitive";
import { asyncBodyOnLoadPackage, makeComputeRenderPipelines, startAnimation, startPackage, stopAnimation } from "./instance";
import { Package, ShapeInfo } from "./package";
import { asyncBodyOnLoadDemo } from "./demo";
import { asyncBodyOnLoadTex } from "./texture";
import { waitClick } from "./ui";
import { showElectrons } from "./electrons";
import { runHMC } from "./hmc";
import { runLGT } from "./lgt";
import { runFeynmanSimulation } from "./feynman";
import { runHiggs } from "./higgs";
import { runGaugeHiggs } from "./gauge_higgs";

const common = "@common";
const cpu    = "@cpu";
const gpu    = "@gpu";
type ScriptMode = "@common" | "@cpu" | "@gpu";
let testPackages : Package[];

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
    scriptName   : string;
    vertName     : string;
    commonDomain : Domain;
    cpuDomain    : Domain;
    gpuDomain    : Domain;
    compShaderText! : string;

    comps : ComputePipeline[] = [];
    meshes: RenderPipeline[] = [];

    constructor(scriptName : string, vertName : string){
        this.scriptName       = scriptName;
        this.vertName     = vertName;
        this.commonDomain = new Domain();
        this.cpuDomain    = new Domain();
        this.gpuDomain    = new Domain();
    }

    async init(){
        msg(`\n------------------------------ ${this.scriptName}`);
        const text = await fetchText(`./script/${this.scriptName}.wgsl`);
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
                "vertName" : this.vertName,
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
        "three-body-com",
        "tube-instance-vert",
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
        "electric-field",
        "hamiltonian-vert",
        "liouville-comp",
        "point-instance-vert"
    ];

    for(const shader_name of shader_names){
        msg(`\n------------------------------ ${shader_name}`);
        const text = await fetchText(`./wgsl/${shader_name}.wgsl`);
        const module = new Module(text);
        const shaderModule = makeShaderModule(module.text);
        // mod.dump();
    }
}


export async function asyncBodyOnLoadTestAll(){
    await asyncBodyOnLoadPackage("test");
    await asyncBodyOnLoadDemo();
    await asyncBodyOnLoadCom();
    await asyncBodyOnLoadTex();

    for(const [scriptName, vertName] of [["test", "mesh-instance-vert"], ["test2", "tube-instance-vert"]]){
        const script = new Script(scriptName, vertName);
        await script.init();
        await waitClick("next-button");
    }
}

window.addEventListener('load', async() => {
    const [ origin, pathname, params, url_base] = parseURL();
    
    console.log('画像も含めてすべてのロードが完了しました');
    await asyncBodyOnLoad();
    await parseAll();

    const test_text = await fetchText(`./package/test.json`);
    testPackages = JSON.parse(test_text) as Package[];

    makeButtons(params);
    console.log('初期化完了');
});

let divButtons = $div("span-buttons");
function makeButton( text : string) : HTMLButtonElement {
    const button = document.createElement("button");
    button.innerText = text;
    divButtons.appendChild(button);

    return button;
}

let currentStopFunction: () => void = () => {};

function stopCurrentAnimation() {
    stopAnimation(); // Stop the framework's default animation loop
    currentStopFunction(); // Stop any custom animation loop (like LGT)
    currentStopFunction = () => {}; // Reset the stopper
}

async function startTestPackage(name : string){
    stopCurrentAnimation();
    const pkg = testPackages.find(x => x.name == name);
    if(pkg == undefined){throw new MyError()};
    await startPackage(pkg);
}

function makeButtons(params: Map<string, string>){
    makeButton("電磁波").addEventListener("click", async() => { await startTestPackage("fdtd") });
    makeButton("電子雲").addEventListener("click", async() => { stopCurrentAnimation(); await showElectrons() });
    makeButton("Hopfのファイバー束").addEventListener("click", async() => { await startTestPackage("hopf") });
    makeButton("Liouvilleの定理").addEventListener("click", async() => { await startTestPackage("liouville") });
    makeButton("ハミルトンベクトル場").addEventListener("click", async() => { await startTestPackage("vector-field") });
    makeButton("三体").addEventListener("click", async() => { await startTestPackage("three-body") });
    makeButton("2D U(1)").addEventListener("click", async() => { 
        stopCurrentAnimation();
        currentStopFunction = await runLGT(g_device, "U1");
    });
    makeButton("2D U(1) vortex").addEventListener("click", async() => { 
        stopCurrentAnimation();
        currentStopFunction = await runLGT(g_device, "U1", "vortex");
    });
    makeButton("2D SU(2)").addEventListener("click", async() => { 
        stopCurrentAnimation();
        currentStopFunction = await runLGT(g_device, "SU2", "plaquette");
    });
    makeButton("2D SU(3)").addEventListener("click", async() => { 
        stopCurrentAnimation();
        currentStopFunction = await runLGT(g_device, "SU3", "plaquette");
    });
    makeButton("2D SU(3) wilson").addEventListener("click", async() => { 
        stopCurrentAnimation();
        currentStopFunction = await runLGT(g_device, "SU3", "wilson");
    });
    makeButton("Beta Decay").addEventListener("click", async() => { 
        stopCurrentAnimation();
        await runFeynmanSimulation(g_device);
    });
    makeButton("higgs").addEventListener("click", async() => { 
        stopCurrentAnimation();
        currentStopFunction = await runHiggs(g_device);
    });
    makeButton("SU(2) gauge higgs").addEventListener("click", async() => { 
        stopCurrentAnimation();
        currentStopFunction = await runGaugeHiggs(g_device, "SU2", "E");
    });
    makeButton("U(1) E gauge higgs").addEventListener("click", async() => { 
        stopCurrentAnimation();
        currentStopFunction = await runGaugeHiggs(g_device, "U1", "E");
    });
    makeButton("U(1) C gauge higgs").addEventListener("click", async() => { 
        stopCurrentAnimation();
        currentStopFunction = await runGaugeHiggs(g_device, "U1", "C");
    });

    makeButton("HMC").addEventListener("click", async() => { stopCurrentAnimation(); await runHMC() });

    if(params.has("debug")){
        makeButton("test all").addEventListener("click", async() => { stopCurrentAnimation(); await asyncBodyOnLoadTestAll() });
        makeButton("Stop").addEventListener("click", () => { stopCurrentAnimation() });
        $("div-color").style.display = "block";
        $("next-button").style.display = "inline-block";
    }
}
