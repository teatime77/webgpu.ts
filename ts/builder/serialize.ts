// ============================================================================
// Serializers for the SimulationBundle
// ----------------------------------------------------------------------------
// `toJsonText` emits a stable, human-readable JSON string that the engine and
// the schema validator already accept verbatim.
// `toDslText` emits text that round-trips through the existing DSL parser
// in `ts/parser.ts`.
// ============================================================================

import type { SimulationBundle } from "./index";
import type { DslStatement, DslArg } from "./dsl";

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

export function toJsonText(bundle: SimulationBundle, indent = 2): string {
    return JSON.stringify(bundle.schema, null, indent) + "\n";
}

// ---------------------------------------------------------------------------
// DSL text
// ---------------------------------------------------------------------------

export function toDslText(bundle: SimulationBundle): string {
    const out: string[] = [];
    emitStatements(bundle.dsl, out, 0);
    return out.join("");
}

function emitStatements(stmts: DslStatement[], out: string[], depth: number) {
    const pad = "    ".repeat(depth);
    for (const s of stmts) {
        switch (s.kind) {
            case "call":
                out.push(`${pad}${s.name}(${s.args.map(formatArg).join(", ")});\n`);
                break;
            case "yield":
                out.push(`${pad}yield;\n`);
                break;
            case "while-true":
                out.push(`${pad}while(true) {\n`);
                emitStatements(s.body, out, depth + 1);
                out.push(`${pad}}\n`);
                break;
            case "for-range":
                out.push(`${pad}for(const ${s.iter} of range(${formatArg(s.count)})) {\n`);
                emitStatements(s.body, out, depth + 1);
                out.push(`${pad}}\n`);
                break;
            case "blank":
                out.push("\n");
                break;
        }
    }
}

function formatArg(arg: DslArg): string {
    switch (arg.kind) {
        case "number":
            return String(arg.value);
        case "ref":
            return arg.name;
        case "metaref":
            return `metadata.${arg.name}`;
    }
}
