// ============================================================================
// build/cli.ts — Node CLI that compiles a TypeScript simulation source file
// into the runtime JSON + DSL artifacts.
//
// Usage:
//   npx tsx webgpu/build/cli.ts <sim.ts> [--out <dir>]
//   npx tsx webgpu/build/cli.ts <sim.ts> --check  (compare against existing files)
//
// The TS sim file must `export default defineSimulation({...})` — the CLI never
// reaches the browser, so this stays Node-only.
// ============================================================================

import { writeFileSync, mkdirSync, existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve, basename, extname } from "node:path";
import { pathToFileURL } from "node:url";
import { argv, exit } from "node:process";

import type { SimulationBundle } from "../ts/builder/index";
import { toJsonText, toDslText } from "../ts/builder/serialize";
import { validateSchema, formatIssues } from "../ts/schema_validator";
import { isPhysicsEngineSchema } from "../ts/schema_public_path";

// Tiny tokenizer for the DSL semantic compare. Anything ignored here is also
// ignored by the project's parser. We deliberately don't import the real lexer
// because it transitively imports `@i18n` (a Vite path alias unresolvable in
// the bare Node runtime).
const DSL_TOKEN_RE = /[A-Za-z_]\w*|[0-9]+(?:\.[0-9]+)?|[{}();,.]/g;
function tokenizeDsl(src: string): string {
    return (src.match(DSL_TOKEN_RE) ?? []).join("|");
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

interface Args {
    simFile: string;
    outDir?: string;
    check: boolean;
}

function parseArgs(): Args {
    const a = argv.slice(2);
    let simFile: string | undefined;
    let outDir: string | undefined;
    let check = false;
    for (let i = 0; i < a.length; i++) {
        const t = a[i];
        if (t === "--out" || t === "-o") {
            outDir = a[++i];
        } else if (t === "--check") {
            check = true;
        } else if (!t.startsWith("-")) {
            simFile = t;
        } else {
            console.error(`Unknown flag: ${t}`);
            exit(2);
        }
    }
    if (!simFile) {
        console.error("Usage: tsx webgpu/build/cli.ts <sim.ts> [--out <dir>] [--check]");
        exit(2);
    }
    return { simFile, outDir, check };
}

// ---------------------------------------------------------------------------
// Diff helper
// ---------------------------------------------------------------------------

function deepEqual(a: unknown, b: unknown, path = "$"): string | null {
    if (a === b) return null;
    if (typeof a !== typeof b) return `${path}: type mismatch (${typeof a} vs ${typeof b})`;
    if (a === null || b === null) return `${path}: null mismatch`;
    if (Array.isArray(a) && Array.isArray(b)) {
        if (a.length !== b.length) return `${path}: array length ${a.length} vs ${b.length}`;
        for (let i = 0; i < a.length; i++) {
            const d = deepEqual(a[i], b[i], `${path}[${i}]`);
            if (d) return d;
        }
        return null;
    }
    if (typeof a === "object" && typeof b === "object") {
        const ao = a as Record<string, unknown>;
        const bo = b as Record<string, unknown>;
        const keys = new Set([...Object.keys(ao), ...Object.keys(bo)]);
        for (const k of keys) {
            if (!(k in ao)) return `${path}.${k}: present in expected but not produced`;
            if (!(k in bo)) return `${path}.${k}: produced but not in expected`;
            const d = deepEqual(ao[k], bo[k], `${path}.${k}`);
            if (d) return d;
        }
        return null;
    }
    return `${path}: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
    const args = parseArgs();
    const simPath = resolve(args.simFile);
    if (!existsSync(simPath)) {
        console.error(`File not found: ${simPath}`);
        exit(1);
    }

    // Dynamic import — tsx loader handles TS extensions transparently.
    const mod = await import(pathToFileURL(simPath).href);
    const bundle = (mod.default ?? mod.bundle) as SimulationBundle | undefined;
    if (!bundle || typeof bundle !== "object" || !("schema" in bundle)) {
        console.error(`${simPath} did not default-export a SimulationBundle.`);
        exit(1);
    }

    // Validate the emitted JSON shape (same gate the runtime applies).
    const issues = validateSchema(bundle.schema);
    const errors = issues.filter(i => i.level === "error");
    if (issues.length > 0) {
        console.log(formatIssues(issues, bundle.schema.name));
    }
    if (errors.length > 0) {
        console.error(`\nValidation failed: ${errors.length} error(s).`);
        exit(1);
    }

    const jsonText = toJsonText(bundle);
    const dslText = toDslText(bundle);

    if (args.check) {
        // Compare against canonical files under public/engines/physics/<id>/ or public/wgsl/<id>/.
        // We use the input TS filename — `schema.name` may be a long display name
        // (e.g. "Sphere GPU Physics") that doesn't match the short directory id.
        const baseName = basename(simPath, extname(simPath));
        const baseDir = isPhysicsEngineSchema(baseName)
            ? resolve("public/engines/physics", baseName)
            : resolve("public/wgsl", baseName);
        const jsonPath = join(baseDir, `${baseName}.json`);
        const dslPath = join(baseDir, `${baseName}.js`);

        let allOk = true;

        if (existsSync(jsonPath)) {
            const expected = JSON.parse(readFileSync(jsonPath, "utf8"));
            const produced = JSON.parse(jsonText);
            const diff = deepEqual(expected, produced);
            if (diff) {
                console.log(`JSON DIFF (${jsonPath}): ${diff}`);
                allOk = false;
            } else {
                console.log(`JSON ✓ semantically identical to ${jsonPath}`);
            }
        } else {
            console.log(`(skip JSON check; ${jsonPath} not found)`);
        }

        if (existsSync(dslPath)) {
            const expected = readFileSync(dslPath, "utf8");
            const normExpected = expected.replace(/\r\n/g, "\n").trimEnd();
            const normProduced = dslText.replace(/\r\n/g, "\n").trimEnd();
            // Token-stream compare: anything the parser ignores (whitespace,
            // blank lines, brace spacing) is ignored here too.
            const tokExpected = tokenizeDsl(normExpected);
            const tokProduced = tokenizeDsl(normProduced);
            if (normExpected === normProduced) {
                console.log(`DSL  ✓ textually identical to ${dslPath}`);
            } else if (tokExpected === tokProduced) {
                console.log(`DSL  ✓ token-stream identical to ${dslPath} (whitespace differs)`);
            } else {
                console.log(`DSL DIFF (${dslPath}):`);
                console.log("---- expected ----");
                console.log(expected);
                console.log("---- produced ----");
                console.log(dslText);
                allOk = false;
            }
        } else {
            console.log(`(skip DSL check; ${dslPath} not found)`);
        }

        exit(allOk ? 0 : 1);
    }

    // Otherwise, write to <outDir>/<name>.json and <outDir>/<name>.js
    const baseName = basename(simPath, extname(simPath));
    const outDir = args.outDir
        ? resolve(args.outDir)
        : isPhysicsEngineSchema(baseName)
            ? resolve("public/engines/physics", baseName)
            : resolve("public/wgsl", baseName);
    mkdirSync(outDir, { recursive: true });

    const jsonOut = join(outDir, `${baseName}.json`);
    const dslOut = join(outDir, `${baseName}.js`);
    writeFileSync(jsonOut, jsonText, "utf8");
    writeFileSync(dslOut, dslText, "utf8");
    console.log(`Wrote ${jsonOut}`);
    console.log(`Wrote ${dslOut}`);
}

main().catch(e => {
    console.error(e);
    exit(1);
});
