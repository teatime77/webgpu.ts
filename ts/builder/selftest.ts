/**
 * Runnable checks for serializer + DSL emitter (no test framework dependency).
 * Run: npx tsx ts/builder/selftest.ts
 */
import type { SimulationBundle } from "./index";
import { toDslText, toJsonText } from "./serialize";

function assert(cond: boolean, msg: string) {
    if (!cond) throw new Error(`selftest: ${msg}`);
}

function testJsonTrailingNewline() {
    const bundle: SimulationBundle = {
        schema: {
            name: "selftest",
            version: "0",
            metadata: { n: 1 },
            resources: {},
            nodes: [],
        },
        dsl: [{ kind: "yield" }],
    };
    const j = toJsonText(bundle);
    assert(j.endsWith("\n"), "JSON text should end with newline");
    assert(JSON.stringify(JSON.parse(j)) === JSON.stringify(bundle.schema), "JSON round-trip");
}

function testDslWhileYieldBlankForMeta() {
    const bundle: SimulationBundle = {
        schema: {
            name: "selftest",
            version: "0",
            metadata: { iters: 3 },
            resources: {},
            nodes: [],
        },
        dsl: [
            { kind: "blank" },
            {
                kind: "while-true",
                body: [
                    { kind: "for-range", iter: "i", count: { kind: "metaref", name: "iters" }, body: [{ kind: "call", name: "tick", args: [] }] },
                    { kind: "yield" },
                ],
            },
        ],
    };
    const d = toDslText(bundle);
    assert(/while\(true\)/.test(d), "while");
    assert(/for\(const i of range\(metadata\.iters\)\)/.test(d), "for metadata");
    assert(/tick\(\)/.test(d), "tick");
    assert(/yield;/.test(d), "yield");
    assert(d.startsWith("\n") || d.includes("\n\n"), "blank newline");
}

testJsonTrailingNewline();
testDslWhileYieldBlankForMeta();
console.log("ts/builder/selftest.ts: ok");
