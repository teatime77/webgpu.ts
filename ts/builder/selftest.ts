/**
 * Runnable checks for serializer + DSL emitter (no test framework dependency).
 * Run: npx tsx ts/builder/selftest.ts
 */
import assert from "node:assert/strict";

import type { SimulationBundle } from "./index";
import { toDslText, toJsonText } from "./serialize";

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
    assert.equal(j.endsWith("\n"), true, "JSON text should end with newline");
    assert.deepEqual(JSON.parse(j), bundle.schema);
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
    assert.match(d, /while\(true\)/);
    assert.match(d, /for\(const i of range\(metadata\.iters\)\)/);
    assert.match(d, /tick\(\)/);
    assert.match(d, /yield;/);
    assert.ok(d.startsWith("\n") || d.includes("\n\n"), "blank statement should add a newline gap");
}

testJsonTrailingNewline();
testDslWhileYieldBlankForMeta();
console.log("ts/builder/selftest.ts: ok");
