// ============================================================================
// DSL Recorder — captures imperative `script` callbacks as a typed AST that
// the serializer can emit as text matching the parser's grammar.
//
// Supported grammar (matches the existing DSL parser in control.ts):
//   - call statement:        nodeId();
//   - swapPingPong call:     swapPingPong(R1, R2, ...);
//   - yield:                 yield;
//   - while-true block:      while (true) { ... }
//   - for-range block:       for (const i of range(N)) { ... }
//                            for (const i of range(metadata.X)) { ... }
// ============================================================================

// ---------------------------------------------------------------------------
// AST shapes
// ---------------------------------------------------------------------------

export type DslStatement =
    | { kind: "call"; name: string; args: DslArg[] }
    | { kind: "yield" }
    | { kind: "while-true"; body: DslStatement[] }
    | { kind: "for-range"; iter: string; count: DslArg; body: DslStatement[] }
    /** Blank line — a no-op in the parser, used purely for readability in the emitted DSL. */
    | { kind: "blank" };

export type DslArg =
    | { kind: "number"; value: number }
    | { kind: "ref"; name: string }
    | { kind: "metaref"; name: string };

// MetaRef is a sentinel returned by the metadata proxy; for_(metadata.X) detects it.
export class MetaRef<K extends string = string> {
    readonly __metaRef = true;
    constructor(public readonly name: K) {}
}

export function isMetaRef(x: unknown): x is MetaRef {
    return typeof x === "object" && x !== null && (x as { __metaRef?: boolean }).__metaRef === true;
}

// ---------------------------------------------------------------------------
// Public ScriptCtx exposed to authors
// ---------------------------------------------------------------------------

export interface ScriptCtx<NodeIds extends string, MetaKeys extends string> {
    /** `call.<nodeId>()` dispatches the node by id. */
    readonly call: { [K in NodeIds]: () => void };
    /** Typed proxy of metadata names. `metadata.cg_iters` returns a sentinel for `for_`. */
    readonly metadata: { [K in MetaKeys]: MetaRef<K> };

    /** `while (true) { body }` */
    loop(body: () => void): void;
    /** `for (const i of range(count)) { body }`. `count` may be a literal number or `metadata.X`. */
    for_(count: number | MetaRef, body: () => void): void;
    /** `swapPingPong(a, b, ...)` — argument names are resource ids (strings). */
    swap(...resourceIds: string[]): void;
    /** `yield;` — flushes the GPU queue and returns control to the host. */
    yieldFrame(): void;
    /** Emit a blank line in the produced DSL text (purely cosmetic). */
    blank(): void;
}

// ---------------------------------------------------------------------------
// Recorder implementation
// ---------------------------------------------------------------------------

class Recorder {
    /** Stack of statement lists; top is the currently-open block. */
    private stack: DslStatement[][] = [[]];

    private push(stmt: DslStatement) {
        this.stack[this.stack.length - 1].push(stmt);
    }

    open(): DslStatement[] {
        const block: DslStatement[] = [];
        this.stack.push(block);
        return block;
    }
    close() {
        if (this.stack.length === 1) throw new Error("DSL recorder: cannot close root block.");
        this.stack.pop();
    }

    addCall(name: string, args: DslArg[] = []) {
        this.push({ kind: "call", name, args });
    }
    addYield() {
        this.push({ kind: "yield" });
    }
    addWhile(body: DslStatement[]) {
        this.push({ kind: "while-true", body });
    }
    addFor(iter: string, count: DslArg, body: DslStatement[]) {
        this.push({ kind: "for-range", iter, count, body });
    }
    addBlank() {
        this.push({ kind: "blank" });
    }

    get root(): DslStatement[] {
        return this.stack[0];
    }
}

// ---------------------------------------------------------------------------
// runScript — invoke the user callback against a recorder and return the AST
// ---------------------------------------------------------------------------

export function runScript<NodeIds extends string, MetaKeys extends string>(
    callback: (ctx: ScriptCtx<NodeIds, MetaKeys>) => void,
    nodeIds: NodeIds[],
    metaKeys: MetaKeys[],
): DslStatement[] {
    const rec = new Recorder();

    // call.<id>() — proxy that records a CallStatement
    const call = {} as { [K in NodeIds]: () => void };
    for (const id of nodeIds) {
        (call as Record<string, () => void>)[id] = () => rec.addCall(id);
    }

    // metadata.<key> — proxy that returns a MetaRef sentinel
    const metadata = {} as { [K in MetaKeys]: MetaRef<K> };
    for (const k of metaKeys) {
        Object.defineProperty(metadata, k, {
            value: new MetaRef(k),
            enumerable: true,
            writable: false,
        });
    }

    const ctx: ScriptCtx<NodeIds, MetaKeys> = {
        call,
        metadata,
        loop(body) {
            const block = rec.open();
            try {
                body();
            } finally {
                rec.close();
            }
            rec.addWhile(block);
        },
        for_(count, body) {
            const arg: DslArg = isMetaRef(count)
                ? { kind: "metaref", name: count.name }
                : { kind: "number", value: count as number };
            const block = rec.open();
            try {
                body();
            } finally {
                rec.close();
            }
            rec.addFor("i", arg, block);
        },
        swap(...resourceIds) {
            rec.addCall("swapPingPong", resourceIds.map(name => ({ kind: "ref", name }) as DslArg));
        },
        yieldFrame() {
            rec.addYield();
        },
        blank() {
            rec.addBlank();
        },
    };

    callback(ctx);
    return rec.root;
}
