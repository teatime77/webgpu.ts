// ============================================================================
// Simulation Builder — typed, AI-friendly authoring API
// ----------------------------------------------------------------------------
// Authors call `defineSimulation({...})` with a strongly-typed spec. The result
// is a `SimulationBundle` that the serializer can turn into the runtime
// JSON + DSL artifacts (no TypeScript ever reaches the browser).
//
// Pure module: no GPU, no DOM, no Node. Safe to import in browser or Node.
// ============================================================================

import type { WgslFormat, BindingDefinition, UiDef } from "../control";
import { type DslStatement, type ScriptCtx, runScript } from "./dsl";

// ---------------------------------------------------------------------------
// Re-exported types
// ---------------------------------------------------------------------------

export type Format = WgslFormat;
export type Access = "read" | "read_write";
export type Topology = "point-list" | "line-list" | "triangle-list";
export type TextureDimension = "1d" | "2d" | "3d";

export type MetaValue = number | string | boolean | number[];
export type MetaSpec = Record<string, MetaValue>;

/** Resource id keys from a `resources` map (used to type bindings and `swap`). */
export type ResourceId<R extends Record<string, ResourceObj>> = Extract<keyof R, string>;

/**
 * Builds a schema expression referencing `metadata[key]` (the engine resolves `$metadata.*`).
 * Prefer this over raw strings so renames are easier to track.
 */
export function dollarMetadata<K extends string>(key: K): `$metadata.${K}` {
    return `$metadata.${key}`;
}

// ---------------------------------------------------------------------------
// Resource builders — return tagged JSON shapes the serializer consumes
// ---------------------------------------------------------------------------

export interface ResourceObj {
    readonly __kind: "resource";
    readonly def: Record<string, unknown>;
}

export function storage(opts: {
    format: Format;
    count: number | string;
    access?: Access;
    bufferCount?: number;
}): ResourceObj {
    const def: Record<string, unknown> = { type: "storage" };
    if (opts.access !== undefined) def.access = opts.access;
    def.format = opts.format;
    def.count = opts.count;
    if (opts.bufferCount !== undefined) def.bufferCount = opts.bufferCount;
    return { __kind: "resource", def };
}

export function uniform(fields: Record<string, Format>): ResourceObj {
    return { __kind: "resource", def: { type: "uniform", fields: { ...fields } } };
}

export function indirect(opts: {
    format: Format;
    count: number | string;
    access?: Access;
    bufferCount?: number;
}): ResourceObj {
    const def: Record<string, unknown> = { type: "indirect" };
    if (opts.access !== undefined) def.access = opts.access;
    def.format = opts.format;
    def.count = opts.count;
    if (opts.bufferCount !== undefined) def.bufferCount = opts.bufferCount;
    return { __kind: "resource", def };
}

export function texture(opts: {
    dimension?: TextureDimension;
    format?: string;
    size?: number[];
    access?: Access;
    bufferCount?: number;
} = {}): ResourceObj {
    const def: Record<string, unknown> = { type: "texture" };
    if (opts.access !== undefined) def.access = opts.access;
    if (opts.format !== undefined) def.format = opts.format;
    if (opts.dimension !== undefined) def.dimension = opts.dimension;
    if (opts.size !== undefined) def.size = [...opts.size];
    if (opts.bufferCount !== undefined) def.bufferCount = opts.bufferCount;
    return { __kind: "resource", def };
}

export function sampler(): ResourceObj {
    return { __kind: "resource", def: { type: "sampler" } };
}

// ---------------------------------------------------------------------------
// Node builders
// ---------------------------------------------------------------------------

export type WorkgroupDim = number | string;

export interface BindingSpec<R extends string = string> extends Omit<BindingDefinition, "binding" | "resource"> {
    binding: number;
    resource: R;
}

export interface ComputeOpts<R extends string = string> {
    shader: string;
    /** Direct dispatch — the engine derives [X,Y,Z]. Mutually exclusive with `indirectBuffer`. */
    workgroups?: [WorkgroupDim, WorkgroupDim, WorkgroupDim] | string;
    /** Indirect dispatch — references a buffer resource. */
    indirectBuffer?: R;
    indirectOffset?: number;
    bindings: BindingSpec<R>[];
}

export interface RenderOpts<R extends string = string> {
    shader: string;
    topology: Topology;
    depthTest?: boolean;
    /** Direct draw — vertex / instance counts. Mutually exclusive with `indirectBuffer`. */
    vertexCount?: number | string;
    instanceCount?: number | string;
    /** Indirect draw. */
    indirectBuffer?: R;
    indirectOffset?: number;
    bindings: BindingSpec<R>[];
}

export interface NodeBuilder<R extends string = string> {
    readonly __kind: "node";
    /** Filled in by `defineSimulation` from the record key. */
    readonly nodeKind: "compute" | "render";
    readonly opts: ComputeOpts<R> | RenderOpts<R>;
}

export function compute<R extends string = string>(opts: ComputeOpts<R>): NodeBuilder<R> {
    return { __kind: "node", nodeKind: "compute", opts };
}

export function render<R extends string = string>(opts: RenderOpts<R>): NodeBuilder<R> {
    return { __kind: "node", nodeKind: "render", opts };
}

// ---------------------------------------------------------------------------
// UI helpers — shaped exactly like UiDef
// ---------------------------------------------------------------------------

export const ui = {
    range(spec: Omit<UiDef, "type"> & Partial<Pick<UiDef, "type">>): UiDef {
        return { ...spec, type: "range" };
    },
    int(spec: Omit<UiDef, "type">): UiDef {
        return { ...spec, type: "int" };
    },
    checkbox(spec: Omit<UiDef, "type">): UiDef {
        return { ...spec, type: "checkbox" };
    },
    select(spec: Omit<UiDef, "type">): UiDef {
        return { ...spec, type: "select" };
    },
    button(spec: Omit<UiDef, "type">): UiDef {
        return { ...spec, type: "button" };
    },
};

// ---------------------------------------------------------------------------
// defineSimulation — the orchestrator
// ---------------------------------------------------------------------------

export interface SimulationBundle {
    /** Final JSON-shaped schema (matches `SimulationSchema`). */
    readonly schema: {
        name: string;
        version: string;
        uis?: UiDef[];
        metadata: Record<string, MetaValue>;
        resources: Record<string, Record<string, unknown>>;
        nodes: Record<string, unknown>[];
    };
    /** DSL AST captured from the script callback. */
    readonly dsl: DslStatement[];
}

export interface DefineSimulationSpec<
    M extends MetaSpec,
    R extends Record<string, ResourceObj>,
    N extends Record<string, NodeBuilder<ResourceId<R>>>,
> {
    name: string;
    version: string;
    uis?: UiDef[];
    metadata: M;
    resources: R;
    /** Node id is the record key. The builder injects it as `id` in the emitted JSON. */
    nodes: N;
    /** Imperative DSL describing per-frame execution. */
    script: (ctx: ScriptCtx<keyof N & string, keyof M & string, ResourceId<R>>) => void;
}

export function defineSimulation<
    M extends MetaSpec,
    R extends Record<string, ResourceObj>,
    N extends Record<string, NodeBuilder<ResourceId<R>>>,
>(spec: DefineSimulationSpec<M, R, N>): SimulationBundle {
    // Build resources record (preserve insertion order).
    const resources: Record<string, Record<string, unknown>> = {};
    for (const [id, r] of Object.entries(spec.resources)) {
        resources[id] = r.def;
    }

    // Build nodes array, injecting id from the key.
    const nodes: Record<string, unknown>[] = [];
    for (const [id, n] of Object.entries(spec.nodes)) {
        nodes.push(buildNodeJson(id, n));
    }

    // Capture DSL.
    const dsl = runScript<keyof N & string, keyof M & string, ResourceId<R>>(
        spec.script,
        Object.keys(spec.nodes) as (keyof N & string)[],
        Object.keys(spec.metadata) as (keyof M & string)[],
        Object.keys(spec.resources) as ResourceId<R>[],
    );

    const schema: SimulationBundle["schema"] = {
        name: spec.name,
        version: spec.version,
        ...(spec.uis ? { uis: spec.uis } : {}),
        metadata: { ...spec.metadata },
        resources,
        nodes,
    };

    return { schema, dsl };
}

// ---------------------------------------------------------------------------
// Internal: turn a NodeBuilder into the JSON shape `control.ts` expects
// ---------------------------------------------------------------------------

function buildNodeJson(id: string, n: NodeBuilder): Record<string, unknown> {
    if (n.nodeKind === "compute") {
        const o = n.opts as ComputeOpts;
        const json: Record<string, unknown> = {
            id,
            type: "compute",
            shader: o.shader,
        };
        if (o.indirectBuffer !== undefined) {
            json.dispatch = {
                type: "indirect",
                buffer: o.indirectBuffer,
                ...(o.indirectOffset !== undefined ? { offset: o.indirectOffset } : {}),
            };
        } else if (o.workgroups !== undefined) {
            json.dispatch = { type: "direct", workgroups: o.workgroups };
        } else {
            throw new Error(`compute node "${id}" requires either workgroups or indirectBuffer.`);
        }
        json.bindings = o.bindings.map(b => ({ ...b }));
        return json;
    }

    const o = n.opts as RenderOpts;
    const json: Record<string, unknown> = {
        id,
        type: "render",
        shader: o.shader,
        topology: o.topology,
    };
    if (o.depthTest !== undefined) json.depthTest = o.depthTest;
    if (o.indirectBuffer !== undefined) {
        json.draw = {
            type: "indirect",
            buffer: o.indirectBuffer,
            ...(o.indirectOffset !== undefined ? { offset: o.indirectOffset } : {}),
        };
    } else if (o.vertexCount !== undefined) {
        const draw: Record<string, unknown> = { type: "direct", vertexCount: o.vertexCount };
        if (o.instanceCount !== undefined) draw.instanceCount = o.instanceCount;
        json.draw = draw;
    } else {
        throw new Error(`render node "${id}" requires either vertexCount or indirectBuffer.`);
    }
    json.bindings = o.bindings.map(b => ({ ...b }));
    return json;
}

// ---------------------------------------------------------------------------
// Convenience: re-export DSL types for end-users
// ---------------------------------------------------------------------------

export type { ScriptCtx, DslStatement } from "./dsl";
