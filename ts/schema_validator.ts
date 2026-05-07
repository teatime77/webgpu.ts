// ============================================================================
// Schema Validator with "Did you mean?" suggestions
// ----------------------------------------------------------------------------
// Validates a SimulationSchema (JSON from public/engines/physics/<name>/ or public/wgsl/<name>/)
// against the structural and semantic rules enforced by control.ts. Unknown keys,
// invalid enums, and dangling references all produce an issue with a Levenshtein-
// based suggestion of the closest valid name.
//
// Pure module: no GPU, no DOM. Safe to unit-test.
// ============================================================================

import type {
    SimulationSchema,
    ResourceDef,
    NodeDef,
    BindingDefinition,
    UiDef,
    WgslFormat,
} from "./control";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export type IssueLevel = "error" | "warning";

export interface ValidationIssue {
    level: IssueLevel;
    /** JSON path of the offending node, e.g. `resources.X.format` or `nodes[3].bindings[2].resource`. */
    path: string;
    message: string;
    /** Closest valid name (or names) when the issue is a typo / unknown identifier. */
    suggestion?: string;
}

export class SchemaValidationError extends Error {
    constructor(
        public readonly issues: ValidationIssue[],
        public readonly schemaName?: string,
    ) {
        super(formatIssues(issues, schemaName));
        this.name = "SchemaValidationError";
    }
}

// ---------------------------------------------------------------------------
// Vocabularies
// ---------------------------------------------------------------------------

const ROOT_KEYS = ["name", "version", "metadata", "resources", "nodes", "uis"] as const;
const ROOT_REQUIRED = ["name", "version", "metadata", "resources", "nodes"] as const;

const RESOURCE_TYPES = ["storage", "uniform", "indirect", "texture", "sampler"] as const;
const RESOURCE_KEYS_COMMON = ["type", "bufferCount"] as const;
const RESOURCE_KEYS_BY_TYPE: Record<string, readonly string[]> = {
    storage: [...RESOURCE_KEYS_COMMON, "access", "format", "count"],
    uniform: [...RESOURCE_KEYS_COMMON, "fields"],
    indirect: [...RESOURCE_KEYS_COMMON, "access", "format", "count"],
    texture: [...RESOURCE_KEYS_COMMON, "access", "format", "dimension", "size"],
    sampler: [...RESOURCE_KEYS_COMMON],
};

const WGSL_FORMATS: readonly WgslFormat[] = [
    "f32", "u32", "i32", "vec2<f32>", "vec3<f32>", "vec4<f32>",
    "mat4x4<f32>", "atomic<u32>", "atomic<i32>",
];

const NODE_TYPES = ["compute", "render", "reduction", "readback"] as const;
const NODE_KEYS_BASE = ["id", "type", "shader", "bindings"] as const;
const NODE_KEYS_BY_TYPE: Record<string, readonly string[]> = {
    compute: [...NODE_KEYS_BASE, "dispatch"],
    render: [...NODE_KEYS_BASE, "topology", "depthTest", "draw"],
    reduction: [...NODE_KEYS_BASE],
    readback: [...NODE_KEYS_BASE],
};

const BINDING_KEYS = ["group", "binding", "resource", "varName", "historyLevel", "access"] as const;
const BINDING_REQUIRED = ["binding", "resource"] as const;

const TOPOLOGIES = ["point-list", "line-list", "triangle-list"] as const;
const ACCESS_MODES = ["read", "read_write"] as const;
const TEXTURE_DIMS = ["1d", "2d", "3d"] as const;
const DISPATCH_TYPES = ["direct", "indirect"] as const;
const DRAW_TYPES = ["direct", "indirect"] as const;

const UI_TYPES = ["range", "int", "checkbox", "select", "button"] as const;
const UI_SCALES = ["linear", "log"] as const;
const UI_ACTIONS = ["restart", "reset"] as const;
const UI_KEYS = [
    "name", "type", "label",
    "min", "max", "step", "scale",
    "options",
    "restart", "live", "format",
    "action",
] as const;

// Identifiers permitted inside metadata expressions but not actual metadata keys.
const META_EXPR_BUILTINS = new Set(["Math", "true", "false"]);

// ---------------------------------------------------------------------------
// Levenshtein + suggestion
// ---------------------------------------------------------------------------

function levenshtein(a: string, b: string): number {
    if (a === b) return 0;
    if (a.length === 0) return b.length;
    if (b.length === 0) return a.length;

    let prev = new Array<number>(b.length + 1);
    let curr = new Array<number>(b.length + 1);
    for (let j = 0; j <= b.length; j++) prev[j] = j;

    for (let i = 1; i <= a.length; i++) {
        curr[0] = i;
        for (let j = 1; j <= b.length; j++) {
            const cost = a.charCodeAt(i - 1) === b.charCodeAt(j - 1) ? 0 : 1;
            curr[j] = Math.min(
                curr[j - 1] + 1,        // insert
                prev[j] + 1,            // delete
                prev[j - 1] + cost,     // substitute
            );
        }
        [prev, curr] = [curr, prev];
    }
    return prev[b.length];
}

/** Return the closest candidate (case-aware) within an edit-distance threshold,
 * or `undefined` if nothing is close enough.
 */
export function suggest(query: string, candidates: Iterable<string>, max = 1): string | undefined {
    const list = [...candidates];
    if (list.length === 0) return undefined;

    // exact case-insensitive match wins immediately
    const lowered = query.toLowerCase();
    const ciHit = list.find(c => c.toLowerCase() === lowered);
    if (ciHit && ciHit !== query) return ciHit;

    // distance ≤ max(2, ⌊len/3⌋) — generous for short names, conservative for long ones
    const threshold = Math.max(2, Math.floor(query.length / 3));
    const scored = list
        .map(c => ({ c, d: levenshtein(query, c) }))
        .filter(s => s.d > 0 && s.d <= threshold)
        .sort((a, b) => a.d - b.d || a.c.localeCompare(b.c));
    if (scored.length === 0) return undefined;
    return scored.slice(0, max).map(s => s.c).join(", ");
}

// ---------------------------------------------------------------------------
// Issue collector
// ---------------------------------------------------------------------------

class Issues {
    readonly list: ValidationIssue[] = [];

    err(path: string, message: string, suggestion?: string) {
        this.list.push({ level: "error", path, message, suggestion });
    }

    warn(path: string, message: string, suggestion?: string) {
        this.list.push({ level: "warning", path, message, suggestion });
    }

    /** Convenience: report an unknown enum/key and look for a suggestion. */
    unknown(
        path: string,
        kind: string,
        offender: string,
        candidates: Iterable<string>,
        level: IssueLevel = "error",
    ) {
        const hint = suggest(offender, candidates);
        const msg = `Unknown ${kind} "${offender}".`;
        if (level === "error") this.err(path, msg, hint);
        else this.warn(path, msg, hint);
    }
}

// ---------------------------------------------------------------------------
// Type-guard helpers
// ---------------------------------------------------------------------------

function isPlainObject(x: unknown): x is Record<string, unknown> {
    return typeof x === "object" && x !== null && !Array.isArray(x);
}

function isStringOrNumber(x: unknown): x is string | number {
    return typeof x === "string" || typeof x === "number";
}

// ---------------------------------------------------------------------------
// Expression scanners
// ---------------------------------------------------------------------------

const DOLLAR_META_RE = /\$metadata\.([a-zA-Z_][a-zA-Z0-9_]*)/g;
const IDENT_RE = /[A-Za-z_][A-Za-z0-9_]*/g;

/** All `$metadata.<name>` references (de-duplicated). */
function extractDollarRefs(expr: string): string[] {
    const out = new Set<string>();
    for (const m of expr.matchAll(DOLLAR_META_RE)) out.add(m[1]);
    return [...out];
}

/** Bare identifiers in a metadata expression like `(segmentsX + 1) * Math.floor(x)`.
 *  Strips `Math.foo` member accesses so only top-level metadata names remain. */
function extractBareIdents(expr: string): string[] {
    const stripped = expr.replace(/Math\s*\.\s*[A-Za-z_][A-Za-z0-9_]*/g, " ");
    const out = new Set<string>();
    for (const m of stripped.matchAll(IDENT_RE)) {
        const id = m[0];
        if (!META_EXPR_BUILTINS.has(id)) out.add(id);
    }
    return [...out];
}

// ---------------------------------------------------------------------------
// Section validators
// ---------------------------------------------------------------------------

function validateRoot(schema: unknown, issues: Issues): schema is Record<string, unknown> {
    if (!isPlainObject(schema)) {
        issues.err("$", `Schema root must be an object, got ${describe(schema)}.`);
        return false;
    }

    for (const k of ROOT_REQUIRED) {
        if (!(k in schema)) {
            issues.err(`$.${k}`, `Missing required field "${k}".`);
        }
    }

    for (const k of Object.keys(schema)) {
        if (!(ROOT_KEYS as readonly string[]).includes(k)) {
            issues.unknown(`$.${k}`, "root key", k, ROOT_KEYS, "warning");
        }
    }

    if ("name" in schema && typeof schema.name !== "string") {
        issues.err("$.name", `"name" must be a string.`);
    }
    if ("version" in schema && typeof schema.version !== "string") {
        issues.err("$.version", `"version" must be a string.`);
    }
    return true;
}

function validateMetadata(
    schema: Record<string, unknown>,
    issues: Issues,
): Set<string> {
    const known = new Set<string>();
    const meta = schema.metadata;
    if (!isPlainObject(meta)) {
        if (meta !== undefined) issues.err("$.metadata", `"metadata" must be an object.`);
        return known;
    }

    // Walk in JSON insertion order so expressions can only reference earlier keys.
    const seen = new Set<string>();
    for (const k of Object.keys(meta)) {
        const v = meta[k];
        const path = `$.metadata.${k}`;

        const ok =
            typeof v === "number" ||
            typeof v === "boolean" ||
            typeof v === "string" ||
            (Array.isArray(v) && v.every(x => typeof x === "number"));
        if (!ok) {
            issues.err(
                path,
                `metadata.${k} must be number | boolean | string-expression | number[]; got ${describe(v)}.`,
            );
        }

        if (typeof v === "string") {
            for (const id of extractBareIdents(v)) {
                if (!seen.has(id)) {
                    issues.err(
                        path,
                        `Expression references "${id}" which is not defined earlier in metadata.`,
                        suggest(id, seen),
                    );
                }
            }
        }

        seen.add(k);
        known.add(k);
    }
    return known;
}

function validateResources(
    schema: Record<string, unknown>,
    metaKeys: Set<string>,
    issues: Issues,
): Set<string> {
    const known = new Set<string>();
    const resources = schema.resources;
    if (!isPlainObject(resources)) {
        if (resources !== undefined) issues.err("$.resources", `"resources" must be an object.`);
        return known;
    }

    for (const id of Object.keys(resources)) {
        known.add(id);
        const def = resources[id];
        const path = `$.resources.${id}`;
        if (!isPlainObject(def)) {
            issues.err(path, `Resource "${id}" must be an object.`);
            continue;
        }

        const rawType = def.type;
        let type: ResourceDef["type"] | undefined;
        if (typeof rawType !== "string") {
            issues.err(`${path}.type`, `Resource "${id}" is missing a "type" field.`);
        } else if (!(RESOURCE_TYPES as readonly string[]).includes(rawType)) {
            // Recover with the closest valid type so nested typos still surface.
            const hint = suggest(rawType, RESOURCE_TYPES);
            issues.err(`${path}.type`, `Unknown resource type "${rawType}".`, hint);
            if (hint && (RESOURCE_TYPES as readonly string[]).includes(hint)) {
                type = hint as ResourceDef["type"];
            }
        } else {
            type = rawType as ResourceDef["type"];
        }
        if (type === undefined) continue;

        // Unknown keys for this resource shape.
        const allowed = RESOURCE_KEYS_BY_TYPE[type] ?? [...RESOURCE_KEYS_COMMON];
        for (const k of Object.keys(def)) {
            if (!allowed.includes(k)) {
                issues.unknown(`${path}.${k}`, `key in ${type} resource`, k, allowed, "warning");
            }
        }

        validateResourceShape(id, type, def, metaKeys, issues, path);
    }

    return known;
}

function validateResourceShape(
    id: string,
    type: ResourceDef["type"],
    def: Record<string, unknown>,
    metaKeys: Set<string>,
    issues: Issues,
    path: string,
) {
    if ("bufferCount" in def) {
        const bc = def.bufferCount;
        if (typeof bc !== "number" || bc < 1 || !Number.isInteger(bc)) {
            issues.err(`${path}.bufferCount`, `"bufferCount" must be a positive integer (got ${describe(bc)}).`);
        }
    }
    if ("access" in def && type !== "uniform" && type !== "sampler") {
        if (!(ACCESS_MODES as readonly string[]).includes(def.access as string)) {
            issues.unknown(`${path}.access`, "access mode", String(def.access), ACCESS_MODES);
        }
    }

    switch (type) {
        case "storage":
        case "indirect": {
            if (def.format === undefined) {
                issues.err(`${path}.format`, `${type} resource "${id}" requires a "format".`);
            } else if (!WGSL_FORMATS.includes(def.format as WgslFormat)) {
                issues.unknown(`${path}.format`, "WGSL format", String(def.format), WGSL_FORMATS);
            }
            if (def.count === undefined) {
                issues.err(`${path}.count`, `${type} resource "${id}" requires a "count".`);
            } else {
                validateMetaRefExpr(def.count, `${path}.count`, metaKeys, issues);
            }
            break;
        }
        case "uniform": {
            const fields = def.fields;
            if (!isPlainObject(fields)) {
                issues.err(`${path}.fields`, `uniform resource "${id}" requires "fields" object.`);
                break;
            }
            for (const [fname, fmt] of Object.entries(fields)) {
                if (typeof fmt !== "string") {
                    issues.err(`${path}.fields.${fname}`, `Uniform field "${fname}" type must be a string.`);
                    continue;
                }
                if (!WGSL_FORMATS.includes(fmt as WgslFormat)) {
                    issues.unknown(
                        `${path}.fields.${fname}`,
                        "WGSL format",
                        fmt,
                        WGSL_FORMATS,
                    );
                }
            }
            break;
        }
        case "texture": {
            if ("dimension" in def && !(TEXTURE_DIMS as readonly string[]).includes(def.dimension as string)) {
                issues.unknown(`${path}.dimension`, "texture dimension", String(def.dimension), TEXTURE_DIMS);
            }
            if ("size" in def) {
                const sz = def.size;
                if (!Array.isArray(sz) || !sz.every(x => typeof x === "number" && x > 0)) {
                    issues.err(`${path}.size`, `texture "size" must be an array of positive numbers.`);
                }
            }
            break;
        }
        case "sampler":
            // No extra fields to check.
            break;
    }
}

function validateNodes(
    schema: Record<string, unknown>,
    resourceIds: Set<string>,
    metaKeys: Set<string>,
    issues: Issues,
) {
    const nodes = schema.nodes;
    if (!Array.isArray(nodes)) {
        if (nodes !== undefined) issues.err("$.nodes", `"nodes" must be an array.`);
        return;
    }

    const seenIds = new Set<string>();
    for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        const base = `$.nodes[${i}]`;
        if (!isPlainObject(node)) {
            issues.err(base, `Node #${i} must be an object.`);
            continue;
        }

        const id = node.id;
        const idPath = `${base}.id`;
        if (typeof id !== "string" || id.length === 0) {
            issues.err(idPath, `Node #${i} requires a non-empty string "id".`);
        } else if (seenIds.has(id)) {
            issues.err(idPath, `Duplicate node id "${id}".`);
        } else {
            seenIds.add(id);
        }

        const rawType = node.type;
        let type: NodeDef["type"] | undefined;
        if (typeof rawType !== "string") {
            issues.err(`${base}.type`, `Node "${id ?? `#${i}`}" requires a "type".`);
        } else if (!(NODE_TYPES as readonly string[]).includes(rawType)) {
            const hint = suggest(rawType, NODE_TYPES);
            issues.err(`${base}.type`, `Unknown node type "${rawType}".`, hint);
            if (hint && (NODE_TYPES as readonly string[]).includes(hint)) {
                type = hint as NodeDef["type"];
            }
        } else {
            type = rawType as NodeDef["type"];
        }

        if (typeof node.shader !== "string" || node.shader.length === 0) {
            issues.err(`${base}.shader`, `Node "${id ?? `#${i}`}" requires a "shader" name.`);
        }

        if (type === undefined) {
            // Still validate bindings — they don't depend on node type.
            validateBindings(node.bindings, resourceIds, issues, base);
            continue;
        }

        // Unknown keys per node-type.
        const allowed = NODE_KEYS_BY_TYPE[type] ?? [...NODE_KEYS_BASE];
        for (const k of Object.keys(node)) {
            if (!allowed.includes(k)) {
                issues.unknown(`${base}.${k}`, `key in ${type} node`, k, allowed, "warning");
            }
        }

        validateBindings(node.bindings, resourceIds, issues, base);

        if (type === "compute") validateDispatch(node.dispatch, resourceIds, metaKeys, issues, base);
        if (type === "render") validateRenderShape(node, resourceIds, metaKeys, issues, base);
    }
}

function validateBindings(
    bindings: unknown,
    resourceIds: Set<string>,
    issues: Issues,
    base: string,
) {
    if (!Array.isArray(bindings)) {
        issues.err(`${base}.bindings`, `"bindings" must be an array.`);
        return;
    }
    const slot = new Map<string, number>();
    for (let j = 0; j < bindings.length; j++) {
        const bind = bindings[j];
        const bp = `${base}.bindings[${j}]`;
        if (!isPlainObject(bind)) {
            issues.err(bp, `Binding #${j} must be an object.`);
            continue;
        }

        for (const k of BINDING_REQUIRED) {
            if (!(k in bind)) issues.err(`${bp}.${k}`, `Binding #${j} is missing "${k}".`);
        }
        for (const k of Object.keys(bind)) {
            if (!(BINDING_KEYS as readonly string[]).includes(k)) {
                issues.unknown(`${bp}.${k}`, "binding key", k, BINDING_KEYS, "warning");
            }
        }

        if ("group" in bind && (typeof bind.group !== "number" || !Number.isInteger(bind.group) || bind.group < 0)) {
            issues.err(`${bp}.group`, `"group" must be a non-negative integer.`);
        }
        if (typeof bind.binding !== "number" || !Number.isInteger(bind.binding) || bind.binding < 0) {
            issues.err(`${bp}.binding`, `"binding" must be a non-negative integer.`);
        }
        if ("access" in bind && !(ACCESS_MODES as readonly string[]).includes(bind.access as string)) {
            issues.unknown(`${bp}.access`, "access mode", String(bind.access), ACCESS_MODES);
        }

        const resName = bind.resource;
        if (typeof resName === "string") {
            if (!resourceIds.has(resName)) {
                issues.unknown(`${bp}.resource`, "resource", resName, resourceIds);
            }
        } else if ("resource" in bind) {
            issues.err(`${bp}.resource`, `"resource" must be a string.`);
        }

        if (typeof bind.binding === "number") {
            const grp = typeof bind.group === "number" ? bind.group : 0;
            const key = `${grp}/${bind.binding}`;
            if (slot.has(key)) {
                issues.err(`${bp}.binding`, `Duplicate slot (group ${grp}, binding ${bind.binding}); previous use at index ${slot.get(key)}.`);
            } else {
                slot.set(key, j);
            }
        }
    }
}

function validateDispatch(
    dispatch: unknown,
    resourceIds: Set<string>,
    metaKeys: Set<string>,
    issues: Issues,
    base: string,
) {
    const path = `${base}.dispatch`;
    if (!isPlainObject(dispatch)) {
        issues.err(path, `compute node requires a "dispatch" object.`);
        return;
    }
    const rawT = dispatch.type;
    let t: typeof DISPATCH_TYPES[number] | undefined;
    if (typeof rawT === "string" && (DISPATCH_TYPES as readonly string[]).includes(rawT)) {
        t = rawT as typeof DISPATCH_TYPES[number];
    } else {
        const hint = suggest(String(rawT), DISPATCH_TYPES);
        issues.err(`${path}.type`, `Unknown dispatch type "${String(rawT)}".`, hint);
        if (hint && (DISPATCH_TYPES as readonly string[]).includes(hint)) {
            t = hint as typeof DISPATCH_TYPES[number];
        }
    }
    if (t === undefined) return;
    if (t === "direct") {
        const wg = dispatch.workgroups;
        if (typeof wg === "string") {
            validateMetaRefExpr(wg, `${path}.workgroups`, metaKeys, issues);
        } else if (Array.isArray(wg)) {
            if (wg.length !== 3) {
                issues.err(`${path}.workgroups`, `"workgroups" array must have exactly 3 elements (got ${wg.length}).`);
            }
            for (let k = 0; k < wg.length; k++) {
                const x = wg[k];
                if (!isStringOrNumber(x)) {
                    issues.err(`${path}.workgroups[${k}]`, `Workgroup component must be number or string-expression.`);
                } else {
                    validateMetaRefExpr(x, `${path}.workgroups[${k}]`, metaKeys, issues);
                }
            }
        } else {
            issues.err(`${path}.workgroups`, `"workgroups" must be a string or 3-element array.`);
        }
    } else {
        // indirect
        const buf = dispatch.buffer;
        if (typeof buf !== "string") {
            issues.err(`${path}.buffer`, `Indirect dispatch requires a "buffer" string.`);
        } else if (!resourceIds.has(buf)) {
            issues.unknown(`${path}.buffer`, "resource", buf, resourceIds);
        }
    }
}

function validateRenderShape(
    node: Record<string, unknown>,
    resourceIds: Set<string>,
    metaKeys: Set<string>,
    issues: Issues,
    base: string,
) {
    if (!(TOPOLOGIES as readonly string[]).includes(node.topology as string)) {
        issues.unknown(`${base}.topology`, "topology", String(node.topology), TOPOLOGIES);
    }
    if ("depthTest" in node && typeof node.depthTest !== "boolean") {
        issues.err(`${base}.depthTest`, `"depthTest" must be a boolean.`);
    }

    const draw = node.draw;
    const path = `${base}.draw`;
    if (!isPlainObject(draw)) {
        issues.err(path, `render node requires a "draw" object.`);
        return;
    }
    const rawT = draw.type;
    let t: typeof DRAW_TYPES[number] | undefined;
    if (typeof rawT === "string" && (DRAW_TYPES as readonly string[]).includes(rawT)) {
        t = rawT as typeof DRAW_TYPES[number];
    } else {
        const hint = suggest(String(rawT), DRAW_TYPES);
        issues.err(`${path}.type`, `Unknown draw type "${String(rawT)}".`, hint);
        if (hint && (DRAW_TYPES as readonly string[]).includes(hint)) {
            t = hint as typeof DRAW_TYPES[number];
        }
    }
    if (t === undefined) return;
    if (t === "direct") {
        if (!isStringOrNumber(draw.vertexCount)) {
            issues.err(`${path}.vertexCount`, `"vertexCount" must be a number or string-expression.`);
        } else {
            validateMetaRefExpr(draw.vertexCount, `${path}.vertexCount`, metaKeys, issues);
        }
        if ("instanceCount" in draw) {
            if (!isStringOrNumber(draw.instanceCount)) {
                issues.err(`${path}.instanceCount`, `"instanceCount" must be a number or string-expression.`);
            } else {
                validateMetaRefExpr(draw.instanceCount, `${path}.instanceCount`, metaKeys, issues);
            }
        }
    } else {
        const buf = draw.buffer;
        if (typeof buf !== "string") {
            issues.err(`${path}.buffer`, `Indirect draw requires a "buffer" string.`);
        } else if (!resourceIds.has(buf)) {
            issues.unknown(`${path}.buffer`, "resource", buf, resourceIds);
        }
    }
}

function validateUis(
    schema: Record<string, unknown>,
    metaKeys: Set<string>,
    issues: Issues,
) {
    const uis = schema.uis;
    if (uis === undefined) return;
    if (!Array.isArray(uis)) {
        issues.err("$.uis", `"uis" must be an array.`);
        return;
    }
    const seenNames = new Set<string>();
    for (let i = 0; i < uis.length; i++) {
        const u = uis[i] as Partial<UiDef> & Record<string, unknown>;
        const base = `$.uis[${i}]`;
        if (!isPlainObject(u)) {
            issues.err(base, `UI #${i} must be an object.`);
            continue;
        }

        for (const k of Object.keys(u)) {
            if (!(UI_KEYS as readonly string[]).includes(k)) {
                issues.unknown(`${base}.${k}`, "ui key", k, UI_KEYS, "warning");
            }
        }

        const name = u.name;
        if (typeof name !== "string" || name.length === 0) {
            issues.err(`${base}.name`, `UI #${i} requires a "name" string.`);
        } else if (seenNames.has(name)) {
            issues.warn(`${base}.name`, `Duplicate UI control name "${name}".`);
        } else {
            seenNames.add(name);
        }

        const type = u.type;
        if (typeof type !== "string" || !(UI_TYPES as readonly string[]).includes(type)) {
            issues.unknown(`${base}.type`, "ui type", String(type), UI_TYPES);
            continue;
        }

        // Synthetic button names start with "_" and don't have to be metadata keys.
        const isButton = type === "button";
        if (typeof name === "string" && !isButton && !name.startsWith("_") && !metaKeys.has(name)) {
            issues.unknown(`${base}.name`, "metadata key", name, metaKeys);
        }

        if (type === "range" || type === "int") {
            if ("min" in u && typeof u.min !== "number") issues.err(`${base}.min`, `"min" must be a number.`);
            if ("max" in u && typeof u.max !== "number") issues.err(`${base}.max`, `"max" must be a number.`);
            if ("step" in u && typeof u.step !== "number") issues.err(`${base}.step`, `"step" must be a number.`);
            if (typeof u.min === "number" && typeof u.max === "number" && u.min > u.max) {
                issues.err(`${base}`, `"min" (${u.min}) is greater than "max" (${u.max}).`);
            }
            if ("scale" in u && !(UI_SCALES as readonly string[]).includes(u.scale as string)) {
                issues.unknown(`${base}.scale`, "ui scale", String(u.scale), UI_SCALES);
            }
        }
        if (type === "select") {
            if (!Array.isArray(u.options) || u.options.length === 0) {
                issues.err(`${base}.options`, `"select" UI requires a non-empty "options" array.`);
            }
        }
        if (isButton) {
            if (typeof u.action !== "string" || !(UI_ACTIONS as readonly string[]).includes(u.action)) {
                issues.unknown(`${base}.action`, "ui action", String(u.action), UI_ACTIONS);
            }
        }
        if ("restart" in u && typeof u.restart !== "boolean") {
            issues.err(`${base}.restart`, `"restart" must be a boolean.`);
        }
        if ("live" in u && typeof u.live !== "boolean") {
            issues.err(`${base}.live`, `"live" must be a boolean.`);
        }
    }
}

// ---------------------------------------------------------------------------
// $metadata.X expression scanner shared by count / workgroups / vertexCount …
// ---------------------------------------------------------------------------

function validateMetaRefExpr(
    expr: unknown,
    path: string,
    metaKeys: Set<string>,
    issues: Issues,
) {
    if (typeof expr === "number") return;
    if (typeof expr !== "string") {
        issues.err(path, `Expected number or string-expression; got ${describe(expr)}.`);
        return;
    }
    for (const ref of extractDollarRefs(expr)) {
        if (!metaKeys.has(ref)) {
            issues.unknown(path, "metadata reference", ref, metaKeys);
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

export function validateSchema(schema: unknown): ValidationIssue[] {
    const issues = new Issues();
    if (!validateRoot(schema, issues)) return issues.list;

    const metaKeys = validateMetadata(schema, issues);
    const resourceIds = validateResources(schema, metaKeys, issues);
    validateNodes(schema, resourceIds, metaKeys, issues);
    validateUis(schema, metaKeys, issues);
    return issues.list;
}

/** Throws `SchemaValidationError` if any error-level issues are found.
 *  Warnings are returned silently (they're surfaced by `formatIssues`).
 */
export function assertValidSchema(schema: unknown): SimulationSchema {
    const issues = validateSchema(schema);
    const errors = issues.filter(i => i.level === "error");
    const name = isPlainObject(schema) && typeof schema.name === "string" ? schema.name : undefined;
    if (errors.length > 0) throw new SchemaValidationError(issues, name);
    return schema as SimulationSchema;
}

export function formatIssues(issues: ValidationIssue[], schemaName?: string): string {
    if (issues.length === 0) return "Schema is valid.";
    const header = `Schema "${schemaName ?? "<anonymous>"}" has ${issues.length} issue${issues.length === 1 ? "" : "s"}:`;
    const body = issues
        .map(i => {
            const tag = i.level === "error" ? "ERROR  " : "WARNING";
            const tail = i.suggestion ? `  — did you mean "${i.suggestion}"?` : "";
            return `  [${tag}] ${i.path}: ${i.message}${tail}`;
        })
        .join("\n");
    return `${header}\n${body}`;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function describe(x: unknown): string {
    if (x === null) return "null";
    if (Array.isArray(x)) return `array(length=${x.length})`;
    return typeof x;
}
