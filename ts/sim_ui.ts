import { GraphManager, SimulationSchema, UiDef } from './control';

// ============================================================================
// UI builder for simulation schemas.
//
// Reads `schema.uis` (declared in the JSON) and builds matching DOM controls
// next to the canvas. Each control is bound to a metadata key by name:
//
//   - "range" / "int"  -> <input type="range">  -> engine.updateVariables({name})
//   - "checkbox"       -> <input type="checkbox">
//   - "select"         -> <select>
//   - "button"         -> <button>  with action "restart" or "reset"
//
// `restart: true` causes engine.restartScript() to fire after each commit.
// `live` controls whether we listen to "input" (per-pixel drag) or "change"
// (release only); default is `!restart`.
// ============================================================================

const STYLE_ID = 'sim-ui-style';

const DEFAULT_STYLE = `
.sim-ui-floating {
    position: fixed;
    top: 8px;
    right: 8px;
    background: rgba(20, 20, 20, 0.85);
    color: #ddd;
    padding: 12px;
    border-radius: 8px;
    max-width: 380px;
    z-index: 1000;
    user-select: none;
}
.sim-ui-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
}
.sim-ui-row label {
    min-width: 160px;
    font: 13px sans-serif;
}
.sim-ui-row input[type="range"] {
    flex: 1;
}
.sim-ui-row .sim-ui-value {
    min-width: 70px;
    text-align: right;
    font: 12px monospace;
}
.sim-ui-row button {
    padding: 4px 12px;
    font: 13px sans-serif;
    cursor: pointer;
}
`;

function injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement('style');
    s.id = STYLE_ID;
    s.textContent = DEFAULT_STYLE;
    document.head.appendChild(s);
}

function ensureContainer(): HTMLElement {
    const existing = document.getElementById('sim-ui') as HTMLElement | null;
    if (existing) return existing;
    const c = document.createElement('div');
    c.id = 'sim-ui';
    c.classList.add('sim-ui-floating');
    document.body.appendChild(c);
    return c;
}

interface ResetHandle {
    name: string;
    reset: () => void;
}

/** Public entry point. Builds (or re-builds) the UI from `schema.uis`. */
export function buildUI(
    engine: GraphManager,
    schema: SimulationSchema,
    container?: HTMLElement,
): void {
    injectStyle();

    const c = container ?? ensureContainer();
    c.innerHTML = '';

    const handles: ResetHandle[] = [];

    for (const u of schema.uis ?? []) {
        const row = buildRow(engine, schema, u, handles);
        if (row) c.appendChild(row);
    }
}

function buildRow(
    engine: GraphManager,
    schema: SimulationSchema,
    u: UiDef,
    handles: ResetHandle[],
): HTMLElement | null {
    if (u.type !== 'button' && schema.metadata[u.name] === undefined) {
        console.warn(`[sim-ui] metadata "${u.name}" is undefined; skipping control`);
        return null;
    }

    const row = document.createElement('div');
    row.className = 'sim-ui-row';

    if (u.type !== 'button') {
        const lbl = document.createElement('label');
        lbl.textContent = u.label ?? u.name;
        row.appendChild(lbl);
    }

    switch (u.type) {
        case 'range':
        case 'int':
            buildRange(row, engine, schema, u, handles);
            break;
        case 'checkbox':
            buildCheckbox(row, engine, schema, u, handles);
            break;
        case 'select':
            buildSelect(row, engine, schema, u, handles);
            break;
        case 'button':
            buildButton(row, engine, u, handles);
            break;
        default:
            console.warn(`[sim-ui] unknown control type "${(u as any).type}"`);
            return null;
    }

    return row;
}

// ---------------------------------------------------------------------------
// Range / int slider
// ---------------------------------------------------------------------------

function buildRange(
    row: HTMLElement,
    engine: GraphManager,
    schema: SimulationSchema,
    u: UiDef,
    handles: ResetHandle[],
): void {
    const slider = document.createElement('input');
    slider.type = 'range';

    const isLog = u.scale === 'log';
    const min = u.min ?? 0;
    const max = u.max ?? 1;
    const step = u.step ?? (u.type === 'int' ? 1 : 0.01);

    if (isLog && (min <= 0 || max <= 0)) {
        console.warn(`[sim-ui] "${u.name}": log scale requires min > 0 and max > 0`);
    }

    slider.min = String(isLog ? Math.log10(min) : min);
    slider.max = String(isLog ? Math.log10(max) : max);
    slider.step = String(step);

    const initial = Number(schema.metadata[u.name] ?? min);
    const setSliderFromValue = (v: number) => {
        slider.value = String(isLog ? Math.log10(Math.max(v, Number.MIN_VALUE)) : v);
    };
    setSliderFromValue(initial);

    const display = document.createElement('span');
    display.className = 'sim-ui-value';

    const decimals = decimalsFromStep(step);
    const fmt = (n: number): string => {
        if (u.format) return printfLike(u.format, n);
        if (u.type === 'int') return Math.round(n).toString();
        return n.toFixed(decimals);
    };

    const decode = (s: string): number => {
        const x = Number(s);
        if (isLog) return Math.pow(10, x);
        return u.type === 'int' ? Math.round(x) : x;
    };

    display.textContent = fmt(initial);

    // Always show live numeric feedback while dragging, even if we don't commit yet.
    slider.addEventListener('input', () => {
        display.textContent = fmt(decode(slider.value));
    });

    // Decide which event commits to the engine.
    const live = u.live ?? !u.restart;
    const commitEvent: 'input' | 'change' = live ? 'input' : 'change';

    slider.addEventListener(commitEvent, () => {
        const v = decode(slider.value);
        engine.updateVariables({ [u.name]: v });
        if (u.restart) engine.restartScript();
    });

    row.appendChild(slider);
    row.appendChild(display);

    handles.push({
        name: u.name,
        reset: () => {
            setSliderFromValue(initial);
            display.textContent = fmt(initial);
            engine.updateVariables({ [u.name]: initial });
        },
    });
}

// ---------------------------------------------------------------------------
// Checkbox
// ---------------------------------------------------------------------------

function buildCheckbox(
    row: HTMLElement,
    engine: GraphManager,
    schema: SimulationSchema,
    u: UiDef,
    handles: ResetHandle[],
): void {
    const cb = document.createElement('input');
    cb.type = 'checkbox';

    const initial = Boolean(schema.metadata[u.name]);
    cb.checked = initial;

    cb.addEventListener('change', () => {
        // Encode booleans as 0/1 since uniform fields are always numeric.
        engine.updateVariables({ [u.name]: cb.checked ? 1 : 0 });
        if (u.restart) engine.restartScript();
    });

    row.appendChild(cb);

    handles.push({
        name: u.name,
        reset: () => {
            cb.checked = initial;
            engine.updateVariables({ [u.name]: initial ? 1 : 0 });
        },
    });
}

// ---------------------------------------------------------------------------
// Select
// ---------------------------------------------------------------------------

function buildSelect(
    row: HTMLElement,
    engine: GraphManager,
    schema: SimulationSchema,
    u: UiDef,
    handles: ResetHandle[],
): void {
    const sel = document.createElement('select');

    for (const opt of u.options ?? []) {
        const o = document.createElement('option');
        o.value = String(opt.value);
        o.textContent = opt.label;
        sel.appendChild(o);
    }

    const initial = schema.metadata[u.name];
    sel.value = String(initial);

    const commit = () => {
        const raw = sel.value;
        const v = isFinite(Number(raw)) ? Number(raw) : raw;
        engine.updateVariables({ [u.name]: v as number });
        if (u.restart) engine.restartScript();
    };
    sel.addEventListener('change', commit);

    row.appendChild(sel);

    handles.push({
        name: u.name,
        reset: () => {
            sel.value = String(initial);
            commit();
        },
    });
}

// ---------------------------------------------------------------------------
// Button
// ---------------------------------------------------------------------------

function buildButton(
    row: HTMLElement,
    engine: GraphManager,
    u: UiDef,
    handles: ResetHandle[],
): void {
    const btn = document.createElement('button');
    btn.textContent = u.label ?? u.name;

    btn.addEventListener('click', () => {
        const action = u.action ?? 'restart';
        if (action === 'reset') {
            for (const h of handles) h.reset();
            engine.restartScript();
        } else {
            engine.restartScript();
        }
    });

    row.appendChild(btn);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function decimalsFromStep(step: number): number {
    if (step >= 1) return 0;
    return Math.max(0, -Math.floor(Math.log10(step)));
}

/** Tiny printf-like formatter: supports %.Nf, %.Ne, %d. */
function printfLike(fmt: string, n: number): string {
    const m = fmt.match(/%(?:\.(\d+))?([fed])/);
    if (!m) return n.toString();
    const decimals = m[1] !== undefined ? parseInt(m[1], 10) : 6;
    switch (m[2]) {
        case 'f': return n.toFixed(decimals);
        case 'e': return n.toExponential(decimals);
        case 'd': return Math.round(n).toString();
        default:  return n.toString();
    }
}
