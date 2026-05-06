# NEXT TASKS (Handoff Checklist)

This file is a short execution checklist for continuing work in a new chat.

## 0) Start-of-chat prompt

Use this at the beginning of a new chat:

> Please read `framework_en.md` and `NEXT_TASKS.md` first.  
> Continue from the TypeScript authoring pipeline (`ts/builder/*`, `build/cli.ts`, `build/sims/*`).  
> Keep runtime artifacts as JSON+WGSL+DSL only.

---

## 1) Immediate verification

- [ ] Run:
  - `npx tsx webgpu/build/cli.ts webgpu/build/sims/fem_cg.ts --check`
  - `npx tsx webgpu/build/cli.ts webgpu/build/sims/ball.ts --check`
- [ ] Run:
  - `npx tsc -p webgpu/tsconfig.json --noEmit`

Expected status:
- `fem_cg`: JSON semantic match + DSL textual match.
- `ball`: JSON semantic match + DSL token-stream match (formatting may differ).

---

## 2) Port remaining simulations to TypeScript authoring

Create these files under `build/sims/`:

- [ ] `collision.ts`
- [ ] `life.ts`
- [ ] `surface.ts`
- [ ] `vector_field.ts`
- [ ] `fem_cg2.ts`

For each port:
- [ ] Keep `schema.name` equal to the current JSON display name.
- [ ] Run `--check` and ensure semantic equivalence.
- [ ] Only adjust whitespace formatting if needed; avoid changing behavior.

---

## 3) Builder API hardening (recommended)

- [ ] Add resource-key typing for bindings (`resource` should only accept declared resource IDs).
- [ ] Add optional helper for metadata references in expression strings.
- [ ] Add unit tests for serializer output and DSL recorder edge cases.

Nice-to-have:
- [ ] Add a strict mode that fails if DSL output is not textually identical.

---

## 4) CI integration

- [ ] Add a CI step to run:
  - `npx tsc -p webgpu/tsconfig.json --noEmit`
  - `npx tsx webgpu/build/cli.ts webgpu/build/sims/<sim>.ts --check` for all sims
- [ ] Fail CI on schema validation errors/warnings policy as desired.

---

## 5) Security model (do not regress)

- [ ] Keep browser runtime input to JSON + WGSL + DSL only.
- [ ] Do not execute user-uploaded TypeScript/JavaScript in the app origin.
- [ ] Use TypeScript only as offline authoring/build-time source.

---

## 6) Optional next milestone

- [ ] Implement a lightweight "doctor mode" for debugging:
  - NaN/Inf checks on selected storage buffers
  - frame-time watchdog
  - compact diagnostics report for AI-driven fix loops

