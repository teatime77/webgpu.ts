# WebGPU Simulation Framework Architecture

The current project consists of a custom data-driven framework (JSON + WGSL + DSL) for efficiently performing physical simulations and rendering using WebGPU.
When resuming work in a new chat, having the AI read this document allows for immediate sharing of the architectural prerequisites.

## 1. Overall Structure (Roles of the 3 Files)
A simulation (e.g., `surface`, `vector_field`, `collision`, etc.) generally consists of three files with the same name.

### ① JSON Schema (`*.json`)
Defines the GPU resources used in the simulation and the nodes (Compute/Render) to be executed.
- **`metadata`**: Variables updated every frame from the CPU side (time, screen size, particle count, etc.).
- **`resources`**: GPU buffer definitions such as `uniform`, `storage`, `texture`, etc.
- **`nodes`**: Definitions of the shader passes to execute (Compute/Render), dispatch sizes, vertex counts, instance counts, and the resources to bind.
  - Mathematical expressions (e.g., `"$metadata.gridWidth * $metadata.gridHeight"`) can be used.

### ② WGSL Shader (`*.wgsl`)
The actual GPU program. Multiple shaders are written in a single file, separated by `// @shader: [name]` comments.
- **Automatic Header Generation**: `WgslHeaderGenerator` in `control.ts` **automatically generates and prepends** binding declarations like `@group(0) @binding(0)` and structures based on the JSON definitions.
- **Naming Convention (Important)**: To prevent name collisions, resources with `type: "uniform"` have the suffix **`Struct`** added to their automatically generated structure names. (e.g., If the resource name is `Params`, the structure will be `ParamsStruct`, and the variable will be `var<uniform> Params: ParamsStruct;`).

### ③ DSL Control Script (`*.js`)
A custom script that controls the execution order and loops of the GPU nodes.
- Inside a `while(true)` loop, it calls the node names defined in the JSON (e.g., `compute_surface();`).
- Performs double buffering swaps with `swapPingPong(A, B);`.
- Ends the processing for one frame and reflects the rendering to the screen with `yield;`.

---

## 2. TypeScript Engine (`ts/`)

### `control.ts` (`GraphManager`)
The core engine of the framework.
- Parses the JSON and dynamically allocates the necessary GPUBuffer/GPUTexture.
- Converts the DSL script into an AST using a custom parser, and executes it every frame as a Generator via the `step()` method.
- Directly transfers CPU-side variables to Uniform buffers in binary form via `updateVariables()` (ultra-fast).
- **DSL generator contract (`step`)**: When the generator yields a `swapPingPong(...)` call, `GraphManager` performs the resource ring swap, then immediately advances the generator again and **requires the next yield to be `yield`** (a `YieldStatement`). Authoring patterns that match this include placing `yield;` right after each `swapPingPong` (including an initial swap after `init_*` when using ping-pong buffers, similar to `ball`).

### `main.ts`
Handles initialization and the main loop.
- Acquires the WebGPU device and initializes the camera.
- **Base Mesh Generation**: Base vertex data for spheres, arrows, etc., is generated using functions in `primitive.ts` (like `makeGeodesicPolyhedron`, `makeArrowMesh`, etc.) before loading the schema. The vertex count is set in the metadata, and the data is written to Storage buffers.
- Calls `engine.updateVariables()` and executes `engine.step()` every frame.
- **Canvas capture (debug / validation)**: A small overlay on `#app-container` offers **Capture** (single PNG download of `#world-webgpu`) and **Burst xN** (N downloads at a configurable interval in ms, e.g. 100 ms for 0.1 s steps). This is browser-only UI; it does not change runtime simulation artifacts.

### `primitive.ts`
A collection of utilities for generating base 3D meshes (vertex arrays) on the CPU side.

---

## 3. Best Practices and Design Philosophy (Important)

1. **Minimize CPU-GPU Communication**
   - We **do not** regenerate vertex arrays on the CPU side and send them to the GPU every frame.
   - All vertex calculations and deformations are performed within compute shaders or vertex shaders.

2. **Utilization of Vertex Pulling and Instancing**
   - Traditional Vertex Buffers (`@location(0)`) are not used. All vertex data and calculation results are bound as **Storage buffers (`read` or `read_write`)**.
   - Within the vertex shader, `@builtin(vertex_index)` and `@builtin(instance_index)` are used to read the vertex's own coordinates, normals, and vector data from the Storage buffer (Vertex Pulling).

3. **Diverse Representations with a Single Mesh (e.g., Vector Field Arrows)**
   - Complex shapes like arrows are generated as a "single base mesh" rather than being split into cylinders and cones.
   - By using conditional branching in the vertex shader (e.g., "if the Y coordinate is 1.0 or less, stretch it as the shaft; if 1.0 or more, translate it as the head"), a large number of variable-length objects are drawn with a single draw call (Instancing).

---

## 4. Current Status (May 2026)

Recent updates have been completed:

1. **Schema validator with "did you mean" suggestions**
   - Added in `ts/schema_validator.ts`.
   - Integrated into `GraphManager.loadSchema()` in `ts/control.ts`.
   - Validates root fields, metadata expression references, resources, nodes, bindings, dispatch/draw definitions, and `uis`.
   - Produces typo suggestions (e.g., unknown key/type/resource names).

2. **WebGPU type definitions**
   - Installed `@webgpu/types` and enabled it in the repo `tsconfig.json` (`compilerOptions.types`).

3. **TypeScript authoring pipeline prototype**
   - Added builder API:
     - `ts/builder/index.ts`
     - `ts/builder/dsl.ts`
     - `ts/builder/serialize.ts`
   - Added CLI:
     - `build/cli.ts` (Node-only; resolves canonical artifacts under `public/wgsl/<sim>/` from the simulation source basename)
   - Authored simulation sources under `build/sims/` include (among others): `fem_cg`, `ball`, `collision`, `life`, `surface`, `vector_field`, `fem_cg2`, `thermal_fem`, `em_fem`, `cfd_simple`.
   - The CLI can emit runtime artifacts and compare against checked-in files under `public/wgsl/<name>/` (from repo root), for example:
     - `npx tsx build/cli.ts build/sims/fem_cg.ts --check`
     - `npx tsx build/cli.ts build/sims/ball.ts --check`

### Recommended production model

- **Authoring**: TypeScript (for AI productivity and type safety).
- **Published/runtime content**: JSON + WGSL + DSL only.
- **Browser runtime** should continue loading only JSON/WGSL/DSL artifacts (no user TypeScript execution).

---

## 5. Suggested Prompt for a New Chat

When starting a new chat, use this:

> Please read `framework_en.md` first.  
> We use TypeScript as authoring source and emit JSON+WGSL+DSL as runtime artifacts.  
> Continue from the current builder prototype (`ts/builder/*`, `build/cli.ts`, `build/sims/*`; canonical runtime under `public/wgsl/`) and help me [your next task].
