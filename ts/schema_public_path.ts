// ============================================================================
// Where each graph-schema bundle lives under public/ (browser + build/cli).
// Physics sims: public/engines/physics/<id>/
// Others:       public/wgsl/<id>/
// ============================================================================

export const PHYSICS_ENGINE_SCHEMA_IDS = [
    "ball",
    "cfd_simple",
    "collision",
    "em_fem",
    "fem_cg",
    "fem_cg2",
    "life",
    "md",
    "surface",
    "thermal_fem",
    "vector_field",
] as const;

const PHYSICS_SET = new Set<string>(PHYSICS_ENGINE_SCHEMA_IDS);

export function isPhysicsEngineSchema(schemaName: string): boolean {
    return PHYSICS_SET.has(schemaName);
}

/** Fetch URL base: ./engines/physics/<id> or ./wgsl/<id> (relative to page origin). */
export function simulationAssetBaseUrl(schemaName: string): string {
    return isPhysicsEngineSchema(schemaName)
        ? `./engines/physics/${schemaName}`
        : `./wgsl/${schemaName}`;
}
