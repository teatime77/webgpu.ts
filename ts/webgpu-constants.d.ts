// webgpu-constants.d.ts
declare const GPUShaderStage: {
    readonly VERTEX: 1;
    readonly FRAGMENT: 2;
    readonly COMPUTE: 4;
};

// ※もし今後以下の定数でもエラーが出た場合は、これらも追記してください
declare const GPUBufferUsage: {
    readonly MAP_READ: 1;
    readonly MAP_WRITE: 2;
    readonly COPY_SRC: 4;
    readonly COPY_DST: 8;
    readonly INDEX: 16;
    readonly VERTEX: 32;
    readonly UNIFORM: 64;
    readonly STORAGE: 128;
    readonly INDIRECT: 256;
    readonly QUERY_RESOLVE: 512;
};

declare const GPUTextureUsage: {
    readonly COPY_SRC: 1;
    readonly COPY_DST: 2;
    readonly TEXTURE_BINDING: 4;
    readonly STORAGE_BINDING: 8;
    readonly RENDER_ATTACHMENT: 16;
};

declare const GPUMapMode: {
    readonly READ: 1;
    readonly WRITE: 2;
};