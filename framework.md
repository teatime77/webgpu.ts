# WebGPU Simulation Framework Architecture

現在のプロジェクトは、WebGPUを用いた物理シミュレーションや描画を効率的に行うための独自のデータドリブン・フレームワーク（JSON + WGSL + DSL）で構成されています。
新しいチャットで作業を再開する際は、このドキュメントをAIに読ませることで、アーキテクチャの前提知識を即座に共有できます。

## 1. 全体構成 (3つのファイルの役割)
シミュレーション（例：`surface`, `vector_field`, `collision` など）は、基本的に同名の3つのファイルで構成されます。

### ① JSON スキーマ (`*.json`)
シミュレーションで使用するGPUリソースと、実行するノード（Compute/Render）を定義します。
- **`metadata`**: CPU側から毎フレーム更新される変数（時間、画面サイズ、パーティクル数など）。
- **`resources`**: `uniform`, `storage`, `texture` などのGPUバッファ定義。
- **`nodes`**: 実行するシェーダのパス（Compute/Render）、ディスパッチサイズ、頂点数、インスタンス数、およびバインドするリソースの定義。
  - 数式（例: `"$metadata.gridWidth * $metadata.gridHeight"`) が使用可能。

### ② WGSL シェーダ (`*.wgsl`)
実際のGPUプログラム。1つのファイル内に複数のシェーダを記述し、`// @shader: [名前]` のコメントで区切ります。
- **ヘッダーの自動生成**: `control.ts` の `WgslHeaderGenerator` が、JSONの定義をもとに `@group(0) @binding(0)` などのバインディング宣言や構造体を**自動的に生成して先頭に付与**します。
- **命名規則（重要）**: `type: "uniform"` のリソースは、名前の衝突を防ぐために自動生成される構造体名に **`Struct`** という接尾辞が付きます。（例：リソース名が `Params` の場合、構造体は `ParamsStruct`、変数は `var<uniform> Params: ParamsStruct;` となる）。

### ③ DSL 制御スクリプト (`*.js`)
GPUノードの実行順序やループを制御する独自のスクリプトです。
- `while(true)` ループの中で、JSONで定義したノード名（例: `compute_surface();`）を呼び出します。
- `swapPingPong(A, B);` でダブルバッファリングのスワップを行います。
- `yield;` で1フレームの処理を終了し、描画を画面に反映させます。

---

## 2. TypeScript側のエンジン (`ts/`)

### `control.ts` (`GraphManager`)
フレームワークのコアエンジンです。
- JSONをパースし、必要なGPUBuffer/GPUTextureを動的にアロケーションします。
- DSLスクリプトを独自のパーサーでASTに変換し、ジェネレーター（Generator）として毎フレーム `step()` メソッドで実行します。
- `updateVariables()` により、CPU側の変数を直接Uniformバッファにバイナリ転送します（超高速）。

### `main.ts`
初期化とメインループを担当します。
- WebGPUデバイスの取得、カメラの初期化。
- **ベースメッシュの生成**: 球や矢印などのベースとなる頂点データは、スキーマのロード前に `primitive.ts` の関数（`makeGeodesicPolyhedron`, `makeArrowMesh` 等）で生成し、メタデータに頂点数をセットした上で Storage バッファに書き込みます。
- 毎フレーム `engine.updateVariables()` を呼び出し、`engine.step()` を実行します。

### `primitive.ts`
CPU側でベースとなる3Dメッシュ（頂点配列）を生成するユーティリティ群です。

---

## 3. ベストプラクティスと設計思想 (重要)

1. **CPU-GPU間の通信を最小化する**
   - 毎フレームCPU側で頂点配列を再生成してGPUに送ることは**しません**。
   - 頂点の計算や変形はすべてコンピュートシェーダまたは頂点シェーダ内で行います。

2. **Vertex Pulling と Instancing の活用**
   - 従来の Vertex Buffer（`@location(0)`）は使わず、すべての頂点データや計算結果は **Storage バッファ (`read` または `read_write`)** としてバインドします。
   - 頂点シェーダ内では、`@builtin(vertex_index)` と `@builtin(instance_index)` を使って、Storage バッファから自分自身の座標や法線、ベクトルデータを読み出します（Vertex Pulling）。

3. **1つのメッシュで多様な表現を行う（例：ベクトル場の矢印）**
   - 矢印などの複雑な形状も、円柱と円錐に分けるのではなく「1つのベースメッシュ」として生成します。
   - 頂点シェーダ内で「Y座標が1.0以下なら軸として伸ばす、1.0以上なら傘として平行移動する」といった条件分岐を行うことで、1回のドローコール（Instancing）で可変長のオブジェクトを大量に描画します。