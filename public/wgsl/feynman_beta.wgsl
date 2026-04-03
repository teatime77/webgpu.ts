// ============================================================================
// Beta Decay Phase Space & Matrix Element Generator
// ============================================================================

const NUM_BINS: u32 = 100u;

// 運動学パラメータ (エネルギーの単位は keV)
const M_T: f32 = 2808921.0; // トリチウムの質量 (約 2.8 GeV)
const M_He: f32 = 2808391.0; // ヘリウム3の質量
const m_e: f32 = 511.0; // 電子の質量
const Q: f32 = 18.6; // ベータ崩壊のQ値 (運動エネルギーの最大値: 18.6 keV)

// --- 1. 4元ベクトルと行列要素 ---
alias FourVector = vec4<f32>;

fn dot4(a: FourVector, b: FourVector) -> f32 {
    return a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w;
}

// --- 2. 乱数生成器 (PCG) ---
fn random_u32(state: ptr<function, u32>) -> u32 {
    let old_state = *state;
    *state = old_state * 747796405u + 2891336453u;
    var word = ((old_state >> ((old_state >> 28u) + 4u)) ^ old_state) * 277803737u;
    word = (word >> 22u) ^ word;
    return word;
}

fn random_f32(state: ptr<function, u32>) -> f32 {
    return f32(random_u32(state)) / 4294967295.0;
}

// --- 3. バッファの定義 ---
// 崩壊した電子のエネルギー分布を記録するヒストグラム
// WebGPUのアトミック演算を使って、全スレッドから同時に安全に足し込みます
@group(0) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> rng_state: array<u32>;

// --- 4. コンピュートカーネル ---
@compute @workgroup_size(64, 1, 1)
fn generate_events(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    var seed = rng_state[idx];

    // GPUの1スレッドが、100回の仮想的な崩壊イベントをテストする
    for (var i = 0u; i < 100u; i++) {
        // [ステップ1: 位相空間 (Phase Space) のランダムサンプリング]
        // 電子の運動エネルギー K_e を 0 〜 Q の間でランダムに生成
        let K_e = random_f32(&seed) * Q;
        let E_e = K_e + m_e; // 電子の全エネルギー
        let p_e_mag = sqrt(E_e * E_e - m_e * m_e); // 電子の運動量の大きさ

        // 反跳エネルギーを無視した近似でのニュートリノのエネルギー
        let E_nu = Q - K_e; 

        // 角度をランダムに生成 (ここでは簡易的にニュートリノと電子のなす角をサンプリング)
        let cos_theta = 2.0 * random_f32(&seed) - 1.0;

        // [ステップ2: 4元運動量の構築]
        // トリチウム (静止系)
        let p_T = FourVector(M_T, 0.0, 0.0, 0.0);
        
        // 電子 (z軸方向に飛んだと仮定)
        let p_e = FourVector(E_e, 0.0, 0.0, p_e_mag);
        
        // 反電子ニュートリノ (角度 theta 方向へ飛ぶ)
        let p_nu = FourVector(E_nu, E_nu * sqrt(1.0 - cos_theta*cos_theta), 0.0, E_nu * cos_theta);

        // [ステップ3: ファインマン・ダイアグラムの行列要素 |M|^2 の計算]
        // トリチウムの許容遷移(V-A理論)では、行列要素は (p_T・p_nu) * (p_He・p_e) などに比例します。
        // 静止系ではこれは近似的に定数 * E_e * E_nu に比例する結果となります。
        // 物理的な確率重み (Weight) は「行列要素 |M|^2」×「位相空間の体積(p_e * dE)」です。
        
        let matrix_element_sq = 1.0; // 許容遷移なので近似的に一定
        let phase_space_density = p_e_mag * E_e * (E_nu * E_nu); // 状態密度
        
        let weight = matrix_element_sq * phase_space_density;

        // [ステップ4: Hit-and-Miss (棄却法) によるイベント受容判定]
        // 最大の weight を適当なスケールで設定 (18.6keV の計算に基づく概算最大値)
        let max_weight = 3.0e9; 
        let r = random_f32(&seed) * max_weight;

        // もし乱数が weight を下回れば、その崩壊イベントは「現実に起きた」として記録する
        if (r < weight) {
            let bin_idx = u32((K_e / Q) * f32(NUM_BINS));
            if (bin_idx < NUM_BINS) {
                atomicAdd(&histogram[bin_idx], 1u); // ヒストグラムに+1
            }
        }
    }

    rng_state[idx] = seed;
}

