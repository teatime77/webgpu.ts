// ============================================================================
// Feynman Diagram & Particle Physics Base Library
// ============================================================================

// --- 1. Complex Numbers (複素数) ---
// 量子力学の確率は複素数の絶対値の2乗で計算されます。
alias Complex = vec2<f32>;

fn cmul(a: Complex, b: Complex) -> Complex {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cconj(a: Complex) -> Complex {
    return vec2(a.x, -a.y);
}

// --- 2. Four-Vectors (4元運動量ベクトル) ---
// p = (E, px, py, pz). 時間成分(エネルギー)と空間成分(運動量)をまとめたもの。
// WGSLの vec4<f32> がそのまま使えますが、内積の計算ルールが特殊です。
alias FourVector = vec4<f32>;

// ミンコフスキー計量 (+, -, -, -) に基づく4元ベクトルの内積 p * q
fn dot4(p: FourVector, q: FourVector) -> f32 {
    return p.x * q.x - p.y * q.y - p.z * q.z - p.w * q.w;
}

// 不変質量 (Invariant Mass) の計算: m^2 = E^2 - p^2
fn invariant_mass_sq(p: FourVector) -> f32 {
    return dot4(p, p);
}

// --- 3. Dirac Spinors (ディラック・スピノル) ---
// 電子やクォークなどのスピン1/2の粒子を表す4成分の複素ベクトルです。
struct Spinor {
    c0: Complex,
    c1: Complex,
    c2: Complex,
    c3: Complex,
};

// スピノルの足し算
fn add_spinor(a: Spinor, b: Spinor) -> Spinor {
    return Spinor(a.c0 + b.c0, a.c1 + b.c1, a.c2 + b.c2, a.c3 + b.c3);
}

// --- 4. Gamma Matrices (ガンマ行列との積) ---
// ガンマ行列は 4x4 の複素行列ですが、成分のほとんどが 0 か ±1, ±i です。
// GPUのメモリとレジスタを節約するため、行列をそのまま定義するのではなく、
// 「スピノルにガンマ行列 γ^μ を掛けた結果を返す関数」としてハードコードするのが
// スパコンなどでも使われる最速の実装(ディラック表現)です。

// γ^0 * psi
fn gamma_0(psi: Spinor) -> Spinor {
    return Spinor(psi.c0, psi.c1, -psi.c2, -psi.c3);
}

// γ^1 * psi
fn gamma_1(psi: Spinor) -> Spinor {
    return Spinor(
        vec2(psi.c3.x, psi.c3.y),
        vec2(psi.c2.x, psi.c2.y),
        vec2(-psi.c1.x, -psi.c1.y),
        vec2(-psi.c0.x, -psi.c0.y)
    );
}

// γ^2 * psi
fn gamma_2(psi: Spinor) -> Spinor {
    return Spinor(
        vec2(psi.c3.y, -psi.c3.x),  // -i * c3
        vec2(-psi.c2.y, psi.c2.x),  // i * c2
        vec2(psi.c1.y, -psi.c1.x),  // i * c1
        vec2(-psi.c0.y, psi.c0.x)   // -i * c0
    );
}

// γ^3 * psi
fn gamma_3(psi: Spinor) -> Spinor {
    return Spinor(
        vec2(psi.c2.x, psi.c2.y),
        vec2(-psi.c3.x, -psi.c3.y),
        vec2(-psi.c0.x, -psi.c0.y),
        vec2(psi.c1.x, psi.c1.y)
    );
}

// γ^5 = i * γ^0 * γ^1 * γ^2 * γ^3
// 弱い相互作用（パリティ対称性の破れ）を計算する際に必須となる行列です。
fn gamma_5(psi: Spinor) -> Spinor {
    return Spinor(psi.c2, psi.c3, psi.c0, psi.c1);
}

// カレント(流れ)の計算などで使う、任意の μ(0,1,2,3) 番目のガンマ行列を掛けるヘルパー
fn apply_gamma(mu: u32, psi: Spinor) -> Spinor {
    if (mu == 0u) { return gamma_0(psi); }
    if (mu == 1u) { return gamma_1(psi); }
    if (mu == 2u) { return gamma_2(psi); }
    return gamma_3(psi); // mu == 3
}

// ディラック共役 (Dirac Adjoint): psi_bar = psi^dagger * γ^0
fn dirac_adjoint(psi: Spinor) -> Spinor {
    let dagger = Spinor(cconj(psi.c0), cconj(psi.c1), cconj(psi.c2), cconj(psi.c3));
    return gamma_0(dagger);
}

// スピノル同士の内積: psi_bar * phi (スカラー量になります)
fn spinor_inner_product(psi_bar: Spinor, phi: Spinor) -> Complex {
    return cmul(psi_bar.c0, phi.c0) + cmul(psi_bar.c1, phi.c1) + 
           cmul(psi_bar.c2, phi.c2) + cmul(psi_bar.c3, phi.c3);
}

