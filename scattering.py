"""
scattering.py — 1次元ポテンシャルによる反射と透過の計算（描画はしない）

参考: 猪木慶治・河合光『基礎量子力学』第5章

やっていること
--------------
ポテンシャル V(x) を「一定値の区間」をつなげたもの（階段・山・山の列）として与え、
左から振幅 A = 1 の電子を入射させたときの

  - 各領域の波動関数 φ(x) の係数
  - 反射率 R と透過率 T

を求める。解き方は教科書 5.1, 5.2 とまったく同じ:

  1. 各領域で一般解 φ = (係数)·e^{ikx} + (係数)·e^{-ikx} を書く       … (5.3)(5.5)(5.25)(5.26)
  2. 右端の領域では「右から来る波」を捨てる（D = 0）                   … (5.5) の後, (5.27)
  3. 各境界で φ と dφ/dx の連続条件を書く                              … (5.6)(5.7)(5.28)〜(5.31)
  4. 出てきた連立一次方程式を解く

教科書では境界が1〜2個なので手で解いたが、ここでは境界が何個あっても
「全部の連続条件を1つの連立一次方程式に並べて、コンピュータに解かせる」という力技で解く。

単位
----
エネルギーは eV、長さは nm。ħ と m は無次元化せず、教科書の式の形のまま使う。
（1 eV 程度の電子を 1 nm 程度の障壁にぶつける、という現実的な数値で話せるようにするため）
"""

import cmath
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# 物理定数
# ---------------------------------------------------------------------------
# 式に現れるのは常に 2m/ħ² の組み合わせなので、ħ²/2m を1つの定数にまとめておく。
# ħ²/2m = (ħc)² / (2mc²) と c を掛けて割ると、覚えやすい ħc と mc² で書ける
# （教科書 例題6 (p.93) で ρ を計算したときと同じ手口）。
HBAR_C = 197.3269804  # ħc [eV·nm]   教科書 例題6 では 197 eV·nm に丸めている
ELECTRON_MC2 = 0.51099895e6  # 電子の静止エネルギー mc² [eV]   教科書では 0.51 MeV
HBAR2_OVER_2M = HBAR_C**2 / (2 * ELECTRON_MC2)  # ħ²/2m ≈ 0.0381 [eV·nm²]
# 時間因子 e^{-iEt/ħ} 用の ħ。E が eV なので、ħ を eV·fs で持つと t がフェムト秒（1 fs = 10^-15 s）になる
HBAR_EV_FS = 0.6582119569  # ħ [eV·fs]

# |k| がこれより小さい領域は「k = 0（E = V ちょうど）」として扱う [1/nm]。
# k = 0 だと e^{ikx} と e^{-ikx} が同じ関数 1 になってしまい、2つの基本解にならないため。
K_ZERO = 1e-6


# ---------------------------------------------------------------------------
# 計算結果をまとめる入れ物
# ---------------------------------------------------------------------------
class Scattering(NamedTuple):
    """
    solve_scattering() の結果。
    NamedTuple は「名前付きで値を持てるタプル」で、作った後に書き換えられないので型安全

    領域の番号は左から 0, 1, ..., M（境界が M 個なら領域は M+1 個）。

    energy          : 入射電子のエネルギー E [eV]
    boundaries      : 境界の位置 (x_0, x_1, ..., x_{M-1}) [nm]、左から順に
    potentials      : 各領域のポテンシャル (V_0, V_1, ..., V_M) [eV]
    wave_numbers    : 各領域の波数 k_j [1/nm]。E < V_j の領域では純虚数 k_j = iρ_j
    coefficients    : 各領域の係数 (c_j1, c_j2)。φ_j = c_j1·f_1 + c_j2·f_2（f は _basis() を参照）
                      左端は (A, B) = (1, B)、右端は (C, 0)
    reflectance     : 反射率 R
    transmittance   : 透過率 T
    """

    energy: float
    boundaries: tuple[float, ...]
    potentials: tuple[float, ...]
    wave_numbers: tuple[complex, ...]
    coefficients: tuple[tuple[complex, complex], ...]
    reflectance: float
    transmittance: float


# ---------------------------------------------------------------------------
# ポテンシャルの形を作る
# ---------------------------------------------------------------------------

# 単純な階段型ポテンシャル用の関数。境界の位置と各領域のポテンシャルを返す。
def step(height: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """
    階段型ポテンシャル（教科書 図5.1, 図5.6）: x < 0 で V = 0、x > 0 で V = height。

    戻り値は (境界の位置, 各領域のポテンシャル)。
    """
    return (0.0,), (0.0, float(height))


def barrier_array(
    count: int, height: float, width: float, gap: float
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """
    障壁の数や高さ、幅、間隔を指定すると、境界の位置と各領域のポテンシャルを返す関数
    高さ height、幅 width の長方形の山を、間隔 gap で count 個並べたポテンシャル。

    count = 1 が教科書 図5.8 の長方形のポテンシャル障壁（0 < x < a で V = V_0）。
    山と山の間と、両端の外側は V = 0。

        V
        ┌──┐     ┌──┐     ┌──┐
        │  │     │  │     │  │
    ────┘  └─────┘  └─────┘  └────  x
        0  a    a+b

    戻り値は (境界の位置, 各領域のポテンシャル)。count = 0 なら境界なし（自由空間）。
    """
    
    if count < 0:
        raise ValueError(f"山の個数は 0 以上にしてください（count = {count}）")
    if count > 0 and width <= 0:
        raise ValueError(f"山の幅は正にしてください（width = {width} nm）")
    if count > 1 and gap <= 0:
        raise ValueError(f"山の間隔は正にしてください（gap = {gap} nm）")

    # i 番目の山は [i(a+b), i(a+b)+a] を占める
    lefts = [i * (width + gap) for i in range(count)]
    boundaries = tuple(x for left in lefts for x in (left, left + width))
    # 領域は「外(0), 山(V), 谷(0), 山(V), ..., 山(V), 外(0)」の交互
    potentials = tuple(float(height) if j % 2 == 1 else 0.0 for j in range(2 * count + 1))
    return boundaries, potentials


def potential_profile(x, boundaries, potentials) -> np.ndarray:
    """位置 x（配列可）でのポテンシャル V(x) を返す。描画用。"""
    region = np.searchsorted(np.asarray(boundaries, dtype=float), x, side="right")
    return np.asarray(potentials, dtype=float)[region]


# ---------------------------------------------------------------------------
# 各領域の解
# ---------------------------------------------------------------------------

def wave_number(energy: float, potential: float, mass_ratio: float = 1.0) -> complex:
    """
        領域の波数 k = √(2m(E − V)) / ħ を返す関数。

        mass_ratio は質量 m を電子の質量 m₀ の何倍とするか（m = mass_ratio × m₀）。省略すると 1（自由電子）。
        半導体中の電子のように「有効質量」m* を使うときは m*/m₀ を渡す。
        式の形は変わらず、m が置き換わるだけ: 2m/ħ² = mass_ratio × (2m₀/ħ²) = mass_ratio / (ħ²/2m₀)

        E > V : k は実数で、k = √(2m(E − V)) / ħがそのまま計算される 教科書 (5.2)(5.4)(5.36)
        E < V : k = iρ（純虚数）, ρ = √(2m(V − E)) / ħ。       教科書 (5.15)(5.24)

    """
    # complex(...) にしておくと、負の数の平方根が i√|…| として計算される（虚部 +0 なので +i 側になる）
    # そのためE,Vの大小で場合分けは不要になっている。勝手に E < V のときは k = iρ となる。
    return cmath.sqrt(complex((energy - potential) * mass_ratio / HBAR2_OVER_2M))


def _basis(k: complex, x_local):
    """
    波数 k の領域での2つの基本解 f_1, f_2 と、その微分 f_1', f_2' を返す。
    係数は後で連続条件を求めるときに使うので、ここではまだ登場しない
    
    x_local は「その領域の原点」から測った位置。領域ごとに原点をずらすのは、
    e^{±ρx} を遠い x で評価すると値が極端に大きく/小さくなり、数値誤差が増えるのを防ぐため。
    （原点をずらしても係数に定数倍がかかるだけで、|係数|² で決まる R, T は変わらない）

    k ≠ 0 : f_1 = e^{ikx}, f_2 = e^{-ikx}                      教科書 (5.3)(5.5)
    k = 0 : φ'' = 0 の一般解 φ = c_1 + c_2 x を使う。
            特性方程式 λ² = 0 が重解 λ = 0 を持つ場合の基本解 {1, x} にあたる。
            （e^{ikx} と e^{-ikx} は k → 0 で両方 1 になって区別がつかなくなるため）
    """
    x_local = np.asarray(x_local, dtype=complex)
    
    # kの絶対値が小さい場合(= Eが小さい場合)は、k = 0 として扱う。(e^{ikx} と e^{-ikx} が同じ関数 1 になってしまうため。)
    if abs(k) < K_ZERO: 
        one = np.ones_like(x_local)
        return one, x_local, np.zeros_like(x_local), one
    
    # k ≠ 0 の場合は、e^{ikx} と e^{-ikx} を返す。
    e_plus = np.exp(1j * k * x_local)
    e_minus = np.exp(-1j * k * x_local)
    
    # e^ikxと e^{-ikx}、そしてその微分を返す。
    # e^{ikx} の微分は 1j * k * e^{ikx}、e^{-ikx} の微分は -1j * k * e^{-ikx} となる。
    return e_plus, e_minus, 1j * k * e_plus, -1j * k * e_minus


def _region_origins(boundaries: tuple[float, ...]) -> tuple[float, ...]:
    """
    各領域の原点。領域 j (≥1) は自分の左端の境界、左端の領域 0 は最初の境界を原点にする。
    教科書の図5.8 では最初の境界が x = 0 なので、左端の係数 A, B は教科書のものと一致する。
    """
    first = boundaries[0] if boundaries else 0.0
    return (first, *boundaries)


# ---------------------------------------------------------------------------
# 連続条件を並べて解く
# ---------------------------------------------------------------------------
def solve_scattering(energy: float, boundaries, potentials, mass_ratio: float = 1.0) -> Scattering:
    """
    エネルギー energy [eV] の電子を左から入射させたときの波動関数の係数と R, T を求める。

    入力
      energy     : 入射電子のエネルギー E [eV]。左端の領域のポテンシャルより大きくなければならない
      boundaries : 境界の位置 [nm]（左から順、M 個）
      potentials : 各領域のポテンシャル [eV]（M+1 個）
      mass_ratio : 質量 m / 電子の質量 m₀（全領域で同じ値）。省略すると 1
                   全領域で同じ質量なら、境界で φ と φ' が連続という条件はそのまま使える

    未知数と方程式の数え上げ（教科書 5.1 の「数え上げ」と同じ考え方）
      - 各領域に係数が2個 → 2(M+1) 個
      - 左端の入射波の係数 A は 1 と決める（答えは比でしか決まらないので）         → −1
      - 右端の「右から来る波」の係数は 0（左からだけ入射させる実験だから）       → −1
      - 残りの未知数は 2M 個。境界1つにつき連続条件が2本（φ と φ'）→ 方程式も 2M 本
      未知数と方程式の数がちょうど釣り合うので、どんな E でも解が1つ決まる（E は量子化されない）。
      なおA=1と決めているため、あくまでも係数の比が求められているだけで、絶対値がもとまるわけではないことに注意
    """
    
    # 入力を float に変換して、型チェックと物理的な前提条件のチェック
    # xbは境界の位置、vsは各領域のポテンシャルである
    xb = tuple(float(x) for x in boundaries)
    vs = tuple(float(v) for v in potentials)
    _validate(energy, xb, vs)

    # 各領域の波数kを計算する。
    # wave_number() により自動的に E < V の領域では k = iρ となる
    if mass_ratio <= 0:
        raise ValueError(f"質量の比 mass_ratio は正にしてください（mass_ratio = {mass_ratio}）")
    ks = tuple(wave_number(energy, v, mass_ratio) for v in vs)
    if abs(ks[0]) < K_ZERO:
        raise ValueError(
            f"E = {energy} eV が左端のポテンシャル V = {vs[0]} eV に近すぎて、入射波 e^(ikx) が作れません"
        )
    # 境界の数を数えておく。境界がない場合は自由な散乱状態になり、自明解 R = 0, T = 1 となる
    m = len(xb)
    if m == 0:
        # 境界がない（自由空間）: 入射波がそのまま進むだけ。R = 0, T = 1。自明。
        return Scattering(energy, xb, vs, ks, ((1.0 + 0j, 0j),), 0.0, 1.0)

    # 接続条件を行列形式で並べて連立方程式を作る
    # 右側にA = 1 の項を移項して、未知数だけの連立方程式にする
    matrix, rhs = _continuity_equations(xb, ks)
    
    try:
        # numpyの線形代数ライブラリを使って連立方程式を解く
        unknowns = np.linalg.solve(matrix, rhs)
        
    except np.linalg.LinAlgError as err:
        raise ValueError(
            f"連続条件の連立方程式が解けませんでした（E = {energy} eV, 境界 = {xb}, V = {vs}）"
        ) from err

    # 並べた未知数 [B, c_11, c_12, ..., c_(M-1)1, c_(M-1)2, C] を領域ごとの組に戻す
    coefficients = (
        (1.0 + 0j, unknowns[0]),  # 左端: (A, B) = (1, B)
        *((unknowns[2 * j - 1], unknowns[2 * j]) for j in range(1, m)),
        (unknowns[-1], 0j),  # 右端: (C, 0)
    )

    return Scattering(
        energy=energy,
        boundaries=xb,
        potentials=vs,
        wave_numbers=ks,
        coefficients=coefficients,
        reflectance=_reflectance(coefficients),
        transmittance=_transmittance(coefficients, ks),
    )


def _validate(energy: float, xb: tuple[float, ...], vs: tuple[float, ...]) -> None:
    """入力の形と物理的な前提をチェックする。おかしければ理由付きで ValueError を出す。"""
    if len(vs) != len(xb) + 1:
        raise ValueError(
            f"領域の数（ポテンシャル {len(vs)} 個）は境界の数（{len(xb)} 個）+ 1 でなければなりません"
        )
    if any(right <= left for left, right in zip(xb, xb[1:])):
        raise ValueError(f"境界の位置は左から順に、重ならないように並べてください: {xb}")
    if energy <= vs[0]:
        raise ValueError(
            f"E = {energy} eV が左端のポテンシャル V = {vs[0]} eV 以下です。"
            "左から入射する波（散乱状態）を考えるには E > V（左端）が必要です"
        )


def _continuity_equations(xb: tuple[float, ...], ks: tuple[complex, ...]):
    """
    全境界の連続条件を、連立一次方程式 matrix × unknowns = rhs の形に並べる。

    境界 n（領域 n と n+1 の間, 位置 x_n）での連続条件:
        φ_n(x_n)  = φ_{n+1}(x_n)        教科書 (5.6)(5.28)(5.30) と同じ
        φ_n'(x_n) = φ_{n+1}'(x_n)       教科書 (5.7)(5.29)(5.31) と同じ
    これを「左辺 − 右辺 = 0」の形に移項して、行列の 2n 行目と 2n+1 行目に書き込む。

    未知数の並び順: 領域 j の i 番目の係数（i = 0, 1）は列 2j + i − 1 に入る。
        → [B, c_10, c_11, c_20, c_21, ..., C]
    既知の A = 1 の項は右辺 rhs に移し、捨てた右端の第2係数（D = 0）の項は書かない。
    """
    
    # 初期化
    m = len(xb)
    last = m  # 右端の領域の番号
    origins = _region_origins(xb) # 各領域の原点
    matrix = np.zeros((2 * m, 2 * m), dtype=complex) # 未知数B,C, ...にかかる係数の行列
    rhs = np.zeros(2 * m, dtype=complex) # 右辺の既知の項（A = 1 の項を移項したもの）

    # 各境界での連続条件を0埋めした行列matrix, rhsに書き込んでいく
    for n, x in enumerate(xb):
        # sign = +1: 境界の左側の領域 n、sign = −1: 右側の領域 n+1（移項したので符号が逆）
        for region, sign in ((n, +1.0), (n + 1, -1.0)):
            f1, f2, df1, df2 = _basis(ks[region], x - origins[region])
            for i, (value, slope) in enumerate(((f1, df1), (f2, df2))):
                if region == 0 and i == 0:
                    # 入射波 A·e^{ikx} は A = 1 と決めてあるので既知。右辺に移す
                    rhs[2 * n] -= sign * value
                    rhs[2 * n + 1] -= sign * slope
                elif region == last and i == 1:
                    # 右端の「右から来る波」は D = 0 と決めたので、方程式に現れない
                    continue
                else:
                    column = 2 * region + i - 1
                    matrix[2 * n, column] = sign * value
                    matrix[2 * n + 1, column] = sign * slope
    
    # 最終的に係数行列と右辺ベクトルを返す。
    # これを Numpyの線形代数ライブラリで解くと、未知数の係数が求まる
    return matrix, rhs


def _reflectance(coefficients) -> float:
    """
    反射率 R = |B/A|² 。教科書 (5.10)(5.12)
    左端の領域で入射波と反射波の速さは同じ（同じ k）なので、流れの比は振幅の比の2乗だけになる。A = 1。
    """
    return float(abs(coefficients[0][1]) ** 2)


def _transmittance(coefficients, ks) -> float:
    """
    透過率 T = (v_右 |C|²) / (v_左 |A|²) = (k_右 / k_左) |C/A|² 。教科書 (5.11)(5.13)

    右端で E < V（k が純虚数）や E = V（k = 0）なら、右端の波は進まず流れを運ばないので T = 0。
    （階段型の E < V_0 で T = 0, R = 1 になる教科書 (5.22) の状況）
    """
    k_left, k_right = ks[0], ks[-1]
    if abs(k_right) < K_ZERO:
        return 0.0
    c_right = coefficients[-1][0]
    return float(k_right.real / k_left.real * abs(c_right) ** 2)


# ---------------------------------------------------------------------------
# 結果を使う
# ---------------------------------------------------------------------------
def wavefunction(x, solution: Scattering) -> np.ndarray:
    """
    位置 x（配列可）での波動関数 φ(x) を返す（複素数。入射波の振幅 A = 1）。
    unknowns から係数を取り出して、波動関数 φ_j = c_j1·f_1 + c_j2·f_2 を組み立てる。
    x がどの領域に入るかを調べ、その領域の係数と基本解を組み立てる。
    """
    # 位置xを取り出してfloat型の配列に変換
    x = np.asarray(x, dtype=float)
    
    # どの領域に入るかを調べる。np.searchsorted() は、境界の配列に対して、x がどの位置に入るかを返す。
    region = np.searchsorted(np.asarray(solution.boundaries, dtype=float), x, side="right")
    
    # 各領域の原点を求める
    origins = _region_origins(solution.boundaries)
    
    # 波動関数 φ(x) の値を格納する配列を0埋めで初期化。もちろん複素数型
    phi = np.zeros(x.shape, dtype=complex)
    
    # 各領域の係数と基本解を組み立ててphiに代入していく
    for j, (k, (c1, c2)) in enumerate(zip(solution.wave_numbers, solution.coefficients)):
        inside = region == j
        f1, f2, _, _ = _basis(k, x[inside] - origins[j])
        phi[inside] = c1 * f1 + c2 * f2
    return phi


def traveling_components(x, solution: Scattering) -> tuple[np.ndarray, np.ndarray]:
    """
    φ(x) を「右向きに進む成分」と「左向きに進む成分」に分けて返す: (右向き, 左向き)。

    各領域の解は φ_j = c_j1 e^{ik_j x} + c_j2 e^{-ik_j x} と、もともと2つの項の和で書いてある。
      左端の領域: 右向き = 入射波 A e^{ik_1x}、左向き = 反射波 B e^{-ik_1x}       教科書 (5.3)(5.25)
      右端の領域: 右向き = 透過波 C e^{ik x}、左向き = 0（D = 0 としたので）       教科書 (5.5)(5.27)
      山の上（E > V）や山と山の間: 両向きの波が行き来している
    どちらの成分にも時間因子 e^{-iEt/ħ} を掛ければ、右向き成分は右へ、左向き成分は左へ流れる。

    E < V の領域（k = iρ）では e^{-ρx}, e^{+ρx} となり、これらは進む波ではない（1つだけでは流れを運ばない）。
    E = V の領域の {1, x} も同様。こうした領域は「分けられない」ので NaN（非数）を入れて返す。
    matplotlib は NaN の点を描かずに飛ばすので、グラフではその区間だけ線が途切れる。

    右向き + 左向き = wavefunction(x, solution) が、進む波の領域で成り立つ。
    """
    x = np.asarray(x, dtype=float)
    region = np.searchsorted(np.asarray(solution.boundaries, dtype=float), x, side="right")
    origins = _region_origins(solution.boundaries)
    right = np.full(x.shape, np.nan, dtype=complex)
    left = np.full(x.shape, np.nan, dtype=complex)
    for j, (k, (c1, c2)) in enumerate(zip(solution.wave_numbers, solution.coefficients)):
        if not _is_propagating(k):
            continue
        inside = region == j
        f1, f2, _, _ = _basis(k, x[inside] - origins[j])
        right[inside] = c1 * f1
        left[inside] = c2 * f2
    return right, left


def _is_propagating(k: complex) -> bool:
    """k が正の実数（E > V）なら、e^{±ikx} は進む波。E < V（k = iρ）や E = V（k = 0）なら進まない"""
    return k.imag == 0.0 and abs(k) >= K_ZERO


def time_evolve(phi, energy: float, t: float) -> np.ndarray:
    """
    定常状態の時間発展 ψ(x, t) = φ(x) e^{-iEt/ħ} を返す。

    教科書 p.84「(5.3) の右辺には e^{-iEt/ħ} が常にかかっている」の e^{-iEt/ħ}。
    （変数分離で出てきた時間因子。ノート「定常解の再構成_積分定数を全部残して.md」）

    入力
      phi    : 波動関数 φ(x) の値（複素数の配列。wavefunction() の戻り値）
      energy : E [eV]
      t      : 時刻 [fs]
    物理的意味
      全体に同じ位相 e^{-iEt/ħ} を掛けるだけなので |ψ|² = |φ|² は時間によらない（定常状態）。
      一方、実部 Re ψ は時間とともに変わり、進行波 e^{ikx} の部分は右へ流れ、
      e^{ikx} と e^{-ikx} が同じ大きさで重なった部分はその場で上下する（定在波）。
    """
    
    # 
    return np.asarray(phi, dtype=complex) * np.exp(-1j * energy * t / HBAR_EV_FS)


def period(energy: float) -> float:
    """時間因子 e^{-iEt/ħ} が1回転する時間 2πħ/E [fs]（ω = E/ħ の周期）"""
    return 2 * np.pi * HBAR_EV_FS / energy


def spectrum(energies, boundaries, potentials, mass_ratio: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    エネルギーの配列 energies [eV] それぞれについて R, T を求め、(R の配列, T の配列) を返す。
    各エネルギーで solve_scattering を1回ずつ呼ぶ（力技）。mass_ratio は solve_scattering と同じ。
    """
    solutions = [solve_scattering(e, boundaries, potentials, mass_ratio) for e in energies]
    # 反射率 R と透過率 T の配列を作る
    reflectances = np.array([s.reflectance for s in solutions])
    transmittances = np.array([s.transmittance for s in solutions])
    return reflectances, transmittances


def find_resonances(
    boundaries,
    potentials,
    e_low: float,
    e_high: float,
    mass_ratio: float = 1.0,
    min_transmittance: float = 0.99,
    points: int = 20001,
) -> tuple[float, ...]:
    """
    E の範囲 [e_low, e_high] [eV] で、透過率 T が山（極大）になり、かつ T ≥ min_transmittance となる
    エネルギー（共鳴エネルギー）を小さい順に返す。

    探し方（力技）
      1. 範囲を points 点に区切って T を計算し、両隣より大きい点（山の候補）を見つける
      2. 各候補の両隣の点の間で「黄金分割探索」を行い、T が最大になる E を 10^-12 eV の精度まで絞り込む
         （黄金分割探索: 区間を一定の比率で狭めながら最大の位置を挟み込んでいく方法。微分を使わずに済む）

    注意: 共鳴ピークの幅が格子の間隔より細いと、手順1 で見落とすことがある。そのときは points を増やす。
    """
    energies = np.linspace(max(e_low, 1e-12), e_high, points)
    transmittances = spectrum(energies, boundaries, potentials, mass_ratio)[1]
    is_peak = (transmittances[1:-1] >= transmittances[:-2]) & (transmittances[1:-1] >= transmittances[2:])
    candidates = np.flatnonzero(is_peak) + 1  # 両端を除いた配列で探したので、番号を1つずらす

    def t_of(e):
        return solve_scattering(e, boundaries, potentials, mass_ratio).transmittance

    peaks = (_golden_section_max(t_of, energies[i - 1], energies[i + 1]) for i in candidates)
    return tuple(e for e in peaks if t_of(e) >= min_transmittance)


def _golden_section_max(func, low: float, high: float, tol: float = 1e-12) -> float:
    """区間 [low, high] で山が1つだけの関数 func が最大になる位置を、黄金分割探索で求める"""
    ratio = (np.sqrt(5) - 1) / 2  # 黄金比 0.618...
    a, b = low, high
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = func(c), func(d)
    while b - a > tol:
        if fc >= fd:  # 最大は [a, d] にある
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = func(c)
        else:  # 最大は [c, b] にある
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = func(d)
    return (a + b) / 2
