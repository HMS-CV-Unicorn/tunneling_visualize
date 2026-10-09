"""
test_scattering.py — scattering.py の計算が正しいかを、教科書の解析式と突き合わせて確かめる

実行: .venv/bin/python -m pytest -v
（pytest は test_ で始まる関数を自動で見つけて実行し、assert が成り立つかを調べるツール）

pytest.approx(値, rel=...) は「相対誤差 rel 以内で等しい」という比較。
コンピュータの小数計算には丸め誤差があるので、== ではなくこれで比べる。
"""

import math

import numpy as np
import pytest

import scattering as sc


def k_of(energy, potential=0.0):
    """テスト用: 実数の波数 √(2m(E−V))/ħ（E > V を想定）"""
    return math.sqrt((energy - potential) / sc.HBAR2_OVER_2M)


def rho_of(energy, potential):
    """テスト用: 減衰の速さ ρ = √(2m(V−E))/ħ（E < V を想定）。教科書 (5.15)(5.24)"""
    return math.sqrt((potential - energy) / sc.HBAR2_OVER_2M)


# ---------------------------------------------------------------------------
# 1. ポテンシャルなし
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("energy", [0.01, 0.5, 3.0])
def test_free_space_transmits_everything(energy):
    """山が0個（V = 0 がずっと続く）なら、跳ね返る理由がないので T = 1, R = 0"""
    boundaries, potentials = sc.barrier_array(0, 1.0, 0.5, 0.5)
    result = sc.solve_scattering(energy, boundaries, potentials)
    assert result.transmittance == pytest.approx(1.0, rel=1e-12)
    assert result.reflectance == pytest.approx(0.0, abs=1e-12)


def test_zero_height_barriers_transmit_everything():
    """境界はあっても高さ 0 の山なら、連立方程式を解いた結果も T = 1, R = 0 になるはず"""
    boundaries, potentials = sc.barrier_array(3, 0.0, 0.5, 0.7)
    result = sc.solve_scattering(1.2, boundaries, potentials)
    assert result.transmittance == pytest.approx(1.0, rel=1e-12)
    assert result.reflectance == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. 階段型ポテンシャル（5.1）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("energy, height", [(65.0, 50.0), (2.0, 1.0), (1.0, 0.3)])
def test_step_above_matches_textbook(energy, height):
    """E > V_0 の階段: 係数 (5.8)(5.9) と R, T (5.12)(5.13) に一致する"""
    k1, k2 = k_of(energy), k_of(energy, height)
    result = sc.solve_scattering(energy, *sc.step(height))
    (a, b), (c, _) = result.coefficients
    assert b / a == pytest.approx((k1 - k2) / (k1 + k2), rel=1e-12)  # (5.8)
    assert c / a == pytest.approx(2 * k1 / (k1 + k2), rel=1e-12)  # (5.9)
    assert result.reflectance == pytest.approx(((k1 - k2) / (k1 + k2)) ** 2, rel=1e-12)  # (5.12)
    assert result.transmittance == pytest.approx(4 * k1 * k2 / (k1 + k2) ** 2, rel=1e-12)  # (5.13)


def test_step_example5_p87():
    """例題5 (p.87): E = 65 eV, V_0 = 50 eV で R = 0.12, T = 0.88"""
    result = sc.solve_scattering(65.0, *sc.step(50.0))
    assert round(result.reflectance, 2) == 0.12
    assert round(result.transmittance, 2) == 0.88


def test_step_down_example7_p88():
    """例題7 (p.88): 下り階段 E = 10 eV, V_0 = −50 eV でも R = 0.18 だけ跳ね返る"""
    result = sc.solve_scattering(10.0, *sc.step(-50.0))
    assert round(result.reflectance, 2) == 0.18
    assert round(result.transmittance, 2) == 0.82


@pytest.mark.parametrize("energy, height", [(0.5, 1.0), (8.0, 10.0)])
def test_step_below_total_reflection(energy, height):
    """E < V_0 の階段: B/A (5.20), D/A (5.21), R = 1 (5.22), T = 0"""
    k1, rho = k_of(energy), rho_of(energy, height)
    result = sc.solve_scattering(energy, *sc.step(height))
    (a, b), (d, _) = result.coefficients
    assert b / a == pytest.approx((k1 - 1j * rho) / (k1 + 1j * rho), rel=1e-12)  # (5.20)
    assert d / a == pytest.approx(2 * k1 / (k1 + 1j * rho), rel=1e-12)  # (5.21)
    assert result.reflectance == pytest.approx(1.0, rel=1e-12)  # (5.22)
    assert result.transmittance == 0.0


# ---------------------------------------------------------------------------
# 3. 長方形の山1個（5.2）
# ---------------------------------------------------------------------------
def barrier_t_below(energy, height, width):
    """E < V_0: 教科書 例題4 ⓓ (p.92)"""
    s = math.sinh(rho_of(energy, height) * width)
    return 4 * energy * (height - energy) / (4 * energy * (height - energy) + height**2 * s**2)


def barrier_t_above(energy, height, width):
    """E > V_0: 教科書 (5.37)"""
    s = math.sin(k_of(energy, height) * width)
    return 4 * energy * (energy - height) / (4 * energy * (energy - height) + height**2 * s**2)


@pytest.mark.parametrize("energy, height, width", [(0.5, 1.0, 0.3), (0.2, 1.0, 1.0), (0.9, 2.0, 0.5)])
def test_single_barrier_tunneling(energy, height, width):
    """E < V_0 の山1個（トンネル効果）が 例題4 ⓓ に一致する"""
    result = sc.solve_scattering(energy, *sc.barrier_array(1, height, width, 1.0))
    assert result.transmittance == pytest.approx(barrier_t_below(energy, height, width), rel=1e-10)


@pytest.mark.parametrize("energy, height, width", [(1.5, 1.0, 0.3), (3.0, 1.0, 1.0), (2.1, 2.0, 0.5)])
def test_single_barrier_above(energy, height, width):
    """E > V_0 の山1個が (5.37)(5.38) に一致する"""
    result = sc.solve_scattering(energy, *sc.barrier_array(1, height, width, 1.0))
    t = barrier_t_above(energy, height, width)
    assert result.transmittance == pytest.approx(t, rel=1e-10)  # (5.37)
    assert result.reflectance == pytest.approx(1 - t, rel=1e-10)  # (5.38)


def test_example6_p93_very_small_transmission():
    """
    例題6 (p.93): V_0 = 10 eV, a = 1 nm の山に E = 5 eV。T ≈ 10^-10 という極端に小さい値でも
    例題4 ⓓ に一致するか（数値誤差に弱い場面の確認）
    """
    result = sc.solve_scattering(5.0, *sc.barrier_array(1, 10.0, 1.0, 1.0))
    assert result.transmittance == pytest.approx(barrier_t_below(5.0, 10.0, 1.0), rel=1e-6)
    # 教科書は ħc = 197 eV·nm, mc² = 0.51 MeV に丸めて 4 × 1.13 × 10^-10 ≈ 4.5 × 10^-10 としている
    assert result.transmittance == pytest.approx(4.5e-10, rel=0.1)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_single_barrier_resonance(n):
    """k_2 a = nπ のとき T = 1（共鳴, p.96 の注意(2)）"""
    height, width = 1.0, 0.8
    energy = height + sc.HBAR2_OVER_2M * (n * math.pi / width) ** 2  # k_2 = nπ/a となる E
    result = sc.solve_scattering(energy, *sc.barrier_array(1, height, width, 1.0))
    assert result.transmittance == pytest.approx(1.0, rel=1e-10)


# ---------------------------------------------------------------------------
# 4. R + T = 1（確率の流れの保存, (5.14)）
# ---------------------------------------------------------------------------
def test_flux_conservation_random_barriers():
    """山の個数・高さ・幅・間隔・E をランダムに選んでも、いつも R + T = 1"""
    rng = np.random.default_rng(seed=0)  # 乱数の種を固定して、毎回同じ乱数列にする
    worst = 0.0
    for _ in range(500):
        count = int(rng.integers(1, 6))
        height = rng.uniform(-3.0, 5.0)
        width, gap = rng.uniform(0.05, 2.0), rng.uniform(0.05, 3.0)
        energy = rng.uniform(0.01, 6.0)
        result = sc.solve_scattering(energy, *sc.barrier_array(count, height, width, gap))
        worst = max(worst, abs(result.reflectance + result.transmittance - 1))
    assert worst < 1e-9, f"R + T − 1 の最大のずれ = {worst:.2e}"


# ---------------------------------------------------------------------------
# 5. E = V_0 ちょうど（k = 0 の特別扱い）
# ---------------------------------------------------------------------------
def test_energy_equal_to_height_is_continuous():
    """
    E = V_0 ちょうどでは基本解を {1, x} に切り替えている。その値が、E を V_0 に左右から
    近づけたときの値と滑らかにつながるか。
    例題4 ⓓ（または (5.37)）で E → V_0 の極限をとると sinh ρa ≈ ρa から
        T → 1 / (1 + V_0 a² / (4 ħ²/2m))
    """
    height, width = 1.0, 0.5
    boundaries, potentials = sc.barrier_array(1, height, width, 1.0)
    t_exact = sc.solve_scattering(height, boundaries, potentials).transmittance
    t_limit = 1 / (1 + height * width**2 / (4 * sc.HBAR2_OVER_2M))
    assert t_exact == pytest.approx(t_limit, rel=1e-12)
    for delta in (1e-6, -1e-6):
        t_near = sc.solve_scattering(height + delta, boundaries, potentials).transmittance
        assert t_near == pytest.approx(t_limit, rel=1e-5)


# ---------------------------------------------------------------------------
# 6. 波動関数そのもの
# ---------------------------------------------------------------------------
def test_wavefunction_is_smooth_at_boundaries():
    """組み立てた φ(x) が、各境界で値も傾きもつながっている（連続条件が本当に満たされている）"""
    boundaries, potentials = sc.barrier_array(3, 1.5, 0.4, 0.6)
    result = sc.solve_scattering(0.8, boundaries, potentials)
    eps = 1e-7
    for x in boundaries:
        left, right = sc.wavefunction([x - eps, x + eps], result)
        assert left == pytest.approx(right, abs=1e-5)
        # 傾きは境界の両側で、片側だけの差分をとって比べる
        slope_left = (sc.wavefunction(x - eps, result) - sc.wavefunction(x - 2 * eps, result)) / eps
        slope_right = (sc.wavefunction(x + 2 * eps, result) - sc.wavefunction(x + eps, result)) / eps
        assert slope_left == pytest.approx(slope_right, abs=1e-4)


# ---------------------------------------------------------------------------
# 7. おかしな入力にはメッセージ付きでエラーを出す
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("energy", [0.0, -1.0])
def test_energy_must_exceed_left_potential(energy):
    with pytest.raises(ValueError, match="E > V"):
        sc.solve_scattering(energy, *sc.step(1.0))


def test_boundaries_must_be_increasing():
    with pytest.raises(ValueError, match="境界"):
        sc.solve_scattering(1.0, (1.0, 0.5), (0.0, 1.0, 0.0))


# ---------------------------------------------------------------------------
# 8. 時間因子 e^{-iEt/ħ}
# ---------------------------------------------------------------------------
def test_probability_density_is_stationary():
    """|ψ(x,t)|² は時間によらず |φ(x)|² のまま（定常状態）"""
    boundaries, potentials = sc.barrier_array(2, 1.0, 0.5, 1.0)
    result = sc.solve_scattering(0.7, boundaries, potentials)
    x = np.linspace(-3, 5, 400)
    phi = sc.wavefunction(x, result)
    for t in (0.0, 0.37, 2.5, 100.0):
        psi = sc.time_evolve(phi, 0.7, t)
        assert np.abs(psi) ** 2 == pytest.approx(np.abs(phi) ** 2, rel=1e-12)


def test_time_factor_returns_after_one_period():
    """t = 2πħ/E で e^{-iEt/ħ} が1回転して元に戻る"""
    energy = 1.3
    result = sc.solve_scattering(energy, *sc.step(0.5))
    phi = sc.wavefunction(np.linspace(-3, 3, 200), result)
    psi = sc.time_evolve(phi, energy, sc.period(energy))
    assert psi == pytest.approx(phi, abs=1e-12)


def test_traveling_wave_moves_right_at_phase_velocity():
    """
    自由空間の e^{ikx} に e^{-iEt/ħ} を掛けると、形を変えずに右へ速さ ω/k で進む（教科書 p.84, (5.3) の直後）:
        ψ(x, t) = φ(x − (ω/k) t)
    """
    energy = 0.8
    result = sc.solve_scattering(energy, *sc.barrier_array(0, 1.0, 0.5, 0.5))
    k = sc.wave_number(energy, 0.0).real
    omega = energy / sc.HBAR_EV_FS  # [1/fs]
    t = 0.9
    x = np.linspace(-2, 2, 300)
    psi = sc.time_evolve(sc.wavefunction(x, result), energy, t)
    shifted = sc.wavefunction(x - omega / k * t, result)
    assert psi == pytest.approx(shifted, abs=1e-12)


# ---------------------------------------------------------------------------
# 9. 右向き成分・左向き成分への分解
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("energy", [0.4, 0.7, 1.6])
def test_components_add_up_to_wavefunction(energy):
    """進む波の領域では 右向き + 左向き = φ。E < V の山の中は分けられないので NaN"""
    boundaries, potentials = sc.barrier_array(2, 1.0, 0.5, 1.0)
    result = sc.solve_scattering(energy, boundaries, potentials)
    x = np.linspace(-3, 5, 2000)
    right, left = sc.traveling_components(x, result)
    phi = sc.wavefunction(x, result)
    in_barrier = sc.potential_profile(x, boundaries, potentials) > energy
    assert np.all(np.isnan(right[in_barrier])) and np.all(np.isnan(left[in_barrier]))
    ok = ~in_barrier
    assert right[ok] + left[ok] == pytest.approx(phi[ok], abs=1e-12)


def test_component_amplitudes_are_incident_reflected_transmitted():
    """左端: 入射の振幅 1、反射の振幅 |B| = √R。右端: 透過の振幅 |C| = √T（両端の V が同じとき）、左向き 0"""
    boundaries, potentials = sc.barrier_array(1, 1.0, 0.4, 1.0)
    result = sc.solve_scattering(0.8, boundaries, potentials)
    x_left, x_right = np.linspace(-3, -0.1, 50), np.linspace(0.5, 3, 50)
    right_l, left_l = sc.traveling_components(x_left, result)
    right_r, left_r = sc.traveling_components(x_right, result)
    assert np.abs(right_l) == pytest.approx(np.ones(50), rel=1e-12)
    assert np.abs(left_l) == pytest.approx(np.full(50, math.sqrt(result.reflectance)), rel=1e-12)
    assert np.abs(right_r) == pytest.approx(np.full(50, math.sqrt(result.transmittance)), rel=1e-12)
    assert np.abs(left_r) == pytest.approx(np.zeros(50), abs=1e-15)
    

#----------------------------------------------------------------------------
# 10. その他の解析解について調べる
#----------------------------------------------------------------------------
"""
バリアの高さ: 500meV
バリア幅: 20å
井戸幅（バリア間隔): 50å
有効質量を全領域において0.1m0とする

すると
共鳴準位
1つ目: 81.193meV
2つ目; 310.822meV

になるか確認したい
"""

# 単位の換算: このプログラムは eV と nm で計算するので、meV と Å を換算して渡す
MEV = 1e-3  # 1 meV = 10^-3 eV
ANGSTROM = 0.1  # 1 Å = 0.1 nm

REF_HEIGHT = 500 * MEV
REF_WIDTH = 20 * ANGSTROM
REF_GAP = 50 * ANGSTROM
REF_MASS = 0.1  # 有効質量 m*/m₀
REF_LEVELS_MEV = (81.193, 310.822)  # 参照した値
# このプログラム（CODATA の ħc, m₀c² を使用）で求めた値。find_resonances で 10^-12 eV まで絞り込んだもの
CODATA_LEVELS_MEV = (80.629867, 308.717746)


def reference_potential():
    return sc.barrier_array(2, REF_HEIGHT, REF_WIDTH, REF_GAP)


def test_reference_resonance_levels():
    """
    バリアの下（0 < E < 500 meV）にある共鳴準位は2つで、CODATA 定数での値は 80.630 meV と 308.718 meV。
    参照値 81.193 / 310.822 meV とは約 0.69% ずれる（原因は次のテストを参照）。ずれは 1% 未満であることを確認する。
    """
    levels = sc.find_resonances(*reference_potential(), 0.1 * MEV, REF_HEIGHT, mass_ratio=REF_MASS)
    levels_mev = [e / MEV for e in levels]
    assert levels_mev == pytest.approx(CODATA_LEVELS_MEV, abs=1e-5)
    for ours, ref in zip(levels_mev, REF_LEVELS_MEV):
        assert abs(ours / ref - 1) < 0.01


def test_reference_resonances_transmit_fully():
    """共鳴エネルギーでは T = 1, R = 0（左右対称な2重障壁なので完全透過）。R + T = 1 も成り立つ"""
    for level in CODATA_LEVELS_MEV:
        result = sc.solve_scattering(level * MEV, *reference_potential(), mass_ratio=REF_MASS)
        assert result.transmittance == pytest.approx(1.0, abs=1e-9)
        assert result.reflectance == pytest.approx(0.0, abs=1e-9)
        assert result.reflectance + result.transmittance == pytest.approx(1.0, abs=1e-12)


def test_reference_difference_is_one_constant_factor(monkeypatch):
    """
    参照値とのずれの原因の確認。ħ²/2m の値を1つの倍率 s で変えたとき、
    「1つ目の準位が 81.193 meV に合う s」で2つ目の準位も 310.822 meV に合うなら、
    ずれは計算方法やポテンシャルの形の違いではなく、物理定数（ħ, m₀ の値の丸め方など）の違いだと言える。
    結果: s ≈ 1.00997（ħ²/2m₀ ≈ 3.848 eV·Å²。CODATA では 3.810 eV·Å²）、または m* ≈ 0.0990 m₀ に相当する。

    monkeypatch は pytest の機能で、テストの間だけ定数を書き換え、終わったら元に戻してくれる。
    """
    base = sc.HBAR2_OVER_2M

    def levels_with_scale(scale):
        monkeypatch.setattr(sc, "HBAR2_OVER_2M", base * scale)
        windows = ((70 * MEV, 90 * MEV), (290 * MEV, 330 * MEV))  # 各準位の付近だけ探して速くする
        return [
            sc.find_resonances(*reference_potential(), low, high, mass_ratio=REF_MASS, points=201)[0] / MEV
            for low, high in windows
        ]

    # 二分法: 1つ目の準位が参照値に一致する倍率 s を探す（準位は s が大きいほど高くなる）
    low, high = 1.0, 1.02
    for _ in range(40):
        middle = (low + high) / 2
        if levels_with_scale(middle)[0] < REF_LEVELS_MEV[0]:
            low = middle
        else:
            high = middle
    scale = (low + high) / 2
    first, second = levels_with_scale(scale)

    assert scale == pytest.approx(1.00997, abs=1e-4)
    assert first == pytest.approx(REF_LEVELS_MEV[0], abs=1e-3)
    assert second == pytest.approx(REF_LEVELS_MEV[1], abs=5e-3)  # 2つ目も、合わせていないのに一致する


def test_effective_mass_equals_stretched_lengths():
    """
    mass_ratio の実装の確認。k = √(2m(E−V))/ħ なので、質量を m 倍にすることは、
    長さ（幅・間隔）を √m 倍にすることと同じ（kx の組み合わせでしか効かないため）。
        T(E; 幅 a, 間隔 b, 質量 m) = T(E; 幅 √m·a, 間隔 √m·b, 質量 1)
    """
    m = REF_MASS
    for energy in (0.05, 0.2, 0.7):
        with_mass = sc.solve_scattering(energy, *reference_potential(), mass_ratio=m)
        stretched = sc.solve_scattering(
            energy, *sc.barrier_array(2, REF_HEIGHT, math.sqrt(m) * REF_WIDTH, math.sqrt(m) * REF_GAP)
        )
        assert with_mass.transmittance == pytest.approx(stretched.transmittance, rel=1e-10)
