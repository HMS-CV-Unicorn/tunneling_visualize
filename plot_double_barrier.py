"""
plot_double_barrier.py — 2重障壁の透過率と、共鳴エネルギーでの波動関数をグラフにして保存する

実行:  .venv/bin/python plot_double_barrier.py
出力:  figures/double_barrier_transmission.png   … 図1 透過率 T と E [meV]
       figures/double_barrier_wavefunction.png   … 図2 共鳴エネルギーでの |ψ(x)| と x [Å]

条件（test_scattering.py の 10 節と同じ）
  バリアの高さ 500 meV、バリア幅 20 Å、井戸幅（バリア間隔）50 Å、有効質量 m* = 0.1 m₀（全領域）

計算は scattering.py（eV と nm で計算する）に任せ、ここでは単位を meV と Å に換算して描くだけ。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import scattering as sc

plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]

# --- 条件 ---
MEV = 1e-3  # 1 meV = 10^-3 eV
ANGSTROM = 0.1  # 1 Å = 0.1 nm
HEIGHT = 500 * MEV
WIDTH = 20 * ANGSTROM
GAP = 50 * ANGSTROM
MASS_RATIO = 0.1

E_RANGE_MEV = (0.1, 700.0)  # 図1 の横軸の範囲。バリアより上の様子も見えるように 500 meV より先まで
X_RANGE_ANGSTROM = (-100.0, 190.0)  # 図2 の横軸の範囲（構造は 0〜90 Å）

OUTPUT_DIR = Path(__file__).parent / "figures"

COLOR_T = "#1baf7a"  # viewer.py の透過率と同じ緑系
COLOR_PSI = "#2a78d6"  # viewer.py の合成波と同じ青
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#8a8985"
COLOR_BARRIER = "#e9e8e4"


def transmission_curve(boundaries, potentials, resonances):
    """
    図1 用の (E [eV] の配列, T の配列)。共鳴ピークはとても細い（1つ目は幅 1 meV 未満）ので、
    全体を等間隔に区切った点に加えて、各共鳴の周り ±2 meV に細かい点を足してから並べ直す。
    """
    coarse = np.linspace(E_RANGE_MEV[0] * MEV, E_RANGE_MEV[1] * MEV, 4001)
    fine = [np.linspace(e - 2 * MEV, e + 2 * MEV, 2001) for e in resonances]
    energies = np.unique(np.concatenate([coarse, *fine]))  # unique は「重複を除いて小さい順に並べる」
    return energies, sc.spectrum(energies, boundaries, potentials, MASS_RATIO)[1]


def plot_transmission(boundaries, potentials, resonances, path):
    """図1: 透過率 T（縦軸）と E [meV]（横軸）。共鳴準位に点線と数値を添える"""
    energies, transmittances = transmission_curve(boundaries, potentials, resonances)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(energies / MEV, transmittances, color=COLOR_T, lw=2)
    ax.axvline(HEIGHT / MEV, color=COLOR_MUTED, lw=1, ls=":")
    ax.text(HEIGHT / MEV, 1.04, " バリアの高さ", color=COLOR_MUTED, fontsize=9, va="bottom")
    for n, e in enumerate(resonances, start=1):
        ax.axvline(e / MEV, color=COLOR_INK, lw=0.8, ls="--")
        ax.text(e / MEV, 0.5, f" E{n} = {e / MEV:.3f} meV", color=COLOR_INK, fontsize=10)
    ax.set_xlim(*E_RANGE_MEV)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("E [meV]")
    ax.set_ylabel("透過率 T")
    ax.set_title(
        f"2重障壁の透過率（V₀ = {HEIGHT / MEV:.0f} meV, 幅 {WIDTH / ANGSTROM:.0f} Å, "
        f"間隔 {GAP / ANGSTROM:.0f} Å, m* = {MASS_RATIO} m₀）",
        loc="left",
        fontsize=11,
    )
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_wavefunctions(boundaries, potentials, resonances, path):
    """
    図2: 各共鳴エネルギーでの |ψ(x)|（縦軸）と x [Å]（横軸）。準位ごとに1段。
    縦軸は入射波の振幅 A = 1 を基準にした値（散乱状態は規格化できず、比しか決まらないため）。
    """
    x_angstrom = np.linspace(*X_RANGE_ANGSTROM, 4000)
    fig, axes = plt.subplots(len(resonances), 1, figsize=(8, 3.2 * len(resonances)), sharex=True)
    for n, (ax, e) in enumerate(zip(np.atleast_1d(axes), resonances), start=1):
        result = sc.solve_scattering(e, boundaries, potentials, MASS_RATIO)
        psi = sc.wavefunction(x_angstrom * ANGSTROM, result)
        for left, right in zip(boundaries[::2], boundaries[1::2]):  # 境界は (山の左端, 右端) の組で並んでいる
            ax.axvspan(left / ANGSTROM, right / ANGSTROM, color=COLOR_BARRIER, zorder=0)
        ax.plot(x_angstrom, np.abs(psi), color=COLOR_PSI, lw=2)
        ax.set_ylabel("|ψ(x)|（A = 1 を基準）")
        ax.set_title(f"E{n} = {e / MEV:.3f} meV（T = {result.transmittance:.6f}）", loc="left", fontsize=11)
        ax.set_ylim(0, None)
        style(ax)
    np.atleast_1d(axes)[-1].set_xlabel("x [Å]（灰色の帯 = バリア）")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def style(ax):
    ax.grid(color="#e9e8e4", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    boundaries, potentials = sc.barrier_array(2, HEIGHT, WIDTH, GAP)
    # バリアより下（0 < E < V₀）の共鳴準位を探す
    resonances = sc.find_resonances(boundaries, potentials, 0.1 * MEV, HEIGHT, mass_ratio=MASS_RATIO)
    for n, e in enumerate(resonances, start=1):
        print(f"共鳴準位 E{n} = {e / MEV:.6f} meV")

    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_transmission(boundaries, potentials, resonances, OUTPUT_DIR / "double_barrier_transmission.png")
    plot_wavefunctions(boundaries, potentials, resonances, OUTPUT_DIR / "double_barrier_wavefunction.png")
    print(f"図を保存しました: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
