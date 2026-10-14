"""
plot_linear_system.py — 2重障壁で scattering.py が組み立てる連立方程式を、発表用の図にする

実行:  .venv/bin/python plot_linear_system.py
出力:  figures/linear_system_1_unknowns.png  … 図1 領域ごとの波動関数と未知数・境界条件の数え上げ
       figures/linear_system_2_matrix.png    … 図2 _continuity_equations() が作る 8×8 の行列と右辺
       figures/linear_system_3_flow.png      … 図3 solve_scattering() の処理の流れ
       figures/linear_system_4_rt.png        … 図4 係数から R, T を出し、E を動かして T(E) を描くまで
       figures/linear_system_5_wavefunction.png … 図5 係数から φ(x) を組み立てて |φ(x)| を描くまで

図は記号で描く（模式図）。障壁の中は E < V を想定し、k = iρ なので e^{ikx} = e^{-ρx} と書いている。
行列の各成分が本当にこの記号どおりかは、末尾の check_matrix() で scattering.py の数値と突き合わせる。
（ここで描いているのは scattering.py がやっている計算の説明図で、新しい物理は入れていない）
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import scattering as sc

# Mac では Hiragino、Linux では Noto を使う。数式部分は LaTeX 風の字体（cm）
plt.rcParams["font.family"] = ["Hiragino Sans", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "cm"

OUTPUT_DIR = Path(__file__).parent / "figures"
SIZE = (13.33, 7.5)  # 16:9 スライド1枚分 [inch]

COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#8a8985"
COLOR_BARRIER = "#e9e8e4"
COLOR_LINE = "#c9c8c3"
# 境界 x₀, x₁, x₂, x₃ ごとの色（行列の2行ずつの組と対応させる）
BOUNDARY_COLORS = ["#2a78d6", "#1baf7a", "#e8833a", "#8a5cd1"]

UNKNOWNS = ["B", "c_{10}", "c_{11}", "c_{20}", "c_{21}", "c_{30}", "c_{31}", "C"]
# 未知数がどの領域の係数か（列の上に書く）
UNKNOWN_REGION = [0, 1, 1, 2, 2, 3, 3, 4]

# _continuity_equations() が作る行列を記号で書いたもの（0 は空欄）
#   境界 n の2行 = 「φ の連続」「φ' の連続」。左の領域は +、右の領域は移項して −
#   右の領域は自分の原点 = 境界そのもので評価するので、値は ±1、傾きは ±ik または ±ρ だけになる
MATRIX = [
    ["1", "-1", "-1", 0, 0, 0, 0, 0],
    ["-ik", r"\rho", r"-\rho", 0, 0, 0, 0, 0],
    [0, r"e^{-\rho a}", r"e^{\rho a}", "-1", "-1", 0, 0, 0],
    [0, r"-\rho e^{-\rho a}", r"\rho e^{\rho a}", "-ik", "ik", 0, 0, 0],
    [0, 0, 0, r"e^{ikb}", r"e^{-ikb}", "-1", "-1", 0],
    [0, 0, 0, r"ik\,e^{ikb}", r"-ik\,e^{-ikb}", r"\rho", r"-\rho", 0],
    [0, 0, 0, 0, 0, r"e^{-\rho a}", r"e^{\rho a}", "-1"],
    [0, 0, 0, 0, 0, r"-\rho e^{-\rho a}", r"\rho e^{\rho a}", "-ik"],
]
RHS = ["-1", "-ik", "0", "0", "0", "0", "0", "0"]


# ---------------------------------------------------------------------------
# 図1: 領域・未知数・境界条件
# ---------------------------------------------------------------------------
def plot_unknowns(path):
    fig, ax = plt.subplots(figsize=SIZE)
    # 模式図なので幅は見やすさ優先（実寸ではない）
    edges = [0.0, 2.8, 5.6, 8.4]  # x₀ … x₃ の描画位置
    left_end, right_end = -3.6, 12.0
    height = 1.0
    for left, right in ((edges[0], edges[1]), (edges[2], edges[3])):
        ax.add_patch(Rectangle((left, 0), right - left, height, color=COLOR_BARRIER, zorder=0))
    ax.plot([left_end, 0, 0, edges[1], edges[1], edges[2], edges[2], edges[3], edges[3], right_end],
            [0, 0, height, height, 0, 0, height, height, 0, 0], color=COLOR_INK, lw=2)

    centers = [(left_end + 0) / 2, sum(edges[0:2]) / 2, sum(edges[1:3]) / 2, sum(edges[2:4]) / 2,
               (edges[3] + right_end) / 2]
    names = ["領域0（外）", "領域1（障壁）", "領域2（井戸）", "領域3（障壁）", "領域4（外）"]
    kinds = ["$k$", r"$k=i\rho$", "$k$", r"$k=i\rho$", "$k$"]
    forms = [
        r"$e^{ikx}+B\,e^{-ikx}$",
        r"$c_{10}e^{-\rho x'}$" + "\n" + r"$+\,c_{11}e^{\rho x'}$",
        r"$c_{20}e^{ikx'}$" + "\n" + r"$+\,c_{21}e^{-ikx'}$",
        r"$c_{30}e^{-\rho x'}$" + "\n" + r"$+\,c_{31}e^{\rho x'}$",
        r"$C\,e^{ikx'}$",
    ]
    notes = ["入射 $A=1$ に固定", "", "", "", "右から来る波 $D=0$"]
    for c, name, kind, form, note in zip(centers, names, kinds, forms, notes):
        ax.text(c, 1.55, name, ha="center", fontsize=13, color=COLOR_INK)
        ax.text(c, 1.25, kind, ha="center", fontsize=14, color=COLOR_MUTED)
        ax.text(c, -0.55, form, ha="center", va="top", fontsize=17, color=COLOR_INK, linespacing=1.5)
        if note:
            ax.text(c, -1.75, note, ha="center", va="top", fontsize=12, color=COLOR_MUTED)

    for n, (x, color) in enumerate(zip(edges, BOUNDARY_COLORS)):
        ax.plot([x, x], [-0.25, 2.25], color=color, lw=2, ls="--")
        ax.text(x, 2.35, f"$x_{n}$", ha="center", fontsize=15, color=color)
        ax.text(x, 2.75, r"$\varphi,\ \varphi'$ 連続" + "\n→ 2本", ha="center", fontsize=11,
                color=color, linespacing=1.3)

    # 幅 a, b の矢印
    for left, right, label in ((edges[0], edges[1], "a"), (edges[1], edges[2], "b"), (edges[2], edges[3], "a")):
        ax.add_patch(FancyArrowPatch((left, 0.5), (right, 0.5), arrowstyle="<->", mutation_scale=12,
                                     color=COLOR_MUTED, lw=1))
        ax.text((left + right) / 2, 0.58, f"${label}$", ha="center", fontsize=14, color=COLOR_MUTED)

    ax.text(left_end, -2.55,
            "未知数  2個 × 5領域 − 2（$A=1$, $D=0$） = 8個     "
            "方程式  2本 × 4境界 = 8本     → 8×8 の連立一次方程式",
            fontsize=14, color=COLOR_INK)
    ax.text(left_end, -3.05, "$x'$ は各領域の左端（領域0 は $x_0$）から測った位置。障壁の中は $E<V$ の場合。",
            fontsize=11, color=COLOR_MUTED)
    ax.set_xlim(left_end - 0.2, right_end + 0.2)
    ax.set_ylim(-3.3, 3.5)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 図2: 行列
# ---------------------------------------------------------------------------
def plot_matrix(path):
    fig, ax = plt.subplots(figsize=SIZE)
    w, h = 1.3, 0.72  # セル1つの幅と高さ
    x0, y_top = 2.2, 0.0

    def cell_xy(row, col, offset=0.0):
        return x0 + col * w + offset, y_top - (row + 1) * h

    # 行ラベルと行の色帯（境界ごとに2行）
    for row in range(8):
        n = row // 2
        color = BOUNDARY_COLORS[n]
        x, y = cell_xy(row, 0)
        ax.add_patch(Rectangle((x, y), 8 * w, h, color=color, alpha=0.07, lw=0))
        label = rf"$x_{n}$:  $\varphi$" if row % 2 == 0 else rf"$x_{n}$:  $\varphi'$"
        ax.text(x0 - 0.15, y + h / 2, label, ha="right", va="center", fontsize=14, color=color)

    # 成分
    for row, entries in enumerate(MATRIX):
        color = BOUNDARY_COLORS[row // 2]
        for col, entry in enumerate(entries):
            x, y = cell_xy(row, col)
            if entry == 0:
                ax.text(x + w / 2, y + h / 2, "0", ha="center", va="center", fontsize=12, color=COLOR_LINE)
                continue
            ax.add_patch(Rectangle((x + 0.04, y + 0.04), w - 0.08, h - 0.08, facecolor=color, alpha=0.18, lw=0))
            ax.text(x + w / 2, y + h / 2, f"${entry}$", ha="center", va="center", fontsize=15, color=COLOR_INK)

    # 列見出し（未知数と所属する領域）
    for col, (name, region) in enumerate(zip(UNKNOWNS, UNKNOWN_REGION)):
        x, _ = cell_xy(0, col)
        ax.text(x + w / 2, y_top + 0.18, f"${name}$", ha="center", va="bottom", fontsize=15, color=COLOR_INK)
        ax.text(x + w / 2, y_top + 0.62, f"領域{region}", ha="center", va="bottom", fontsize=10, color=COLOR_MUTED)
    # 行列の括弧
    left, right = x0 - 0.02, x0 + 8 * w + 0.02
    bottom = y_top - 8 * h
    for xs in ((left + 0.12, left, left, left + 0.12), (right - 0.12, right, right, right - 0.12)):
        ax.plot(xs, [y_top, y_top, bottom, bottom], color=COLOR_INK, lw=1.5)

    # 未知数ベクトル、=、右辺
    xv = right + 0.35
    for row, name in enumerate(UNKNOWNS):
        ax.text(xv + 0.4, y_top - (row + 0.5) * h, f"${name}$", ha="center", va="center", fontsize=15)
    for xs in ((xv + 0.1, xv, xv, xv + 0.1), (xv + 0.7, xv + 0.8, xv + 0.8, xv + 0.7)):
        ax.plot(xs, [y_top, y_top, bottom, bottom], color=COLOR_INK, lw=1.5)
    ax.text(xv + 1.15, y_top - 4 * h, "$=$", ha="center", va="center", fontsize=20)
    xr = xv + 1.5
    for row, entry in enumerate(RHS):
        ax.text(xr + 0.45, y_top - (row + 0.5) * h, f"${entry}$", ha="center", va="center", fontsize=15,
                color=COLOR_INK if entry != "0" else COLOR_LINE)
    for xs in ((xr + 0.1, xr, xr, xr + 0.1), (xr + 0.8, xr + 0.9, xr + 0.9, xr + 0.8)):
        ax.plot(xs, [y_top, y_top, bottom, bottom], color=COLOR_INK, lw=1.5)
    ax.text(x0 + 4 * w, y_top + 1.1, "matrix", ha="center", fontsize=13, color=COLOR_MUTED, family="monospace")
    ax.text(xv + 0.4, y_top + 1.1, "unknowns", ha="center", fontsize=13, color=COLOR_MUTED, family="monospace")
    ax.text(xr + 0.45, y_top + 1.1, "rhs", ha="center", fontsize=13, color=COLOR_MUTED, family="monospace")

    # 注記
    ax.text(x0 - 1.9, bottom - 0.45,
            "・境界 $x_n$ の2行には、隣り合う領域 $n$（＋）と $n+1$（−, 移項）の係数しか現れない → 帯状に並ぶ",
            fontsize=12.5, va="top")
    ax.text(x0 - 1.9, bottom - 0.95,
            "・右側の領域は自分の原点（＝その境界）で評価するので $-1$ と $\\mp ik$, $\\pm\\rho$ だけ。"
            "既知の $A=1$ の項は右辺 rhs へ",
            fontsize=12.5, va="top")
    ax.text(x0 - 1.9, bottom - 1.45, "・これを np.linalg.solve(matrix, rhs) に渡すと、未知数 8個がまとめて求まる",
            fontsize=12.5, va="top")
    ax.set_xlim(x0 - 2.0, xr + 1.2)
    ax.set_ylim(bottom - 1.9, y_top + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 図3: 処理の流れ
# ---------------------------------------------------------------------------
def plot_flow(path):
    fig, ax = plt.subplots(figsize=SIZE)
    steps = [
        ("① 形を決める", "barrier_array()",
         "境界  $x_0, x_1, x_2, x_3$\nポテンシャル\n$0,\\ V_0,\\ 0,\\ V_0,\\ 0$"),
        ("② 波数", "wave_number()  ×5",
         "$k,\\ i\\rho,\\ k,\\ i\\rho,\\ k$\n\n$E<V$ なら自動で\n純虚数になる"),
        ("③ 行列に書き込む", "_continuity_equations()\n_basis()",
         "境界 $x_n$ ごとに\n左(＋)・右(−)の領域の\n値と傾きを2行に書く\n\n$A$ の項 → rhs\n$D$ の項 → 書かない"),
        ("④ 解く", "np.linalg.solve()",
         "8×8 の matrix と rhs\n↓\n$B,\\ c_{10},\\ \\ldots,\\ c_{31},\\ C$"),
        ("⑤ R と T", "_reflectance()\n_transmittance()",
         "$R=|B|^2$\n$T=\\dfrac{k_4}{k_0}|C|^2$\n（両端は同じ $k$）\n\n$R+T=1$ で検算"),
    ]
    n = len(steps)
    box_w, box_h, gap = 2.4, 3.2, 0.4
    total = n * box_w + (n - 1) * gap
    x_start = -total / 2
    for i, (title, func, body) in enumerate(steps):
        x = x_start + i * (box_w + gap)
        color = BOUNDARY_COLORS[1] if i == 2 else COLOR_MUTED
        ax.add_patch(FancyBboxPatch((x, -box_h / 2), box_w, box_h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor="white", edgecolor=color, lw=2 if i == 2 else 1.2))
        ax.text(x + box_w / 2, box_h / 2 - 0.35, title, ha="center", va="top", fontsize=15, weight="bold")
        ax.text(x + box_w / 2, box_h / 2 - 0.95, func, ha="center", va="top", fontsize=9.5,
                family="monospace", color=COLOR_MUTED)
        ax.text(x + box_w / 2, box_h / 2 - 1.55, body, ha="center", va="top", fontsize=12.5, linespacing=1.55)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + box_w + 0.05, 0), (x + box_w + gap - 0.05, 0), arrowstyle="-|>",
                                         mutation_scale=18, color=COLOR_INK, lw=1.5))
    ax.text(x_start, -box_h / 2 - 0.45,
            "③のループ: 境界 4個 × 領域 2つ（左・右）× 基本解 2つ ＝ 16回、各回で「値」「傾き」の2成分を書き込む",
            fontsize=12.5, va="top")
    ax.text(x_start, box_h / 2 + 0.55, "solve_scattering(E, 境界, ポテンシャル) の中身（2重障壁の場合）",
            fontsize=16, va="bottom")
    ax.set_xlim(x_start - 0.2, -x_start + 0.2)
    ax.set_ylim(-box_h / 2 - 1.1, box_h / 2 + 1.1)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)



# ---------------------------------------------------------------------------
# 図4・図5 で共通: 左半分に処理の手順を縦に並べる
# ---------------------------------------------------------------------------
# 図4・図5 の計算条件（plot_double_barrier.py と同じ）
MEV = 1e-3  # 1 meV = 10^-3 eV
ANGSTROM = 0.1  # 1 Å = 0.1 nm
HEIGHT, WIDTH, GAP, MASS_RATIO = 500 * MEV, 20 * ANGSTROM, 50 * ANGSTROM, 0.1
E_OFF = 100 * MEV  # 共鳴していないエネルギーの例
COLOR_T = "#1baf7a"  # 透過率（viewer.py と同じ緑系）
COLOR_R = "#e8833a"  # 反射率
COLOR_PSI = "#2a78d6"  # 波動関数（viewer.py と同じ青）


def draw_steps(ax, steps, title):
    """
    ax（軸なし）に、手順の箱を上から順に並べる。
    steps は (見出し, 関数名, 本文) の並び。本文は数式（$...$）を含んでよい。
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0, 0.985, title, fontsize=15, va="top")
    top, bottom, gap = 0.9, 0.0, 0.025
    box_h = (top - bottom - gap * (len(steps) - 1)) / len(steps)
    for i, (head, func, body) in enumerate(steps):
        y = top - i * (box_h + gap) - box_h
        ax.add_patch(FancyBboxPatch((0.0, y), 1.0, box_h, boxstyle="round,pad=0,rounding_size=0.015",
                                    facecolor="white", edgecolor=COLOR_MUTED, lw=1.1))
        ax.text(0.03, y + box_h - 0.022, head, fontsize=13, weight="bold", va="top")
        ax.text(0.97, y + box_h - 0.03, func, fontsize=9, family="monospace", color=COLOR_MUTED,
                va="top", ha="right")
        ax.text(0.06, y + box_h - 0.075, body, fontsize=12, va="top", linespacing=1.45)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((0.5, y + 0.004), (0.5, y - gap + 0.004), arrowstyle="-|>",
                                         mutation_scale=11, color=COLOR_INK, lw=1.2))


def style(ax):
    ax.grid(color=COLOR_BARRIER, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# 図4: R, T と T(E)
# ---------------------------------------------------------------------------
def plot_rt(path):
    boundaries, potentials = sc.barrier_array(2, HEIGHT, WIDTH, GAP)
    resonances = sc.find_resonances(boundaries, potentials, 0.1 * MEV, HEIGHT, mass_ratio=MASS_RATIO)
    one = sc.solve_scattering(E_OFF, boundaries, potentials, MASS_RATIO)

    fig = plt.figure(figsize=SIZE)
    steps = [
        ("① 係数を取り出す", "solution.coefficients",
         "左端 $(A,\\ B)=(1,\\ B)$ → $B$\n右端 $(C,\\ D)=(C,\\ 0)$ → $C$"),
        ("② 流れの比をとる", "_reflectance() / _transmittance()",
         "流れ $\\propto\\ k\\,|$振幅$|^2$ なので\n"
         "$R=|B|^2,\\quad T=\\dfrac{k_4}{k_0}\\,|C|^2$"),
        ("③ 検算", "", "$R+T=1$（確率の流れの保存）"),
        ("④ E を変えて繰り返す", "spectrum(energies, ...)",
         "E ごとに ①〜③（連立方程式を解き直す）\n→ R, T の配列 → plt.plot(E, T)"),
    ]
    draw_steps(fig.add_axes([0.035, 0.06, 0.38, 0.9]), steps, "連立方程式の解 → R, T → T(E)")

    # 右: T(E) と R(E)。共鳴ピークは細いので、共鳴の周りだけ点を細かくする
    coarse = np.linspace(0.1 * MEV, 700 * MEV, 3001)
    fine = [np.linspace(e - 2 * MEV, e + 2 * MEV, 1001) for e in resonances]
    energies = np.unique(np.concatenate([coarse, *fine]))
    reflectances, transmittances = sc.spectrum(energies, boundaries, potentials, MASS_RATIO)

    ax = fig.add_axes([0.5, 0.13, 0.47, 0.72])
    ax.plot(energies / MEV, transmittances, color=COLOR_T, lw=2, label="透過率 $T$")
    ax.plot(energies / MEV, reflectances, color=COLOR_R, lw=1.2, alpha=0.8, label="反射率 $R$")
    ax.axvline(HEIGHT / MEV, color=COLOR_MUTED, lw=1, ls=":")
    ax.text(HEIGHT / MEV, 1.04, " 障壁の高さ $V_0$", color=COLOR_MUTED, fontsize=10)
    for n, e in enumerate(resonances, start=1):
        ax.text(e / MEV, 1.04, f"$E_{n}$", ha="center", fontsize=12)
    # ①〜③ を1回やった点
    ax.plot([E_OFF / MEV] * 2, [one.transmittance, one.reflectance], "o", color=COLOR_INK, ms=5, zorder=5)
    ax.annotate(f"①〜③を1回\nE = {E_OFF / MEV:.0f} meV\nR = {one.reflectance:.4f}\nT = {one.transmittance:.4f}",
                xy=(E_OFF / MEV, one.reflectance), xytext=(135, 0.42), fontsize=11,
                arrowprops=dict(arrowstyle="-", color=COLOR_MUTED, lw=0.8))
    ax.set_xlim(0, 700)
    ax.set_ylim(-0.02, 1.1)
    ax.set_xlabel("入射エネルギー E [meV]", fontsize=12)
    ax.set_ylabel("R, T", fontsize=12)
    ax.legend(loc="center", bbox_to_anchor=(0.66, 0.5), frameon=False, fontsize=11)
    ax.set_title(f"2重障壁（$V_0$ = {HEIGHT / MEV:.0f} meV, 幅 {WIDTH / ANGSTROM:.0f} Å, "
                 f"間隔 {GAP / ANGSTROM:.0f} Å, $m^*$ = {MASS_RATIO} $m_0$）", loc="left", fontsize=11, pad=22)
    style(ax)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 図5: φ(x) の組み立てと |φ(x)|
# ---------------------------------------------------------------------------
def plot_wavefunction_steps(path):
    boundaries, potentials = sc.barrier_array(2, HEIGHT, WIDTH, GAP)
    resonances = sc.find_resonances(boundaries, potentials, 0.1 * MEV, HEIGHT, mass_ratio=MASS_RATIO)
    x_angstrom = np.linspace(-100, 190, 4000)
    x = x_angstrom * ANGSTROM

    fig = plt.figure(figsize=SIZE)
    steps = [
        ("① 位置を細かく並べる", "np.linspace()", "$x$ を 4000 点（描きたい範囲）"),
        ("② どの領域か調べる", "np.searchsorted(boundaries, x)",
         "各点に領域番号 $j$ = 0, 1, …, 4 を付ける\n（右上の帯）"),
        ("③ 領域の原点から測る", "x - origins[j]",
         "$x' = x\\,-$（領域 $j$ の左端）\n連立方程式を立てたときと同じ原点"),
        ("④ 組み立てる", "wavefunction() → _basis()",
         "$\\varphi(x)=c_{j0}\\,f_1(x')+c_{j1}\\,f_2(x')$\n$f_1, f_2 = e^{\\pm ikx'}$ または $e^{\\mp\\rho x'}$"),
        ("⑤ 描く", "plt.plot(x, abs(phi))",
         "$\\varphi$ は複素数なので $|\\varphi(x)|$ を描く\n（$A=1$ を基準にした大きさ）"),
    ]
    draw_steps(fig.add_axes([0.035, 0.06, 0.38, 0.9]), steps, "連立方程式の解 → φ(x) → グラフ")

    # 右上: searchsorted が付ける領域番号の帯
    region = np.searchsorted(np.asarray(boundaries), x, side="right")
    strip = fig.add_axes([0.5, 0.83, 0.47, 0.06])
    edges = [x_angstrom[0], *(b / ANGSTROM for b in boundaries), x_angstrom[-1]]
    for j, (left, right) in enumerate(zip(edges, edges[1:])):
        barrier = potentials[j] > 0
        strip.add_patch(Rectangle((left, 0), right - left, 1, facecolor="#d6d5d0" if barrier else "white",
                                  edgecolor=COLOR_MUTED, lw=0.8))
        strip.text((left + right) / 2, 0.5, f"{j}", ha="center", va="center", fontsize=12)
    assert list(np.unique(region)) == [0, 1, 2, 3, 4]
    strip.set_xlim(x_angstrom[0], x_angstrom[-1])
    strip.set_ylim(0, 1)
    strip.axis("off")
    strip.text(x_angstrom[0], 1.25, "② で付く領域番号 j（灰色 = 障壁）", fontsize=10, color=COLOR_MUTED)

    # 右下: 共鳴と非共鳴の |φ(x)|
    cases = [(resonances[0], "共鳴"), (E_OFF, "非共鳴")]
    for i, (energy, label) in enumerate(cases):
        ax = fig.add_axes([0.5, 0.5 - i * 0.4, 0.47, 0.27])
        result = sc.solve_scattering(energy, boundaries, potentials, MASS_RATIO)
        psi = sc.wavefunction(x, result)
        for left, right in zip(boundaries[::2], boundaries[1::2]):
            ax.axvspan(left / ANGSTROM, right / ANGSTROM, color=COLOR_BARRIER, zorder=0)
        ax.plot(x_angstrom, np.abs(psi), color=COLOR_PSI, lw=2)
        ax.set_xlim(x_angstrom[0], x_angstrom[-1])
        ax.set_ylim(0, None)
        ax.set_ylabel("$|\\varphi(x)|$", fontsize=12)
        ax.set_title(f"{label}: E = {energy / MEV:.1f} meV（T = {result.transmittance:.4f}）",
                     loc="left", fontsize=11)
        style(ax)
        if i == 1:
            ax.set_xlabel("x [Å]", fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------------------
# 記号の行列が scattering.py の数値と一致するかの確認
# ---------------------------------------------------------------------------
def check_matrix(energy=0.1, height=0.5, width=2.0, gap=5.0, mass_ratio=0.1):
    """
    図2 の記号の行列に数値を入れたものと、_continuity_equations() が実際に作る行列を比べる。
    一致しなければ図が間違っているので AssertionError を出す。
    条件は plot_double_barrier.py と同じ（V₀ = 0.5 eV, a = 2 nm, b = 5 nm, m* = 0.1 m₀）、E = 0.1 eV < V₀。
    """
    boundaries, potentials = sc.barrier_array(2, height, width, gap)
    ks = tuple(sc.wave_number(energy, v, mass_ratio) for v in potentials)
    matrix, rhs = sc._continuity_equations(boundaries, ks)

    # 図2 の MATRIX, RHS と同じ並びで、記号を数値に置き換えて手で書いたもの
    k, rho, a, b = ks[0].real, ks[1].imag, width, gap
    ep, em = np.exp(rho * a), np.exp(-rho * a)  # e^{ρa}, e^{-ρa}
    wp, wm = np.exp(1j * k * b), np.exp(-1j * k * b)  # e^{ikb}, e^{-ikb}
    ik = 1j * k
    expected = np.array([
        [1, -1, -1, 0, 0, 0, 0, 0],
        [-ik, rho, -rho, 0, 0, 0, 0, 0],
        [0, em, ep, -1, -1, 0, 0, 0],
        [0, -rho * em, rho * ep, -ik, ik, 0, 0, 0],
        [0, 0, 0, wp, wm, -1, -1, 0],
        [0, 0, 0, ik * wp, -ik * wm, rho, -rho, 0],
        [0, 0, 0, 0, 0, em, ep, -1],
        [0, 0, 0, 0, 0, -rho * em, rho * ep, -ik],
    ], dtype=complex)
    expected_rhs = np.array([-1, -ik, 0, 0, 0, 0, 0, 0], dtype=complex)

    error = max(np.max(np.abs(expected - matrix)), np.max(np.abs(expected_rhs - rhs)))
    assert error < 1e-9, f"図の行列と scattering.py の行列が一致しません（最大差 {error}）"
    print(f"図の記号の行列と scattering.py の行列は一致（最大差 {error:.1e}）")


def main():
    check_matrix()
    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_unknowns(OUTPUT_DIR / "linear_system_1_unknowns.png")
    plot_matrix(OUTPUT_DIR / "linear_system_2_matrix.png")
    plot_flow(OUTPUT_DIR / "linear_system_3_flow.png")
    plot_rt(OUTPUT_DIR / "linear_system_4_rt.png")
    plot_wavefunction_steps(OUTPUT_DIR / "linear_system_5_wavefunction.png")
    print(f"図を保存しました: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
