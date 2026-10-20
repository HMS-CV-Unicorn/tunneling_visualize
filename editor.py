"""
editor.py — 高さ・幅・間隔がバラバラな障壁を、マウスで直接動かして反射と透過を見る

実行:  .venv/bin/python editor.py

viewer.py（同じ形の障壁をスライダーで並べる版）の発展版。viewer.py はそのまま残してあり、どちらも使える。

画面の構成
  左上: ポテンシャル V(x)。障壁をマウスでつかんで形を変える操作面
  左中: 波動関数（チェックボックスで表示するものを選ぶ。「再生」で時間因子 e^{-iEt/ħ} を回す）
  左下: 透過率 T(E) と反射率 R(E)。いまの E の位置に印
  右側: 選んだ障壁の数値入力、障壁の追加・削除、ひな形（等間隔に並べる）、表示の切り替え

障壁の操作（上段の図）
  - カーソルを障壁に合わせると、つかめる場所が太線で強調され、その障壁の値が吹き出しに出る
      上の辺   → 上下にドラッグで高さ（0 より下に引くと井戸になる）
      左右の辺 → 左右にドラッグで、その辺だけを動かす（幅が変わる）
      中ほど   → 左右にドラッグで、障壁ごと移動（両隣との間隔が変わる）
    値は 5 meV・0.5 Å の刻みに吸着する。隣の障壁とぶつかる手前・幅が 0 になる手前で止まる
  - 障壁をクリックすると「選択中」になり（色が付く）、右上の欄で高さ・幅・左の間隔を数値で入れられる。
    数値で幅や間隔を変えたときは、右側の障壁がまとめてずれる（ほかの幅と間隔は変わらない）
  - 何もない所をクリックすると選択を外す

そのほかの操作は viewer.py と同じ
  - E と有効質量はスライダーか入力欄。T(E) の図をクリックするとその E に移動
  - ツールバーの虫眼鏡・十字で拡大・移動（そのモード中は障壁をつかめない。もう一度押してモードを解除する）
  - 「→ meV」「→ Å」ボタンで表示の単位を切り替える（計算は常に eV・nm）

T(E) の図の表示（右下のチェックボックス）
  - 透過率 T・反射率 R は、それぞれ表示/非表示を切り替えられる
  - 半値幅: いまの E に一番近い T の山について、頂上の半分の高さでの幅 ΔE（半値全幅）を矢印と数値で示す。
    頂上が T = 1 の共鳴なら、T = 0.5 になる2点の間の幅。山の裾が表示範囲の外なら、範囲を広げるよう表示する。
    E をクリックで山の近くに動かせば、その山の幅が出る。計算は scattering.half_maximum_width

計算は scattering.py、障壁の並びの作り変え（ドラッグ・数値入力・追加・削除）は barrier_model.py に任せ、
このファイルは「マウスや入力欄の操作を受け取る → 新しい並びを作ってもらう → 計算を呼ぶ → 描き直す」だけを担当する。
色・単位換算・入力欄の読み取りなど viewer.py と共通の部品は、viewer.py から読み込んで使う。

使っているライブラリ
  - matplotlib.widgets       : スライダー・入力欄・ボタン
  - matplotlib.patches       : 障壁を塗りつぶす長方形（Rectangle）
  - matplotlib.backend_tools : マウスカーソルの形（Cursors）
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_tools import Cursors
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox

import barrier_model as bm
import scattering as sc
from viewer import (
    COLOR_BARRIER,
    COLOR_DENSITY,
    COLOR_ERROR,
    COLOR_INK,
    COLOR_LEFT,
    COLOR_MUTED,
    COLOR_RIGHT,
    COLOR_TOTAL,
    ENERGY_FLOOR,
    ENERGY_VIEW,
    FRAME_INTERVAL_MS,
    FRAMES_PER_PERIOD,
    INITIAL_UNITS,
    LOG_SCALE,
    MARGIN,
    N_POINTS,
    SHOW_COMPONENTS,
    SHOW_DENSITY,
    SHOW_TOTAL,
    SLIDER_SPECS,
    UNIT_SCALES,
    barrier_spans,
    format_parameter,
    log_lower_limit,
    other_unit,
    padded_limits,
    parameter_label,
    parse_parameter,
    potential_outline,
    reflected_amplitude_label,
    set_slider_range,
    set_tick_unit,
    show_text_silently,
    spectrum_energies,
    transmitted_amplitude_label,
    unit_button_label,
    wave_axis_label,
)

COLOR_SELECTED = "#cde2fb"  # 選択中の障壁の塗り（薄い青）
COLOR_HOVER = "#2a78d6"  # カーソルを合わせた辺の強調（青）

HIT_TOLERANCE_PX = 7  # 辺から何ピクセル以内なら「辺をつかんだ」とみなすか

# 最初に表示する構造: 2重障壁（高さ 0.5 eV、幅 0.5 nm、間隔 1 nm）
INITIAL_BARRIERS = bm.uniform(2, 0.5, 0.5, 1.0)
INITIAL_ENERGY = 0.3  # [eV]

# つかんだ場所ごとのカーソルの形と、吹き出しに出す操作の説明
PART_CURSORS = {
    bm.PART_TOP: Cursors.RESIZE_VERTICAL,
    bm.PART_LEFT: Cursors.RESIZE_HORIZONTAL,
    bm.PART_RIGHT: Cursors.RESIZE_HORIZONTAL,
    bm.PART_BODY: Cursors.MOVE,
}
PART_HINTS = {
    bm.PART_TOP: "上下にドラッグ → 高さ",
    bm.PART_LEFT: "左右にドラッグ → 左端（幅）",
    bm.PART_RIGHT: "左右にドラッグ → 右端（幅）",
    bm.PART_BODY: "左右にドラッグ → 移動（間隔）／クリック → 数値入力",
}

# 右上の数値入力欄（選択中の障壁）と、ひな形の入力欄。キーは viewer.py の SLIDER_SPECS と同じ
# （同じキーにすることで、viewer.py の parse_parameter・format_parameter で読み書きと単位換算ができる）
EDIT_KEYS = ("height", "width", "gap")
TEMPLATE_KEYS = ("count", "height", "width", "gap")
TEMPLATE_INITIAL = {"count": 2, "height": 0.5, "width": 0.5, "gap": 1.0}  # [個, eV, nm, nm]
# T(E) の図に表示するものを選ぶチェックボックスの項目
SHOW_T = "透過率 T"
SHOW_R = "反射率 R"
SHOW_WIDTH = "半値幅（いまの E に近い山）"

# 1つの障壁の値の呼び名。障壁ごとに値が違うので、ひな形の V₀・a・b とは別の名前にする
EDIT_NAMES = {"height": "高さ V", "width": "幅", "gap": "左の間隔"}


# ---------------------------------------------------------------------------
# 描画に使う値を作る（副作用のない関数）
# ---------------------------------------------------------------------------
def structure_x_range(barriers) -> tuple[float, float]:
    """描画する x の範囲。障壁のある区間の左右に MARGIN の余白をとる（障壁がなければ 0 の前後）"""
    if not barriers:
        return -MARGIN, MARGIN
    return barriers[0].left - MARGIN, barriers[-1].right + MARGIN


def edit_label(key: str, units: dict) -> str:
    """1つの障壁の値のラベル（例: 左の間隔 [Å]）。単位は viewer.py の単位切り替えと同じものを付ける"""
    unit = parameter_label(key, units).rsplit("[", 1)[1]  # 「間隔 b [Å]」の「Å]」の部分を取り出す
    return f"{EDIT_NAMES[key]} [{unit}"


def barrier_values(barriers, index: int) -> dict:
    """index 番目の障壁の {高さ, 幅, 左の間隔}（入力欄の key と同じ名前。左端の障壁の間隔は None）"""
    b = barriers[index]
    return {"height": b.height, "width": b.width, "gap": bm.gap_before(barriers, index)}


def describe_barrier(barriers, index: int, units: dict) -> str:
    """吹き出し用の「2 番目: 高さ 500 meV, 幅 5 Å, 左の間隔 10 Å」という文字列"""
    values = barrier_values(barriers, index)
    parts = [
        f"{edit_label(key, units)} = {format_parameter(key, value, units)}"
        for key, value in values.items()
        if value is not None
    ]
    return f"{index + 1} 番目の障壁\n" + "\n".join(parts)


def nearest_peak_index(energies, transmittances, energy: float) -> int | None:
    """
    T(E) を計算した点の列の中で、山（両隣以上の点。両端は除く）のうち energy に一番近いものの番号。
    山がなければ None。T が一定（障壁なしで T = 1）なら山とは呼べないので None。
    """
    t = np.asarray(transmittances)
    if np.ptp(t) == 0:
        return None
    is_peak = (t[1:-1] >= t[:-2]) & (t[1:-1] >= t[2:]) & ((t[1:-1] > t[:-2]) | (t[1:-1] > t[2:]))
    candidates = np.flatnonzero(is_peak) + 1  # 両端を除いた配列で探したので、番号を1つずらす
    if candidates.size == 0:
        return None
    return int(candidates[np.argmin(np.abs(np.asarray(energies)[candidates] - energy))])


def peak_width_text(peak: sc.PeakWidth | None, units: dict) -> str:
    """右側に出す半値幅の説明。単位は表示の単位（eV か meV）"""
    if peak is None:
        return "半値幅: 表示範囲に T の山がありません"
    unit = units["energy"]
    scale = UNIT_SCALES["energy"][unit]
    head = f"山の頂上 E = {peak.e_peak * scale:.6g} {unit}（T = {peak.t_peak:.4g}）\n"
    if peak.width is None:
        side = "左" if peak.e_left is None else "右"
        return head + f"半値幅: 山の{side}の裾が表示範囲の外です（T(E) の図を広げてください）"
    return head + f"半値幅 ΔE = {peak.width * scale:.4g} {unit}（T = {peak.t_peak / 2:.4g} での幅）"


def edge_segment(barrier: bm.Barrier, part: str):
    """カーソルを合わせた場所を強調する線の頂点 (xs, vs)。中ほどなら長方形の外周"""
    left, right, height = barrier.left, barrier.right, barrier.height
    if part == bm.PART_TOP:
        return [left, right], [height, height]
    if part == bm.PART_LEFT:
        return [left, left], [0.0, height]
    if part == bm.PART_RIGHT:
        return [right, right], [0.0, height]
    return [left, left, right, right, left], [0.0, height, height, 0.0, 0.0]


# ---------------------------------------------------------------------------
# 画面を作る
# ---------------------------------------------------------------------------
def build_editor():
    """
    図と操作部品を作り、マウス・入力欄の操作で描き直すようにつなぐ。

    戻り値の widgets は必ず変数に保持しておくこと。
    （保持しないと Python が部品を「使われていない」とみなして消してしまい、操作が効かなくなる）
    """
    fig = plt.figure(figsize=(13, 10))
    fig.canvas.manager.set_window_title("反射と透過 — 障壁エディタ")

    # 左の列に図を3つ、右の列に操作部品を並べる。上2つの図は x 軸を共有（sharex）して位置を揃える
    ax_pot = fig.add_axes((0.07, 0.70, 0.62, 0.25))
    ax_wave = fig.add_axes((0.07, 0.40, 0.62, 0.23), sharex=ax_pot)
    ax_spec = fig.add_axes((0.07, 0.135, 0.62, 0.19))

    # --- 1. ポテンシャル（操作面） ---
    (line_pot,) = ax_pot.plot([], [], color=COLOR_INK, lw=2, zorder=3)
    (line_energy,) = ax_pot.plot([], [], color=COLOR_INK, lw=1.2, ls="--", zorder=2)
    (line_hover,) = ax_pot.plot([], [], color=COLOR_HOVER, lw=5, alpha=0.6, zorder=4, solid_capstyle="round")
    label_energy = ax_pot.text(0, 0, "", color=COLOR_INK, va="bottom", ha="left", fontsize=10)
    tooltip = ax_pot.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=9,
        multialignment="left",  # 吹き出しを右寄せで出しても、行の頭はそろえる
        zorder=10,
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": COLOR_MUTED, "alpha": 0.95},
    )
    tooltip.set_visible(False)
    ax_pot.tick_params(labelbottom=False)
    ax_pot.set_title("ポテンシャル V(x) と電子のエネルギー E — 障壁の辺や中をドラッグ、クリックで選択", loc="left", fontsize=11)
    filled = []  # 障壁の塗りつぶし（毎回作り直すので、消すために覚えておく）

    # --- 2. 波動関数（viewer.py と同じ） ---
    (line_right,) = ax_wave.plot([], [], color=COLOR_RIGHT, lw=1.6, ls="--", label="右向き成分（入射・透過）")
    (line_left,) = ax_wave.plot([], [], color=COLOR_LEFT, lw=1.6, ls="--", label="左向き成分（反射）")
    (line_total,) = ax_wave.plot([], [], color=COLOR_TOTAL, lw=2.2, label=r"$\mathrm{Re}\,\psi$（合成波）")
    (line_density,) = ax_wave.plot([], [], color=COLOR_DENSITY, lw=2, label=r"$|\psi|^2 = |\phi|^2$")
    (guide_reflected,) = ax_wave.plot([], [], color=COLOR_LEFT, lw=1, ls=":")
    (guide_transmitted,) = ax_wave.plot([], [], color=COLOR_RIGHT, lw=1, ls=":")
    text_reflected = ax_wave.text(0, 0, "", color=COLOR_INK, fontsize=9, va="bottom", ha="left")
    text_transmitted = ax_wave.text(0, 0, "", color=COLOR_INK, fontsize=9, va="bottom", ha="right")
    time_label = ax_wave.text(0.01, 0.97, "", transform=ax_wave.transAxes, va="top", fontsize=10)
    ax_wave.axhline(0, color=COLOR_MUTED, lw=0.6)
    ax_wave.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, fontsize=8, frameon=False)
    shaded = []

    # --- 3. 透過率・反射率のスペクトル（viewer.py と同じ） ---
    (line_t,) = ax_spec.plot([], [], color=COLOR_RIGHT, lw=2.2, label="透過率 T")
    (line_r,) = ax_spec.plot([], [], color=COLOR_LEFT, lw=2, label="反射率 R")
    (dot_t,) = ax_spec.plot([], [], "o", color=COLOR_RIGHT, ms=8, mec="white", mew=2)
    (dot_r,) = ax_spec.plot([], [], "o", color=COLOR_LEFT, ms=8, mec="white", mew=2)
    marker_energy = ax_spec.axvline(0, color=COLOR_INK, lw=1, ls="--")
    # 半値幅: 頂上の半分の高さに、左右の E を結ぶ両矢印と ΔE の文字を出す
    width_arrow = ax_spec.annotate(
        "", xy=(0, 0), xytext=(0, 0), arrowprops={"arrowstyle": "<->", "color": COLOR_INK, "lw": 1.2}, zorder=5
    )
    # ΔE の文字は矢印の右端のすぐ右に出す（細い山でも文字が矢印や軸に重ならないように）
    width_label = ax_spec.annotate(
        "", xy=(0, 0), xytext=(6, 0), textcoords="offset points", color=COLOR_INK, fontsize=9, ha="left", va="center", zorder=5
    )
    (width_top,) = ax_spec.plot([], [], "v", color=COLOR_INK, ms=6, zorder=5)  # 山の頂上の印
    ax_spec.set_xlim(*ENERGY_VIEW)
    ax_spec.set_ylim(-0.03, 1.03)
    ax_spec.set_ylabel("確率")
    ax_spec.set_title("T(E), R(E)（破線 = いまの E、クリックで E を移動）", loc="left", fontsize=11)

    for ax in (ax_pot, ax_wave, ax_spec):
        ax.grid(color=COLOR_BARRIER, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    error_text = fig.text(0.97, 0.99, "", color=COLOR_ERROR, fontsize=10, ha="right", va="top")

    # --- 右の列: 選択中の障壁の数値入力 ---
    PANEL_X = 0.80  # 入力欄の左端（ラベルはその左に出る）
    fig.text(0.735, 0.955, "選択中の障壁", fontsize=11, weight="bold")
    selection_text = fig.text(0.735, 0.93, "", fontsize=9, color=COLOR_MUTED, va="top")
    edit_boxes = {}
    for row, key in enumerate(EDIT_KEYS):
        ax_box = fig.add_axes((PANEL_X + 0.06, 0.865 - 0.035 * row, 0.10, 0.028))
        edit_boxes[key] = TextBox(ax_box, edit_label(key, INITIAL_UNITS) + " ", initial="")
        edit_boxes[key].label.set_fontsize(9)
    add_button = Button(fig.add_axes((0.735, 0.75, 0.11, 0.032)), "障壁を追加（右端）")
    delete_button = Button(fig.add_axes((0.855, 0.75, 0.105, 0.032)), "選択中を削除")
    for button in (add_button, delete_button):
        button.label.set_fontsize(9)

    # --- 右の列: ひな形（等間隔に並べる） ---
    fig.text(0.735, 0.715, "同じ障壁を等間隔に並べる", fontsize=11, weight="bold")
    template_boxes = {}
    for row, key in enumerate(TEMPLATE_KEYS):
        ax_box = fig.add_axes((PANEL_X + 0.06, 0.665 - 0.035 * row, 0.10, 0.028))
        template_boxes[key] = TextBox(
            ax_box,
            parameter_label(key, INITIAL_UNITS) + " ",
            initial=format_parameter(key, TEMPLATE_INITIAL[key], INITIAL_UNITS),
        )
        template_boxes[key].label.set_fontsize(9)
    template_button = Button(fig.add_axes((0.735, 0.51, 0.225, 0.032)), "この形に並べ直す")
    template_button.label.set_fontsize(9)

    # --- 右の列: 表示の切り替え・再生・数値表示 ---
    check = CheckButtons(
        fig.add_axes((0.735, 0.39, 0.225, 0.09)),
        (SHOW_TOTAL, SHOW_COMPONENTS, SHOW_DENSITY),
        actives=(True, True, False),
    )
    play_button = Button(fig.add_axes((0.735, 0.345, 0.105, 0.032)), "再生")
    reset_button = Button(fig.add_axes((0.855, 0.345, 0.105, 0.032)), "t = 0")
    readout = fig.text(0.735, 0.33, "", fontsize=10, va="top")
    width_text = fig.text(0.735, 0.255, "", fontsize=9, va="top")
    spec_check = CheckButtons(
        fig.add_axes((0.735, 0.14, 0.225, 0.07)), (SHOW_T, SHOW_R, SHOW_WIDTH), actives=(True, True, True)
    )
    log_check = CheckButtons(fig.add_axes((0.735, 0.10, 0.225, 0.035)), (LOG_SCALE,), actives=(False,))
    for spine in log_check.ax.spines.values():
        spine.set_visible(False)

    # --- 下: E と有効質量のスライダー、単位の切り替え ---
    sliders, slider_boxes = {}, {}
    for row, key in enumerate(("energy", "mass")):
        _label, vmin, vmax, _vinit, vstep = SLIDER_SPECS[key]
        y = 0.055 - 0.03 * row
        vinit = INITIAL_ENERGY if key == "energy" else 1.0
        sliders[key] = Slider(
            fig.add_axes((0.17, y, 0.45, 0.02)), parameter_label(key, INITIAL_UNITS), vmin, vmax, valinit=vinit, valstep=vstep
        )
        sliders[key].valtext.set_visible(False)  # 値はスライダー横の文字ではなく入力欄に出す
        slider_boxes[key] = TextBox(
            fig.add_axes((0.635, y - 0.003, 0.07, 0.026)), "", initial=format_parameter(key, vinit, INITIAL_UNITS)
        )
    unit_buttons = {
        "energy": Button(fig.add_axes((0.735, 0.052, 0.105, 0.028)), unit_button_label("energy", INITIAL_UNITS)),
        "length": Button(fig.add_axes((0.855, 0.052, 0.105, 0.028)), unit_button_label("length", INITIAL_UNITS)),
    }
    for button in unit_buttons.values():
        button.label.set_fontsize(9)
    fig.text(0.735, 0.03, "単位: E・V（左）、長さ（右）", fontsize=9, color=COLOR_MUTED, va="top")

    # 画面の状態
    #   barriers : いまの障壁の並び（barrier_model.Barrier の tuple）
    #   selected : 選択中の障壁の番号（なければ None）
    #   hover    : カーソルが指している場所（barrier_model.Hit、なければ None）
    #   drag     : ドラッグ中なら {"hit", "origin"（始めたときの並び）, "x0", "v0"（始めた点）}、そうでなければ None
    #   cursor   : いまのカーソルの形（同じ形を何度も設定しないため）
    #   spectrum : いま描いている T(E), R(E) の点 (energies, reflectances, transmittances)
    #   peak     : いまの E に一番近い T の山の半値幅（scattering.PeakWidth、山がなければ None）
    #   そのほかは viewer.py と同じ
    state = {
        "barriers": INITIAL_BARRIERS,
        "selected": None,
        "hover": None,
        "drag": None,
        "cursor": Cursors.POINTER,
        "phase": 0.0,
        "x": None,
        "phi": None,
        "right": None,
        "left": None,
        "energy": None,
        "x_range": None,
        "playing": False,
        "guide_reflected": False,
        "guide_transmitted": False,
        "log": False,
        "units": dict(INITIAL_UNITS),
        "spectrum": None,
        "peak": None,
    }

    def shown():
        """チェックボックスで選ばれている項目の集合"""
        return {label.get_text() for label, on in zip(check.labels, check.get_status()) if on}

    def show_error(message: str) -> None:
        """右上にエラーを出す（空文字なら消す）"""
        error_text.set_text(f"入力エラー: {message}" if message else "")

    # -----------------------------------------------------------------------
    # 描き直し
    # -----------------------------------------------------------------------
    def redraw_structure():
        """上段: 障壁の輪郭と塗りつぶし（選択中は青）、カーソルを合わせた場所の強調"""
        barriers = state["barriers"]
        boundaries, potentials = bm.to_potential(barriers)
        x_min, x_max = structure_x_range(barriers)
        # 輪郭の左右の端は、拡大・移動しても切れないよう十分遠くまで伸ばす
        xs, vs = potential_outline(boundaries, potentials, x_min - 100, x_max + 100)
        line_pot.set_data(xs, vs)

        for patch in filled:
            patch.remove()
        filled.clear()
        for i, b in enumerate(barriers):
            color = COLOR_SELECTED if i == state["selected"] else COLOR_BARRIER
            filled.append(ax_pot.add_patch(Rectangle((b.left, 0.0), b.width, b.height, color=color, zorder=1)))

        hover = state["drag"]["hit"] if state["drag"] else state["hover"]
        if hover is not None and hover.index < len(barriers):
            line_hover.set_data(*edge_segment(barriers[hover.index], hover.part))
            line_hover.set_visible(True)
        else:
            line_hover.set_visible(False)

    def fit_axes():
        """
        上2つの図の表示範囲を、いまの構造に合わせる。ドラッグ中は呼ばない
        （ドラッグ中に範囲が変わると、マウスの下の位置がずれて、つかんだ辺が逃げてしまうため）。
        """
        barriers = state["barriers"]
        x_range = structure_x_range(barriers)
        if state["x_range"] != x_range:
            ax_pot.set_xlim(*x_range)
            state["x_range"] = x_range
        heights = [b.height for b in barriers]
        low, high = padded_limits([*heights, state["energy"], 0.0], minimum_span=1.0)
        ax_pot.set_ylim(low, high + 0.15 * (high - low))  # 上に E のラベルが入る余白をとる

    def apply_visibility():
        """チェックボックスの状態に合わせて線の表示/非表示を切り替え、縦軸の範囲を決め直す（viewer.py と同じ）"""
        selected = shown()
        show_components = SHOW_COMPONENTS in selected
        line_total.set_visible(SHOW_TOTAL in selected)
        line_density.set_visible(SHOW_DENSITY in selected)
        for artist in (line_right, line_left):
            artist.set_visible(show_components)
        for artist in (guide_reflected, text_reflected):
            artist.set_visible(show_components and state["guide_reflected"])
        for artist in (guide_transmitted, text_transmitted):
            artist.set_visible(show_components and state["guide_transmitted"])

        values = [0.0, 1.0]
        if SHOW_TOTAL in selected:
            amplitude = np.abs(state["phi"])
            values += [amplitude.max(), -amplitude.max()]
        if show_components:
            amplitude = np.nanmax(np.abs([state["right"], state["left"]]), initial=0.0)
            values += [amplitude, -amplitude]
        if SHOW_DENSITY in selected:
            values.append((np.abs(state["phi"]) ** 2).max())
        ax_wave.set_ylim(*padded_limits(values, minimum_span=2.5))
        ax_wave.set_ylabel(
            wave_axis_label(show_wave=bool(selected & {SHOW_TOTAL, SHOW_COMPONENTS}), show_density=SHOW_DENSITY in selected)
        )

    def redraw_real_part():
        """いまの位相で Re ψ と、その右向き・左向き成分を描き直す（viewer.py と同じ）"""
        energy = state["energy"]
        t = state["phase"] * sc.HBAR_EV_FS / energy  # 位相 Et/ħ から時刻 t [fs] に戻す
        line_total.set_data(state["x"], sc.time_evolve(state["phi"], energy, t).real)
        line_right.set_data(state["x"], sc.time_evolve(state["right"], energy, t).real)
        line_left.set_data(state["x"], sc.time_evolve(state["left"], energy, t).real)
        time_label.set_text(f"t = {t:.3f} fs（1周期 2πħ/E = {sc.period(energy):.3f} fs）")

    def redraw_amplitude_guides(result, boundaries, x_min, x_max):
        """左端の領域に反射波の振幅 |B|、右端の領域に透過波の振幅 |C| の目安線を引く（viewer.py と同じ）"""
        state["guide_reflected"] = bool(boundaries)
        state["guide_transmitted"] = bool(boundaries) and result.transmittance > 0
        if not boundaries:
            return
        b_amp = np.sqrt(result.reflectance)
        guide_reflected.set_data([x_min, boundaries[0]], [b_amp, b_amp])
        text_reflected.set_position((x_min + 0.05, b_amp))
        text_reflected.set_text(reflected_amplitude_label(result))
        c_amp = abs(result.coefficients[-1][0])
        guide_transmitted.set_data([boundaries[-1], x_max], [c_amp, c_amp])
        text_transmitted.set_position((x_max - 0.05, c_amp))
        text_transmitted.set_text(transmitted_amplitude_label(result))

    def redraw_wave():
        """いまの E と構造で波動関数を計算し、E の線・中段の図・数値表示を描き直す"""
        barriers = state["barriers"]
        boundaries, potentials = bm.to_potential(barriers)
        energy = sliders["energy"].val
        x_min, x_max = structure_x_range(barriers)

        result = sc.solve_scattering(energy, boundaries, potentials, sliders["mass"].val)
        x = np.linspace(x_min, x_max, N_POINTS)
        phi = sc.wavefunction(x, result)
        right, left = sc.traveling_components(x, result)

        line_energy.set_data([x_min - 100, x_max + 100], [energy, energy])
        energy_unit = state["units"]["energy"]
        label_energy.set_position((x_min + 0.05, energy))
        label_energy.set_text(f"E = {energy * UNIT_SCALES['energy'][energy_unit]:.6g} {energy_unit}")

        state.update(x=x, phi=phi, right=right, left=left, energy=energy)
        redraw_real_part()
        line_density.set_data(x, np.abs(phi) ** 2)
        redraw_amplitude_guides(result, boundaries, x_min, x_max)
        for patch in shaded:
            patch.remove()
        shaded.clear()
        shaded.extend(
            ax_wave.axvspan(max(left_edge, x_min), min(right_edge, x_max), color=COLOR_BARRIER, zorder=0)
            for left_edge, right_edge in barrier_spans(boundaries, potentials)
        )
        apply_visibility()

        marker_energy.set_xdata([energy, energy])
        dot_t.set_data([energy], [result.transmittance])
        dot_r.set_data([energy], [result.reflectance])
        readout.set_text(
            f"反射率 R = {result.reflectance:.4f}\n"
            f"透過率 T = {result.transmittance:.4g}\n"
            f"R + T    = {result.reflectance + result.transmittance:.10f}"
        )
        redraw_peak_width()

    def spectrum_shown():
        """T(E) の図のチェックボックスで選ばれている項目の集合"""
        return {label.get_text() for label, on in zip(spec_check.labels, spec_check.get_status()) if on}

    def redraw_peak_width():
        """
        いまの E に一番近い T の山について、半値全幅を求めて図と文字で示す。
        E を動かしたとき・T(E) を計算し直したときに呼ぶ（山の探し直しは数 ms で済む）。
        """
        energies, _reflectances, transmittances = state["spectrum"]
        index = nearest_peak_index(energies, transmittances, state["energy"])
        if index is None or not state["barriers"]:
            state["peak"] = None
        else:
            boundaries, potentials = bm.to_potential(state["barriers"])
            state["peak"] = sc.half_maximum_width(
                energies, transmittances, index, boundaries, potentials, sliders["mass"].val
            )
        peak = state["peak"]
        width_text.set_text(peak_width_text(peak, state["units"]))
        if peak is not None and peak.width is not None:
            half = peak.t_peak / 2
            width_arrow.xy, width_arrow.xyann = (peak.e_left, half), (peak.e_right, half)
            width_label.xy = (peak.e_right, half)
            unit = state["units"]["energy"]
            width_label.set_text(f"ΔE = {peak.width * UNIT_SCALES['energy'][unit]:.4g} {unit}")
        if peak is not None:
            width_top.set_data([peak.e_peak], [peak.t_peak])
        apply_spectrum_visibility()

    def apply_spectrum_visibility():
        """T(E) の図で、チェックボックスで選んだものだけを表示し、凡例と対数表示の縦軸の範囲を合わせる"""
        selected = spectrum_shown()
        show_t, show_r = SHOW_T in selected, SHOW_R in selected
        for artist in (line_t, dot_t):
            artist.set_visible(show_t)
        for artist in (line_r, dot_r):
            artist.set_visible(show_r)
        # 半値幅は T の山についての量なので、T を表示しているときだけ出す
        peak = state["peak"]
        show_width = show_t and SHOW_WIDTH in selected and peak is not None
        width_top.set_visible(show_width)
        for artist in (width_arrow, width_label):
            artist.set_visible(show_width and peak.width is not None)
        width_text.set_visible(show_t and SHOW_WIDTH in selected)
        # 凡例は表示している線だけで作り直す（隠した線が凡例に残らないように）
        visible = [line for line in (line_t, line_r) if line.get_visible()]
        if visible:
            ax_spec.legend(handles=visible, loc="center right", frameon=False)
        elif ax_spec.get_legend() is not None:
            ax_spec.get_legend().remove()
        if state["log"] and state["spectrum"] is not None:
            _energies, reflectances, transmittances = state["spectrum"]
            values = [t for t, on in ((transmittances, show_t), (reflectances, show_r)) if on]
            if values:
                ax_spec.set_ylim(log_lower_limit(np.concatenate(values)), 2.0)

    def redraw_spectrum():
        """T(E), R(E) を、いま表示している E の範囲で計算し直す（構造が変わったときと、T(E) の図の範囲が変わったとき）"""
        boundaries, potentials = bm.to_potential(state["barriers"])
        energies = spectrum_energies(*ax_spec.get_xlim())
        reflectances, transmittances = sc.spectrum(energies, boundaries, potentials, sliders["mass"].val)
        line_t.set_data(energies, transmittances)
        line_r.set_data(energies, reflectances)
        state["spectrum"] = (energies, reflectances, transmittances)
        if state["energy"] is not None:  # 最初の1回は E がまだ決まっていないので、redraw_wave に任せる
            redraw_peak_width()  # 対数表示の縦軸の範囲もここで合わせる

    def refresh_edit_panel():
        """右上の数値入力欄に、選択中の障壁の値を表示する（選択がなければ空欄）"""
        index, barriers, units = state["selected"], state["barriers"], state["units"]
        if index is None:
            selection_text.set_text("障壁をクリックすると、ここで数値を編集できます")
            for box in edit_boxes.values():
                show_text_silently(box, "")
            return
        selection_text.set_text(f"{index + 1} 番目（左から）/ 全 {len(barriers)} 個")
        for key, value in barrier_values(barriers, index).items():
            # 左端の障壁には左隣がないので、間隔の欄は「—」にする
            show_text_silently(edit_boxes[key], "—" if value is None else format_parameter(key, value, units))

    def on_structure_change(fit: bool = True):
        """
        障壁の並びが変わったとき。上段・スペクトル・波動関数・入力欄を描き直す。
        ドラッグ中は入力欄を書き換えない（TextBox は文字を書き換えるたびに画面全体を描き直すので、
        ドラッグが重くなる）。ドラッグ中の値は吹き出しに出し、入力欄はマウスを離したときに書き換える。
        """
        redraw_structure()
        redraw_spectrum()
        redraw_wave()
        if fit:
            fit_axes()
        if state["drag"] is None:
            refresh_edit_panel()
        fig.canvas.draw_idle()

    def set_barriers(barriers, fit: bool = True):
        """新しい並びにして描き直す。選択中の番号が並びの外になったら選択を外す"""
        state["barriers"] = barriers
        if state["selected"] is not None and state["selected"] >= len(barriers):
            state["selected"] = None
        on_structure_change(fit)

    # -----------------------------------------------------------------------
    # マウスで障壁を操作する
    # -----------------------------------------------------------------------
    def data_position(event):
        """
        マウスの位置を上段の図のデータ座標 (x [nm], V [eV]) に直す。
        event.xdata は図の外では None になるので、ピクセル座標 (event.x, event.y) から自分で変換する
        （ドラッグ中にマウスが図の外に出ても、操作を続けられるように）。
        """
        return ax_pot.transData.inverted().transform((event.x, event.y))

    def hit_tolerance(event):
        """HIT_TOLERANCE_PX ピクセルが、いまの表示範囲でデータの単位でいくつに当たるか (tol_x [nm], tol_v [eV])"""
        to_data = ax_pot.transData.inverted()
        x0, v0 = to_data.transform((event.x, event.y))
        x1, v1 = to_data.transform((event.x + HIT_TOLERANCE_PX, event.y + HIT_TOLERANCE_PX))
        return abs(x1 - x0), abs(v1 - v0)

    def toolbar_busy() -> bool:
        """ツールバーの拡大・移動モード中か（そのときのクリックはツールバーの操作なので、障壁は触らない）"""
        toolbar = fig.canvas.toolbar
        return toolbar is not None and toolbar.mode != ""

    def set_cursor(cursor):
        if state["cursor"] != cursor:
            fig.canvas.set_cursor(cursor)
            state["cursor"] = cursor

    def show_tooltip(index: int, part: str, x: float, v: float) -> None:
        """(x, v) の近くに index 番目の障壁の値と操作の説明を出す。図の右寄りでは、はみ出さないよう左側に出す"""
        tooltip.xy = (x, v)
        tooltip.set_text(describe_barrier(state["barriers"], index, state["units"]) + f"\n［{PART_HINTS[part]}］")
        x_low, x_high = ax_pot.get_xlim()
        on_right_side = (x - x_low) / (x_high - x_low) > 0.65
        tooltip.set_position((-12, 12) if on_right_side else (12, 12))
        tooltip.set_horizontalalignment("right" if on_right_side else "left")
        tooltip.set_visible(True)

    def on_press(event):
        if event.inaxes is not ax_pot or event.button != 1 or toolbar_busy():
            return
        x, v = data_position(event)
        hit = bm.hit_test(state["barriers"], x, v, *hit_tolerance(event))
        # クリックした障壁を選択する（何もない所なら選択を外す）
        state["selected"] = None if hit is None else hit.index
        if hit is not None:
            # つかんだらドラッグ開始。入力欄はマウスを離したときに書き換える（on_release）
            state["drag"] = {"hit": hit, "origin": state["barriers"], "x0": x, "v0": v}
        else:
            refresh_edit_panel()
        redraw_structure()
        fig.canvas.draw_idle()

    def on_motion(event):
        if state["drag"] is not None:
            drag = state["drag"]
            x, v = data_position(event)
            new = bm.drag(drag["origin"], drag["hit"], x - drag["x0"], v - drag["v0"])
            # 吸着の刻みより小さい動きでは並びが変わらないので、計算し直さない（動きを軽くするため）
            if new != state["barriers"]:
                state["barriers"] = new
                show_tooltip(drag["hit"].index, drag["hit"].part, x, v)  # ドラッグ中の値は吹き出しで見せる
                on_structure_change(fit=False)
            return
        if event.inaxes is not ax_pot or toolbar_busy():
            if state["hover"] is not None:
                state["hover"] = None
                tooltip.set_visible(False)
                redraw_structure()
                set_cursor(Cursors.POINTER)
                fig.canvas.draw_idle()
            return
        x, v = data_position(event)
        hit = bm.hit_test(state["barriers"], x, v, *hit_tolerance(event))
        if hit != state["hover"]:
            state["hover"] = hit
            set_cursor(Cursors.POINTER if hit is None else PART_CURSORS[hit.part])
            redraw_structure()
        if hit is None:
            tooltip.set_visible(False)
        else:
            show_tooltip(hit.index, hit.part, x, v)
        fig.canvas.draw_idle()

    def on_release(event):
        if state["drag"] is None:
            return
        state["drag"] = None
        tooltip.set_visible(False)
        fit_axes()  # ドラッグが終わってから、表示範囲を新しい構造に合わせる
        redraw_structure()
        refresh_edit_panel()  # ドラッグ中は止めていた入力欄の更新
        fig.canvas.draw_idle()

    # -----------------------------------------------------------------------
    # 入力欄・ボタン
    # -----------------------------------------------------------------------
    def on_edit_submit(key, text):
        """選択中の障壁の数値入力欄で Enter が押されたとき"""
        index = state["selected"]
        if index is None:
            show_error("先に障壁をクリックして選んでください")
            refresh_edit_panel()
            fig.canvas.draw_idle()
            return
        try:
            value = parse_parameter(key, text, state["units"])
            setter = {"height": bm.set_height, "width": bm.set_width, "gap": bm.set_gap_before}[key]
            new = setter(state["barriers"], index, value)
            bm.validate(new)
        except ValueError as err:
            show_error(str(err))
            refresh_edit_panel()  # 入力欄を今の値に戻す
            fig.canvas.draw_idle()
            return
        show_error("")
        set_barriers(new)

    def on_add(_event):
        try:
            new = bm.append_barrier(state["barriers"])
        except ValueError as err:
            show_error(str(err))
            fig.canvas.draw_idle()
            return
        show_error("")
        state["selected"] = len(new) - 1  # 足した障壁を選択して、すぐ数値を入れられるようにする
        set_barriers(new)

    def on_delete(_event):
        if state["selected"] is None:
            show_error("削除する障壁をクリックして選んでください")
            fig.canvas.draw_idle()
            return
        show_error("")
        new = bm.remove_barrier(state["barriers"], state["selected"])
        state["selected"] = None
        set_barriers(new)

    def on_template(_event):
        """ひな形の入力欄の値で、同じ障壁を等間隔に並べ直す"""
        try:
            values = {key: parse_parameter(key, box.text, state["units"]) for key, box in template_boxes.items()}
            new = bm.uniform(int(values["count"]), values["height"], values["width"], values["gap"])
        except ValueError as err:
            show_error(str(err))
            fig.canvas.draw_idle()
            return
        show_error("")
        state["selected"] = None
        set_barriers(new)

    # --- E・有効質量（viewer.py と同じ動き） ---
    def sync_slider_box(key):
        show_text_silently(slider_boxes[key], format_parameter(key, sliders[key].val, state["units"]))

    def on_energy_change(_value):
        sync_slider_box("energy")
        redraw_wave()
        if state["drag"] is None:
            fit_axes()
        fig.canvas.draw_idle()

    def on_mass_change(_value):
        sync_slider_box("mass")
        redraw_spectrum()
        redraw_wave()
        fig.canvas.draw_idle()

    def on_slider_submit(key, text):
        try:
            value = parse_parameter(key, text, state["units"])
        except ValueError as err:
            show_error(str(err))
            sync_slider_box(key)
            fig.canvas.draw_idle()
            return
        show_error("")
        slider = sliders[key]
        if key == "energy":
            e_low, e_high = ax_spec.get_xlim()
            if not e_low <= value <= e_high:
                margin = 0.05 * (max(e_high, value) - min(e_low, value))
                ax_spec.set_xlim(min(e_low, max(value - margin, ENERGY_FLOOR)), max(e_high, value + margin))
        elif not slider.valmin <= value <= slider.valmax:
            set_slider_range(slider, min(slider.valmin, value), max(slider.valmax, value))
        slider.set_val(value)

    def on_spectrum_xlim(_ax):
        """T(E) の図の横軸の範囲が変わったら、E スライダーの範囲を合わせ、T(E) を計算し直す（viewer.py と同じ）"""
        e_low, e_high = ax_spec.get_xlim()
        e_low = max(e_low, ENERGY_FLOOR)
        energy_slider = sliders["energy"]
        set_slider_range(energy_slider, e_low, e_high)
        redraw_spectrum()
        if not e_low <= energy_slider.val <= e_high:
            energy_slider.set_val((e_low + e_high) / 2)
        fig.canvas.draw_idle()

    def on_spectrum_click(event):
        if event.inaxes is not ax_spec or event.button != 1 or event.xdata is None or toolbar_busy():
            return
        if event.xdata >= ENERGY_FLOOR:
            sliders["energy"].set_val(event.xdata)

    def on_check(_label):
        apply_visibility()
        fig.canvas.draw_idle()

    def on_log_toggle(_label):
        state["log"] = log_check.get_status()[0]
        if state["log"]:
            ax_spec.set_yscale("log")
            ax_spec.set_ylabel("確率（対数）")
            apply_spectrum_visibility()  # 縦軸の下限を、表示している曲線の最小値に合わせて決める
        else:
            ax_spec.set_yscale("linear")
            ax_spec.set_ylabel("確率")
            ax_spec.set_ylim(-0.03, 1.03)
        fig.canvas.draw_idle()

    # --- 単位 ---
    def apply_units():
        """いまの表示単位に合わせて、ラベル・入力欄・目盛り・ボタンの文字を書き直す（計算の値は変えない）"""
        units = state["units"]
        for key, slider in sliders.items():
            slider.label.set_text(parameter_label(key, units))
            sync_slider_box(key)
        for key, box in edit_boxes.items():
            box.label.set_text(edit_label(key, units) + " ")
        # ひな形の欄の数値の換算は on_unit_toggle で行う（ここではラベルだけ）
        for key, box in template_boxes.items():
            box.label.set_text(parameter_label(key, units) + " ")
        for kind, button in unit_buttons.items():
            button.label.set_text(unit_button_label(kind, units))
        refresh_edit_panel()
        energy_scale = UNIT_SCALES["energy"][units["energy"]]
        length_scale = UNIT_SCALES["length"][units["length"]]
        set_tick_unit(ax_wave.xaxis, length_scale)  # 上段と x 軸を共有しているので、上段も同じになる
        set_tick_unit(ax_pot.yaxis, energy_scale)
        set_tick_unit(ax_spec.xaxis, energy_scale)
        ax_wave.set_xlabel(f"x [{units['length']}]")
        ax_pot.set_ylabel(f"V [{units['energy']}]")
        ax_spec.set_xlabel(f"E [{units['energy']}]")

    def on_unit_toggle(kind):
        old_units = state["units"]
        new_units = {**old_units, kind: other_unit(kind, old_units[kind])}
        # ひな形の欄の数値を、古い単位で読んで新しい単位で書き直す
        for key, box in template_boxes.items():
            try:
                show_text_silently(box, format_parameter(key, parse_parameter(key, box.text, old_units), new_units))
            except ValueError:
                pass  # 読めない値はそのまま残す（並べ直すボタンを押したときにエラーとして知らせる）
        state["units"] = new_units
        show_error("")
        apply_units()
        redraw_wave()
        fig.canvas.draw_idle()

    # --- 再生（viewer.py と同じ） ---
    timer = fig.canvas.new_timer(interval=FRAME_INTERVAL_MS)

    def on_tick():
        state["phase"] += 2 * np.pi / FRAMES_PER_PERIOD
        redraw_real_part()
        fig.canvas.draw_idle()

    def on_play(_event):
        state["playing"] = not state["playing"]
        if state["playing"]:
            timer.start()
        else:
            timer.stop()
        play_button.label.set_text("停止" if state["playing"] else "再生")
        fig.canvas.draw_idle()

    def on_reset(_event):
        state["phase"] = 0.0
        redraw_real_part()
        fig.canvas.draw_idle()

    # -----------------------------------------------------------------------
    # 部品と関数をつなぐ
    # -----------------------------------------------------------------------
    timer.add_callback(on_tick)
    play_button.on_clicked(on_play)
    reset_button.on_clicked(on_reset)
    add_button.on_clicked(on_add)
    delete_button.on_clicked(on_delete)
    template_button.on_clicked(on_template)
    check.on_clicked(on_check)
    log_check.on_clicked(on_log_toggle)
    spec_check.on_clicked(lambda _label: (apply_spectrum_visibility(), fig.canvas.draw_idle()))
    sliders["energy"].on_changed(on_energy_change)
    sliders["mass"].on_changed(on_mass_change)
    # key=key は「この行を実行した時点の key を覚えておく」ための書き方（ないと全部最後の key になる）
    for key, box in slider_boxes.items():
        box.on_submit(lambda text, key=key: on_slider_submit(key, text))
    for key, box in edit_boxes.items():
        box.on_submit(lambda text, key=key: on_edit_submit(key, text))
    for kind, button in unit_buttons.items():
        button.on_clicked(lambda _event, kind=kind: on_unit_toggle(kind))
    ax_spec.callbacks.connect("xlim_changed", on_spectrum_xlim)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_press_event", on_spectrum_click)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    # 最初の1回を描く
    apply_units()
    on_spectrum_xlim(ax_spec)
    on_structure_change()
    return fig, {
        "sliders": sliders,
        "slider_boxes": slider_boxes,
        "edit_boxes": edit_boxes,
        "template_boxes": template_boxes,
        "buttons": {
            "add": add_button,
            "delete": delete_button,
            "template": template_button,
            "play": play_button,
            "reset": reset_button,
        },
        "units": unit_buttons,
        "check": check,
        "log_check": log_check,
        "spec_check": spec_check,
        "timer": timer,
        "state": state,
    }


if __name__ == "__main__":
    fig, widgets = build_editor()
    plt.show()
