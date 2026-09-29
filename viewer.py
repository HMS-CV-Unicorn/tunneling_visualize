"""
viewer.py — 反射と透過をスライダーで動かして見る

実行:  .venv/bin/python viewer.py

画面の構成（上から）
  1. ポテンシャル V(x) と、電子のエネルギー E（横線）
  2. 波動関数。チェックボックスで表示するものを選ぶ
       - Re ψ(x,t)            : 合成波（いまの波動関数そのものの実部）
       - 右向き成分・左向き成分: 合成波を「入射・透過（右向き）」と「反射（左向き）」に分けたもの
       - |ψ|² = |φ|²          : 存在確率（時間によらない）
     「再生」を押すと時間因子 e^{-iEt/ħ} を掛けた ψ(x,t) = φ(x) e^{-iEt/ħ} が動く
  3. 透過率 T と反射率 R を E の関数として描いたもの。いまの E の位置に印

操作
  - スライダーを動かすか、右の入力欄に数値を打って Enter。スライダーの範囲外の値を入れると範囲が広がる
  - ウィンドウ上部のツールバー（虫眼鏡 = 範囲を囲んで拡大、十字 = 平行移動、家 = 元に戻す）で図を拡大できる。
    T(E) の図を拡大すると、表示範囲の中で T(E) を計算し直し、E スライダーの範囲もその表示範囲になる
  - T(E) の図をクリックすると、その E に移動する（ツールバーの拡大・移動モード中を除く）
  - 「T, R を対数表示」で縦軸を対数にすると、トンネル効果の非常に小さい T も見える

計算はすべて scattering.py に任せ、このファイルは「スライダーの値を読む → 計算を呼ぶ → 描き直す」だけを担当する。

使っているライブラリ
  - matplotlib        : グラフを描くライブラリ
  - matplotlib.widgets: グラフの画面にスライダー・入力欄・ボタンを置くための部品
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox

import scattering as sc

# macOS の日本語フォント（ヒラギノ）を使う。無い環境では DejaVu Sans（英字のみ）で代用される
plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]

# 色は「波の向き」で分ける。赤と緑は色覚によって区別しにくいので避け、緑系と橙にしている
# （色覚シミュレーションで区別できることを確認済み）。色だけに頼らず、線の種類（実線/破線）でも区別する。
COLOR_TOTAL = "#2a78d6"  # 青  : 合成波 Re ψ
COLOR_RIGHT = "#1baf7a"  # 緑系: 右向き成分（入射・透過）、スペクトルの T
COLOR_LEFT = "#eb6834"  # 橙  : 左向き成分（反射）、スペクトルの R
COLOR_DENSITY = "#52514e"  # 濃い灰: |ψ|²
COLOR_INK = "#0b0b0b"  # 文字や軸（文字は系列の色にしない）
COLOR_MUTED = "#8a8985"
COLOR_BARRIER = "#e9e8e4"  # 波動関数の図で山の場所を示す薄い灰色
COLOR_ERROR = "#b3261e"  # 入力エラーの文字

MODE_BARRIERS = "山 × N 個"
MODE_STEP = "階段"

# 中段のチェックボックスの項目（表示名）
SHOW_TOTAL = "Re ψ（合成波）"
SHOW_COMPONENTS = "入射・反射・透過に分解"
SHOW_DENSITY = "|ψ|²（存在確率）"
LOG_SCALE = "T, R を対数表示"

MARGIN = 3.0  # ポテンシャルの左右に表示する余白 [nm]
N_POINTS = 3000  # 波動関数を描く点の数

# T(E), R(E) のグラフ。表示している E の範囲の中で、いつも ENERGY_POINTS 点で計算する。
# 拡大すれば点の間隔が細かくなるので、幅 1 meV 程度の鋭い共鳴ピークも形が見える
ENERGY_POINTS = 800
ENERGY_VIEW = (0.005, 6.0)  # 最初に表示する E の範囲 [eV]
ENERGY_FLOOR = 1e-4  # E はこれより大きくする [eV]（左端は V = 0 なので、入射するには E > 0 が必要）

MAX_COUNT = 30  # 山の個数の上限。力技の連立方程式は山の数に比例して大きくなり、T(E) の計算が遅くなるため

# アニメーションの速さ。実際の周期 2πħ/E は数フェムト秒（10^-15 s）なので、そのままでは見えない。
# そこで「時間因子が1回転する間を FRAMES_PER_PERIOD コマで描く」スローモーションにする。
# E によって実際の周期は違うが、画面上ではどの E でも約 3 秒で1回転する（本当の時刻 t は画面に表示する）
FRAMES_PER_PERIOD = 90
FRAME_INTERVAL_MS = 33  # 1コマの間隔 [ミリ秒]（約 30 コマ/秒）

# スライダーの設定: (表示名, 最小, 最大, 初期値, 刻み)
# E の範囲は T(E) の図の表示範囲に合わせて変わる。刻みは付けない（拡大したとき細かく動かせるように）
SLIDER_SPECS = {
    "energy": ("エネルギー E [eV]", ENERGY_VIEW[0], ENERGY_VIEW[1], 0.60, None),
    "height": ("高さ V₀ [eV]", -3.0, 5.0, 1.00, 0.05),
    "width": ("幅 a [nm]", 0.05, 3.0, 0.50, 0.05),
    "count": ("個数 N", 0, 5, 1, 1),
    "gap": ("間隔 b [nm]", 0.05, 5.0, 1.00, 0.05),
}


# ---------------------------------------------------------------------------
# 描画に使う値を作る（副作用のない関数）
# ---------------------------------------------------------------------------
def build_potential(mode: str, params: dict):
    """選ばれたモードとスライダーの値から (境界の位置, 各領域のポテンシャル) を作る"""
    if mode == MODE_STEP:
        return sc.step(params["height"])
    return sc.barrier_array(int(params["count"]), params["height"], params["width"], params["gap"])


def x_range(boundaries) -> tuple[float, float]:
    """描画する x の範囲。ポテンシャルが変化する区間の左右に MARGIN の余白をとる"""
    right_end = boundaries[-1] if boundaries else 0.0
    return -MARGIN, right_end + MARGIN


def potential_outline(boundaries, potentials, x_min, x_max):
    """
    V(x) を角ばった線として描くための頂点列。境界では同じ x に2点置いて、垂直な段差にする。
    """
    xs = [x_min]
    vs = [potentials[0]]
    for x, v_left, v_right in zip(boundaries, potentials[:-1], potentials[1:]):
        xs += [x, x]
        vs += [v_left, v_right]
    xs.append(x_max)
    vs.append(potentials[-1])
    return np.array(xs), np.array(vs)


def barrier_spans(boundaries, potentials):
    """V ≠ 0 の区間 (左端, 右端) の一覧。波動関数の図で、山のある場所に薄い背景を敷くために使う"""
    edges = (-np.inf, *boundaries, np.inf)
    return [
        (left, right)
        for left, right, v in zip(edges[:-1], edges[1:], potentials)
        if v != 0.0
    ]


def padded_limits(values, minimum_span: float) -> tuple[float, float]:
    """値の範囲に 10% ほど余白を足した軸の範囲。少なくとも minimum_span の幅はとる"""
    low, high = float(np.min(values)), float(np.max(values))
    span = max(high - low, minimum_span)
    center = (low + high) / 2
    return center - 0.55 * span, center + 0.55 * span


def spectrum_energies(e_low: float, e_high: float) -> np.ndarray:
    """T(E) を計算する E の点。表示範囲 [e_low, e_high] の中に ENERGY_POINTS 点を等間隔に置く"""
    return np.linspace(max(e_low, ENERGY_FLOOR), e_high, ENERGY_POINTS)


def log_lower_limit(values) -> float:
    """
    対数表示の縦軸の下限。0 は対数で描けないので、正の値のうち最小のものより少し下にする。
    （階段の E < V₀ では T = 0 ちょうどなので、その点は対数表示では描かれない）
    """
    positive = np.asarray(values)[np.asarray(values) > 0]
    return max(positive.min() / 3, 1e-300) if positive.size else 1e-12


def reflected_amplitude_label(result: sc.Scattering) -> str:
    """反射波の振幅 |B| のラベル。R = |B|² なので、振幅は R ではなく √R であることを明示する"""
    return f"反射波の振幅 |B| = √R = {np.sqrt(result.reflectance):.2f}"


def transmitted_amplitude_label(result: sc.Scattering) -> str:
    """
    透過波の振幅 |C| のラベル。T = (k_右/k_左)|C|²（教科書 (5.11)(5.13)）なので、
    両端のポテンシャルが同じ（山 × N 個）なら |C| = √T、階段なら速さの比の分だけずれる。
    """
    amplitude = abs(result.coefficients[-1][0])
    if result.potentials[0] == result.potentials[-1]:
        return f"透過波の振幅 |C| = √T = {amplitude:.2f}"
    return f"透過波の振幅 |C| = {amplitude:.2f}（T = (k₂/k₁)|C|²）"


def parse_parameter(key: str, text: str) -> float:
    """
    入力欄の文字列を数値に変換する。数値として読めない、または物理的に意味のない値なら、
    理由を書いた ValueError を出す（呼び出し側で画面に表示する）。
    """
    name = SLIDER_SPECS[key][0]
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{name}: 「{text}」は数値として読めません") from None
    if not np.isfinite(value):
        raise ValueError(f"{name}: 有限の数値を入れてください")
    if key == "energy" and value < ENERGY_FLOOR:
        raise ValueError(f"{name}: 0 より大きくしてください（左端は V = 0 なので、入射するには E > 0 が必要）")
    if key in ("width", "gap") and value <= 0:
        raise ValueError(f"{name}: 0 より大きくしてください")
    if key == "count":
        if value < 0 or value != int(value):
            raise ValueError(f"{name}: 0 以上の整数にしてください")
        if value > MAX_COUNT:
            raise ValueError(f"{name}: {MAX_COUNT} 以下にしてください（連立方程式が大きくなり計算が遅くなるため）")
    return value


def format_parameter(key: str, value: float) -> str:
    """入力欄に表示する文字列。個数は整数、それ以外は有効数字 6 桁"""
    return str(int(value)) if key == "count" else f"{value:.6g}"


def wave_axis_label(show_wave: bool, show_density: bool) -> str:
    """
    中段の図の縦軸のラベル。表示している量に合わせて変える。

    散乱状態は規格化できず、連続条件で決まるのは B/A, C/A などの比だけなので、入射波の振幅を A = 1 と決めて
    描いている。だから縦軸は単位のない「入射波を基準にした相対値」。
    振幅（Re ψ）とその2乗（|ψ|²）は別の量なので、どちらを表示しているかをラベルで明示する。
    """
    if show_wave and show_density:
        return r"Re $\psi$ と $|\psi|^2$（A = 1 を基準）"
    if show_density:
        return r"$|\psi|^2$（入射波の密度 $|A|^2$ = 1 を基準）"
    return r"Re $\psi$（入射波の振幅 A = 1 を基準）"


# ---------------------------------------------------------------------------
# 部品の見た目を変える小さな関数（matplotlib の部品を書き換える）
# ---------------------------------------------------------------------------
def set_slider_range(slider: Slider, low: float, high: float) -> None:
    """
    スライダーの左端・右端を変える。matplotlib には専用の関数がないので、
    最小値・最大値・表示範囲・塗りつぶし部分（左端から今の値まで）をまとめて書き換える。
    """
    slider.valmin, slider.valmax = low, high
    slider.ax.set_xlim(low, high)
    slider.poly.set_x(low)
    slider.poly.set_width(slider.val - low)


def show_text_silently(box: TextBox, text: str) -> None:
    """
    入力欄の表示だけを書き換える。TextBox.set_val は「入力が確定した」という合図も出してしまい、
    スライダー → 入力欄 → スライダー … と無限に呼び合うので、書き換える間だけ合図を止める。
    """
    box.eventson = False
    box.set_val(text)
    box.eventson = True


# ---------------------------------------------------------------------------
# 画面を作る
# ---------------------------------------------------------------------------
def build_viewer():
    """
    図とスライダーを作り、スライダーを動かしたら描き直すようにつなぐ。

    戻り値の widgets は必ず変数に保持しておくこと。
    （保持しないと Python がスライダーを「使われていない」とみなして消してしまい、操作が効かなくなる）
    """
    fig = plt.figure(figsize=(12, 9.5))
    fig.canvas.manager.set_window_title("反射と透過")

    # 上2つの図は x 軸を共有（sharex）して、V(x) と φ(x) の位置がぴったり揃うようにする
    ax_pot = fig.add_axes((0.08, 0.825, 0.88, 0.13))
    ax_wave = fig.add_axes((0.08, 0.50, 0.88, 0.28), sharex=ax_pot)
    ax_spec = fig.add_axes((0.08, 0.20, 0.58, 0.22))

    # --- 1. ポテンシャル ---
    (line_pot,) = ax_pot.plot([], [], color=COLOR_INK, lw=2)
    (line_energy,) = ax_pot.plot([], [], color=COLOR_INK, lw=1.2, ls="--")
    label_energy = ax_pot.text(0, 0, "", color=COLOR_INK, va="bottom", ha="left", fontsize=10)
    ax_pot.set_ylabel("V [eV]")
    ax_pot.tick_params(labelbottom=False)
    ax_pot.set_title("ポテンシャル V(x) と電子のエネルギー E（左から入射）", loc="left", fontsize=11)

    # --- 2. 波動関数 ---
    # 成分は細い破線、合成波は太い実線にして、色が見分けにくくても区別できるようにする
    (line_right,) = ax_wave.plot(
        [], [], color=COLOR_RIGHT, lw=1.6, ls="--", label="右向き成分（入射・透過）"
    )
    (line_left,) = ax_wave.plot([], [], color=COLOR_LEFT, lw=1.6, ls="--", label="左向き成分（反射）")
    (line_total,) = ax_wave.plot([], [], color=COLOR_TOTAL, lw=2.2, label=r"$\mathrm{Re}\,\psi$（合成波）")
    (line_density,) = ax_wave.plot(
        [], [], color=COLOR_DENSITY, lw=2, label=r"$|\psi|^2 = |\phi|^2$"
    )
    # 反射波・透過波の振幅の目安線（左端の領域と右端の領域にだけ引く）
    (guide_reflected,) = ax_wave.plot([], [], color=COLOR_LEFT, lw=1, ls=":")
    (guide_transmitted,) = ax_wave.plot([], [], color=COLOR_RIGHT, lw=1, ls=":")
    text_reflected = ax_wave.text(0, 0, "", color=COLOR_INK, fontsize=9, va="bottom", ha="left")
    text_transmitted = ax_wave.text(0, 0, "", color=COLOR_INK, fontsize=9, va="bottom", ha="right")
    time_label = ax_wave.text(0.01, 0.97, "", transform=ax_wave.transAxes, va="top", fontsize=10)
    ax_wave.axhline(0, color=COLOR_MUTED, lw=0.6)
    ax_wave.set_xlabel("x [nm]")
    # 凡例は図の中に置くと波や振幅ラベルと重なるので、図の上に横一列で並べる
    ax_wave.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, fontsize=9, frameon=False)
    shaded = []  # 山の背景（毎回作り直すので、消すために覚えておく）

    # --- 3. 透過率・反射率のスペクトル（色は中段の成分とそろえる: 透過 = 緑系、反射 = 橙） ---
    (line_t,) = ax_spec.plot([], [], color=COLOR_RIGHT, lw=2.2, label="透過率 T")
    (line_r,) = ax_spec.plot([], [], color=COLOR_LEFT, lw=2, label="反射率 R")
    (dot_t,) = ax_spec.plot([], [], "o", color=COLOR_RIGHT, ms=8, mec="white", mew=2)
    (dot_r,) = ax_spec.plot([], [], "o", color=COLOR_LEFT, ms=8, mec="white", mew=2)
    marker_energy = ax_spec.axvline(0, color=COLOR_INK, lw=1, ls="--")
    marker_height = ax_spec.axvline(0, color=COLOR_MUTED, lw=1, ls=":")
    ax_spec.set_xlim(*ENERGY_VIEW)
    ax_spec.set_ylim(-0.03, 1.03)
    ax_spec.set_xlabel("E [eV]")
    ax_spec.set_ylabel("確率")
    ax_spec.legend(loc="center right", frameon=False)
    ax_spec.set_title("T(E), R(E)（点線 = V₀、破線 = いまの E、クリックで E を移動）", loc="left", fontsize=11)

    for ax in (ax_pot, ax_wave, ax_spec):
        ax.grid(color="#e9e8e4", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    # いまの R, T の数値（右側の枠に表示）と、入力エラーの表示（右上）
    readout = fig.text(0.70, 0.445, "", fontsize=12, va="top")
    error_text = fig.text(0.96, 0.99, "", color=COLOR_ERROR, fontsize=10, ha="right", va="top")

    # --- 操作部品 ---
    check = CheckButtons(
        fig.add_axes((0.70, 0.285, 0.26, 0.075)),
        (SHOW_TOTAL, SHOW_COMPONENTS, SHOW_DENSITY),
        actives=(True, True, False),
    )
    log_check = CheckButtons(fig.add_axes((0.50, 0.425, 0.16, 0.03)), (LOG_SCALE,), actives=(False,))
    for spine in log_check.ax.spines.values():
        spine.set_visible(False)
    radio_ax = fig.add_axes((0.70, 0.17, 0.16, 0.075))
    radio_ax.set_title("ポテンシャルの形", fontsize=10, loc="left")
    radio = RadioButtons(radio_ax, (MODE_BARRIERS, MODE_STEP), active=0)
    play_button = Button(fig.add_axes((0.88, 0.21, 0.08, 0.035)), "再生")
    reset_button = Button(fig.add_axes((0.88, 0.17, 0.08, 0.035)), "t = 0")

    # スライダーと、その右の数値入力欄（TextBox = 文字を打ち込める欄）
    sliders, boxes = {}, {}
    for row, (key, (label, vmin, vmax, vinit, vstep)) in enumerate(SLIDER_SPECS.items()):
        y = 0.13 - 0.028 * row
        sliders[key] = Slider(fig.add_axes((0.20, y, 0.56, 0.02)), label, vmin, vmax, valinit=vinit, valstep=vstep)
        sliders[key].valtext.set_visible(False)  # 値はスライダー横の文字ではなく入力欄に出す
        boxes[key] = TextBox(fig.add_axes((0.79, y - 0.003, 0.09, 0.026)), "", initial=format_parameter(key, vinit))

    # 画面の状態
    #   mode    : どのポテンシャルの形が選ばれているか
    #   phase   : 時間因子の位相 ωt = Et/ħ [rad]。E を動かしても絵が飛ばないよう、時刻 t ではなく位相で覚える
    #   x, phi  : いま描いている位置の配列と、そこでの φ(x)（再生中は φ を計算し直さず、位相だけ回す）
    #   right, left : φ の右向き成分・左向き成分（進まない領域は NaN）
    #   x_range : 中段の図の x の範囲。ポテンシャルの形が変わったときだけ設定し直す（拡大した表示を保つため）
    #   playing : 再生中かどうか
    #   guide_reflected, guide_transmitted : 振幅の目安線を引けるか（自由空間や、右端で波が進まないときは引かない）
    #   log     : T(E) の図の縦軸が対数表示かどうか
    state = {
        "mode": MODE_BARRIERS,
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
    }

    def current_params():
        return {key: s.val for key, s in sliders.items()}

    def shown():
        """チェックボックスで選ばれている項目の集合"""
        # check.labels は文字を描く部品（Text）の一覧なので、get_text() で文字列を取り出して比べる
        return {label.get_text() for label, on in zip(check.labels, check.get_status()) if on}

    # --- 描き直し ---
    def apply_visibility():
        """チェックボックスの状態に合わせて線の表示/非表示を切り替え、縦軸の範囲を決め直す"""
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

        # 縦軸: 波（Re ψ や成分）は時間とともに ±振幅 の間を動くので、振幅で範囲を決めておく。
        # |ψ|² は振幅の2乗なので桁が大きく違うことがあり、表示しているときだけ範囲に含める
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
        """いまの位相で Re ψ と、その右向き・左向き成分を描き直す（|ψ|² は時間で変わらないので触らない）"""
        energy = state["energy"]
        t = state["phase"] * sc.HBAR_EV_FS / energy  # 位相 Et/ħ から時刻 t [fs] に戻す
        # 合成波にも成分にも、同じ時間因子 e^{-iEt/ħ} を掛ける（NaN は NaN のまま残り、線が途切れる）
        line_total.set_data(state["x"], sc.time_evolve(state["phi"], energy, t).real)
        line_right.set_data(state["x"], sc.time_evolve(state["right"], energy, t).real)
        line_left.set_data(state["x"], sc.time_evolve(state["left"], energy, t).real)
        time_label.set_text(f"t = {t:.3f} fs（1周期 2πħ/E = {sc.period(energy):.3f} fs）")

    def redraw_amplitude_guides(result, boundaries, x_min, x_max):
        """
        左端の領域に反射波の振幅 |B|、右端の領域に透過波の振幅 |C| の目安線を引く。
        境界がない（自由空間）ときや、右端で波が進まない（階段の E < V₀）ときは引かない。
        """
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
        """いまの E とポテンシャルで波動関数を計算し、上2つの図と数値表示を描き直す"""
        params = current_params()
        boundaries, potentials = build_potential(state["mode"], params)
        energy = params["energy"]
        x_min, x_max = x_range(boundaries)

        result = sc.solve_scattering(energy, boundaries, potentials)
        x = np.linspace(x_min, x_max, N_POINTS)
        phi = sc.wavefunction(x, result)
        right, left = sc.traveling_components(x, result)

        # 1. ポテンシャルと E
        xs, vs = potential_outline(boundaries, potentials, x_min, x_max)
        line_pot.set_data(xs, vs)
        line_energy.set_data([x_min, x_max], [energy, energy])
        label_energy.set_position((x_min + 0.05, energy))
        label_energy.set_text(f"E = {energy:.6g} eV")
        # x の範囲はポテンシャルの形が変わったときだけ設定し直す。
        # E を動かすたびに設定すると、ツールバーで拡大した表示が元に戻ってしまうため
        if state["x_range"] != (x_min, x_max):
            ax_pot.set_xlim(x_min, x_max)
            state["x_range"] = (x_min, x_max)
        low, high = padded_limits([*vs, energy, 0.0], minimum_span=1.0)
        ax_pot.set_ylim(low, high + 0.15 * (high - low))  # 上に E のラベルが入る余白をとる

        # 2. 波動関数
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

        # 3. スペクトル上の「いまの E」
        marker_energy.set_xdata([energy, energy])
        dot_t.set_data([energy], [result.transmittance])
        dot_r.set_data([energy], [result.reflectance])

        total = result.reflectance + result.transmittance
        readout.set_text(
            f"反射率 R = {result.reflectance:.4f}\n"
            f"透過率 T = {result.transmittance:.4g}\n"
            f"R + T    = {total:.10f}"
        )

    def redraw_spectrum():
        """
        T(E), R(E) の曲線を、いま表示している E の範囲で計算し直す。
        ポテンシャルが変わったときと、T(E) の図を拡大・移動したときに呼ぶ（E を動かしただけなら不要）
        """
        params = current_params()
        boundaries, potentials = build_potential(state["mode"], params)
        energies = spectrum_energies(*ax_spec.get_xlim())
        reflectances, transmittances = sc.spectrum(energies, boundaries, potentials)
        line_t.set_data(energies, transmittances)
        line_r.set_data(energies, reflectances)
        marker_height.set_xdata([params["height"]] * 2)
        marker_height.set_visible(params["height"] > 0)
        if state["log"]:
            ax_spec.set_ylim(log_lower_limit(np.concatenate([transmittances, reflectances])), 2.0)

    # --- スライダー・入力欄が動いたときに呼ばれる関数 ---
    def sync_box(key):
        """スライダーの値を入力欄にも表示する"""
        show_text_silently(boxes[key], format_parameter(key, sliders[key].val))

    def on_energy_change(_value):
        sync_box("energy")
        redraw_wave()
        fig.canvas.draw_idle()  # 「次に余裕があるときに画面を更新して」と頼む

    def on_potential_change(key):
        sync_box(key)
        redraw_spectrum()
        redraw_wave()
        fig.canvas.draw_idle()

    def on_text_submit(key, text):
        """
        入力欄で Enter が押されたとき。値を確かめてからスライダーに渡す（スライダー側が描き直す）。
        おかしな値なら、何も変えずに理由を右上に表示し、入力欄を今の値に戻す。
        """
        try:
            value = parse_parameter(key, text)
        except ValueError as err:
            error_text.set_text(f"入力エラー: {err}")
            sync_box(key)
            fig.canvas.draw_idle()
            return
        error_text.set_text("")
        slider = sliders[key]
        if key == "energy":
            # E は T(E) の図の表示範囲がそのままスライダーの範囲。範囲外なら図の範囲を広げる
            # （図の範囲が変わると on_spectrum_xlim が呼ばれ、スライダーの範囲も追従する）
            e_low, e_high = ax_spec.get_xlim()
            if not e_low <= value <= e_high:
                margin = 0.05 * (max(e_high, value) - min(e_low, value))
                ax_spec.set_xlim(min(e_low, max(value - margin, ENERGY_FLOOR)), max(e_high, value + margin))
        elif not slider.valmin <= value <= slider.valmax:
            set_slider_range(slider, min(slider.valmin, value), max(slider.valmax, value))
        slider.set_val(value)

    def on_spectrum_xlim(_ax):
        """
        T(E) の図の横軸の範囲が変わったとき（ツールバーで拡大・移動・元に戻す、または E の入力で広げたとき）。
        E スライダーの範囲をこの表示範囲に合わせ、T(E) を表示範囲の中で計算し直す。
        今の E が新しい範囲の外に出たら、E を範囲の中央に移す（スライダーのつまみが範囲外に行かないように）。
        """
        e_low, e_high = ax_spec.get_xlim()
        e_low = max(e_low, ENERGY_FLOOR)
        energy_slider = sliders["energy"]
        set_slider_range(energy_slider, e_low, e_high)
        redraw_spectrum()
        if not e_low <= energy_slider.val <= e_high:
            energy_slider.set_val((e_low + e_high) / 2)  # これで on_energy_change が呼ばれて描き直される
        fig.canvas.draw_idle()

    def on_spectrum_click(event):
        """T(E) の図をクリックしたら、その位置の E に移動する"""
        if event.inaxes is not ax_spec or event.button != 1 or event.xdata is None:
            return
        # ツールバーの拡大（虫眼鏡）・移動（十字）モード中のクリックは、そちらの操作なので無視する
        toolbar = fig.canvas.toolbar
        if toolbar is not None and toolbar.mode != "":
            return
        if event.xdata >= ENERGY_FLOOR:
            sliders["energy"].set_val(event.xdata)

    def on_mode_change(label):
        state["mode"] = label
        # 階段モードでは「個数」と「間隔」は使わないので隠す（階段は幅も無限なので幅も隠す）
        for key in ("count", "gap", "width"):
            sliders[key].ax.set_visible(label != MODE_STEP)
            boxes[key].ax.set_visible(label != MODE_STEP)
        on_potential_change("height")

    def on_check(_label):
        apply_visibility()
        fig.canvas.draw_idle()

    def on_log_toggle(_label):
        """T, R の縦軸を 対数 ⇔ 普通の目盛り で切り替える"""
        state["log"] = log_check.get_status()[0]
        if state["log"]:
            ax_spec.set_yscale("log")
            ax_spec.set_ylabel("確率（対数）")
            redraw_spectrum()  # 縦軸の下限を、いまの曲線の最小値に合わせて決める
        else:
            ax_spec.set_yscale("linear")
            ax_spec.set_ylabel("確率")
            ax_spec.set_ylim(-0.03, 1.03)
        fig.canvas.draw_idle()

    # --- 再生 ---
    # タイマーは「一定間隔で関数を呼んでくれる」matplotlib の部品。再生中は毎コマ位相を少し進めて描き直す
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
        """t = 0 に戻す。このとき Re ψ = Re φ（時間因子を掛ける前の φ そのもの）"""
        state["phase"] = 0.0
        redraw_real_part()
        fig.canvas.draw_idle()

    # --- 部品と関数をつなぐ ---
    timer.add_callback(on_tick)
    play_button.on_clicked(on_play)
    reset_button.on_clicked(on_reset)
    check.on_clicked(on_check)
    log_check.on_clicked(on_log_toggle)
    radio.on_clicked(on_mode_change)
    sliders["energy"].on_changed(on_energy_change)
    for key in ("height", "width", "count", "gap"):
        # key=key は「この行を実行した時点の key を覚えておく」ための書き方（ないと全部最後の key になる）
        sliders[key].on_changed(lambda _value, key=key: on_potential_change(key))
    for key, box in boxes.items():
        box.on_submit(lambda text, key=key: on_text_submit(key, text))
    ax_spec.callbacks.connect("xlim_changed", on_spectrum_xlim)
    fig.canvas.mpl_connect("button_press_event", on_spectrum_click)

    # 最初の1回を描く
    on_spectrum_xlim(ax_spec)
    redraw_wave()
    return fig, {
        "sliders": sliders,
        "boxes": boxes,
        "radio": radio,
        "check": check,
        "log_check": log_check,
        "play": play_button,
        "reset": reset_button,
        "timer": timer,
    }


if __name__ == "__main__":
    fig, widgets = build_viewer()
    plt.show()
