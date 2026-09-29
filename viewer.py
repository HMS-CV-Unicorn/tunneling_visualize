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

計算はすべて scattering.py に任せ、このファイルは「スライダーの値を読む → 計算を呼ぶ → 描き直す」だけを担当する。

使っているライブラリ
  - matplotlib        : グラフを描くライブラリ
  - matplotlib.widgets: グラフの画面にスライダーやボタンを置くための部品
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

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

MODE_BARRIERS = "山 × N 個"
MODE_STEP = "階段"

# 中段のチェックボックスの項目（表示名）
SHOW_TOTAL = "Re ψ（合成波）"
SHOW_COMPONENTS = "入射・反射・透過に分解"
SHOW_DENSITY = "|ψ|²（存在確率）"

MARGIN = 3.0  # ポテンシャルの左右に表示する余白 [nm]
N_POINTS = 3000  # 波動関数を描く点の数
ENERGY_GRID = np.linspace(0.005, 6.0, 800)  # T(E), R(E) を計算するエネルギー [eV]

# アニメーションの速さ。実際の周期 2πħ/E は数フェムト秒（10^-15 s）なので、そのままでは見えない。
# そこで「時間因子が1回転する間を FRAMES_PER_PERIOD コマで描く」スローモーションにする。
# E によって実際の周期は違うが、画面上ではどの E でも約 3 秒で1回転する（本当の時刻 t は画面に表示する）
FRAMES_PER_PERIOD = 90
FRAME_INTERVAL_MS = 33  # 1コマの間隔 [ミリ秒]（約 30 コマ/秒）

# スライダーの設定: (表示名, 最小, 最大, 初期値, 刻み)
SLIDER_SPECS = {
    "energy": ("エネルギー E [eV]", 0.01, 6.0, 0.60, 0.01),
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
    ax_wave.set_ylabel(r"$\psi$（入射振幅 A = 1）")
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
    ax_spec.set_xlim(ENERGY_GRID[0], ENERGY_GRID[-1])
    ax_spec.set_ylim(-0.03, 1.03)
    ax_spec.set_xlabel("E [eV]")
    ax_spec.set_ylabel("確率")
    ax_spec.legend(loc="center right", frameon=False)
    ax_spec.set_title("T(E), R(E)（点線 = V₀、破線 = いまの E）", loc="left", fontsize=11)

    for ax in (ax_pot, ax_wave, ax_spec):
        ax.grid(color="#e9e8e4", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    # いまの R, T の数値（右側の枠に表示）
    readout = fig.text(0.70, 0.445, "", fontsize=12, va="top")

    # --- 操作部品 ---
    check = CheckButtons(
        fig.add_axes((0.70, 0.285, 0.26, 0.075)),
        (SHOW_TOTAL, SHOW_COMPONENTS, SHOW_DENSITY),
        actives=(True, True, False),
    )
    radio_ax = fig.add_axes((0.70, 0.17, 0.16, 0.075))
    radio_ax.set_title("ポテンシャルの形", fontsize=10, loc="left")
    radio = RadioButtons(radio_ax, (MODE_BARRIERS, MODE_STEP), active=0)
    play_button = Button(fig.add_axes((0.88, 0.21, 0.08, 0.035)), "再生")
    reset_button = Button(fig.add_axes((0.88, 0.17, 0.08, 0.035)), "t = 0")

    sliders = {}
    for row, (key, (label, vmin, vmax, vinit, vstep)) in enumerate(SLIDER_SPECS.items()):
        slider_ax = fig.add_axes((0.20, 0.13 - 0.028 * row, 0.60, 0.02))
        sliders[key] = Slider(slider_ax, label, vmin, vmax, valinit=vinit, valstep=vstep)

    # 画面の状態
    #   mode    : どのポテンシャルの形が選ばれているか
    #   phase   : 時間因子の位相 ωt = Et/ħ [rad]。E を動かしても絵が飛ばないよう、時刻 t ではなく位相で覚える
    #   x, phi  : いま描いている位置の配列と、そこでの φ(x)（再生中は φ を計算し直さず、位相だけ回す）
    #   right, left : φ の右向き成分・左向き成分（進まない領域は NaN）
    #   playing : 再生中かどうか
    #   guide_reflected, guide_transmitted : 振幅の目安線を引けるか（自由空間や、右端で波が進まないときは引かない）
    state = {
        "mode": MODE_BARRIERS,
        "phase": 0.0,
        "x": None,
        "phi": None,
        "right": None,
        "left": None,
        "energy": None,
        "playing": False,
        "guide_reflected": False,
        "guide_transmitted": False,
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
        label_energy.set_text(f"E = {energy:.2f} eV")
        ax_pot.set_xlim(x_min, x_max)
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
        """ポテンシャルが変わったときだけ T(E), R(E) の曲線を計算し直す（E を動かしただけなら不要）"""
        params = current_params()
        boundaries, potentials = build_potential(state["mode"], params)
        reflectances, transmittances = sc.spectrum(ENERGY_GRID, boundaries, potentials)
        line_t.set_data(ENERGY_GRID, transmittances)
        line_r.set_data(ENERGY_GRID, reflectances)
        marker_height.set_xdata([params["height"]] * 2)
        marker_height.set_visible(params["height"] > 0)

    # --- スライダーやボタンが動いたときに呼ばれる関数 ---
    def on_energy_change(_value):
        redraw_wave()
        fig.canvas.draw_idle()  # 「次に余裕があるときに画面を更新して」と頼む

    def on_potential_change(_value):
        redraw_spectrum()
        redraw_wave()
        fig.canvas.draw_idle()

    def on_mode_change(label):
        state["mode"] = label
        # 階段モードでは「個数」と「間隔」は使わないので隠す（階段は幅も無限なので幅も隠す）
        for key in ("count", "gap", "width"):
            sliders[key].ax.set_visible(label != MODE_STEP)
        on_potential_change(None)

    def on_check(_label):
        apply_visibility()
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

    timer.add_callback(on_tick)
    play_button.on_clicked(on_play)
    reset_button.on_clicked(on_reset)
    check.on_clicked(on_check)

    sliders["energy"].on_changed(on_energy_change)
    for key in ("height", "width", "count", "gap"):
        sliders[key].on_changed(on_potential_change)
    radio.on_clicked(on_mode_change)

    on_potential_change(None)  # 最初の1回を描く
    return fig, {
        "sliders": sliders,
        "radio": radio,
        "check": check,
        "play": play_button,
        "reset": reset_button,
        "timer": timer,
    }


if __name__ == "__main__":
    fig, widgets = build_viewer()
    plt.show()
