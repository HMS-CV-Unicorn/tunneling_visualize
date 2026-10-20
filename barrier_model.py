"""
barrier_model.py — 高さ・幅・間隔がバラバラな障壁の並びを表し、マウス操作や数値入力で作り変える

editor.py（画面）から使う「描画しない部分」。ここにある関数はすべて、障壁の並びを受け取って
新しい並びを返すだけで、渡された並びは書き換えない（元の並びが残るので、ドラッグ開始時の状態から
何度でもやり直せる）。画面なしで test_barrier_model.py から確かめられる。

単位は scattering.py と同じく、長さ nm・エネルギー eV。

障壁の並び（構造）は Barrier の tuple で、左から順に並んでいる。障壁と障壁の間と、両端の外側は V = 0。

        V
        ┌──┐          ┌────┐
        │h0│   ┌─┐    │ h2 │
    ────┘  └───┘h1└───┘    └────  x
        ↑  ↑   ↑ ↑
     left0 right0 ...           間隔 = 次の left − この right
"""

from dataclasses import dataclass, replace

# ドラッグしたときに値を吸着させる刻み（数値入力ではこの刻みに縛られない）
ENERGY_SNAP = 0.005  # 5 meV [eV]
LENGTH_SNAP = 0.05  # 0.5 Å [nm]

# 幅と間隔の最小値 [nm]。0 にすると境界が重なり、領域が消えてしまうので正の値で止める
MIN_WIDTH = LENGTH_SNAP
MIN_GAP = LENGTH_SNAP

# 障壁の個数の上限。力技の連立方程式は障壁の数に比例して大きくなり、T(E) の計算が遅くなるため
MAX_BARRIERS = 30

# マウスで障壁のどこをつかんだか
PART_TOP = "top"  # 上の辺 → 高さを変える
PART_LEFT = "left"  # 左の辺 → 左端だけ動かす（幅が変わる）
PART_RIGHT = "right"  # 右の辺 → 右端だけ動かす（幅が変わる）
PART_BODY = "body"  # 中ほど → 障壁ごと左右に動かす（両隣との間隔が変わる）


@dataclass(frozen=True)
class Barrier:
    """
    長方形の障壁1つ。left < x < left + width で V = height、その外は（ほかの障壁がなければ）V = 0。
    height < 0 なら井戸になる。frozen=True は「作った後に値を書き換えられない」という指定。
    """

    left: float  # 左端の位置 [nm]
    width: float  # 幅 [nm]
    height: float  # 高さ [eV]

    @property
    def right(self) -> float:
        """右端の位置 [nm]"""
        return self.left + self.width


@dataclass(frozen=True)
class Hit:
    """マウスが指している場所。index 番目の障壁の part（PART_TOP など）"""

    index: int
    part: str


# ---------------------------------------------------------------------------
# 作る・調べる
# ---------------------------------------------------------------------------
def uniform(count: int, height: float, width: float, gap: float) -> tuple[Barrier, ...]:
    """
    同じ高さ・幅の障壁を、同じ間隔で count 個並べる（ひな形）。左端の障壁は x = 0 から始まる。
    scattering.barrier_array と同じ並びになる。
    """
    if not 0 <= count <= MAX_BARRIERS:
        raise ValueError(f"障壁の個数は 0 以上 {MAX_BARRIERS} 以下にしてください（count = {count}）")
    barriers = tuple(Barrier(i * (width + gap), width, height) for i in range(count))
    validate(barriers)
    return barriers


def validate(barriers: tuple[Barrier, ...]) -> None:
    """幅が正で、左から順に重ならずに並んでいるかを確かめる。だめなら理由を書いた ValueError を出す"""
    if len(barriers) > MAX_BARRIERS:
        raise ValueError(f"障壁は {MAX_BARRIERS} 個までにしてください（いま {len(barriers)} 個）")
    for i, b in enumerate(barriers):
        if b.width <= 0:
            raise ValueError(f"{i + 1} 番目の障壁の幅が正ではありません（幅 = {b.width} nm）")
    for i, (a, b) in enumerate(zip(barriers, barriers[1:])):
        if b.left - a.right <= 0:
            raise ValueError(
                f"{i + 1} 番目と {i + 2} 番目の障壁が重なっています（間隔 = {b.left - a.right:.6g} nm）"
            )


def to_potential(barriers: tuple[Barrier, ...]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """
    scattering.solve_scattering に渡す (境界の位置 [nm], 各領域のポテンシャル [eV]) に変換する。
    領域は「外(0), 障壁0, 谷(0), 障壁1, ..., 外(0)」の順に交互になる。
    """
    validate(barriers)
    boundaries = tuple(x for b in barriers for x in (b.left, b.right))
    potentials = (0.0, *(v for b in barriers for v in (b.height, 0.0)))
    return boundaries, potentials


def gap_before(barriers: tuple[Barrier, ...], index: int) -> float | None:
    """index 番目の障壁の、左隣の障壁との間隔 [nm]。左端の障壁なら None（左隣がない）"""
    if index == 0:
        return None
    return barriers[index].left - barriers[index - 1].right


def hit_test(barriers: tuple[Barrier, ...], x: float, v: float, tol_x: float, tol_v: float) -> Hit | None:
    """
    位置 (x [nm], V [eV]) のマウスが、どの障壁のどこを指しているか。どこも指していなければ None。

    tol_x, tol_v は「辺から これだけ離れていても辺をつかんだとみなす」距離（画面の数ピクセル分を
    データの単位に直したもの。画面側で計算して渡す）。
    角の近くなど複数の辺が候補になるときは、許容距離を基準にして一番近い辺を選ぶ。
    辺のどれにも近くなく、障壁の内側なら「中ほど（PART_BODY）」。
    """
    best = None  # (距離 / 許容距離, Hit)
    for i, b in enumerate(barriers):
        v_low, v_high = min(0.0, b.height), max(0.0, b.height)  # 井戸（height < 0）なら下向きの長方形
        within_x = b.left - tol_x <= x <= b.right + tol_x
        within_v = v_low - tol_v <= v <= v_high + tol_v
        candidates = []
        if within_x and abs(v - b.height) <= tol_v:
            candidates.append((abs(v - b.height) / tol_v, Hit(i, PART_TOP)))
        if within_v and abs(x - b.left) <= tol_x:
            candidates.append((abs(x - b.left) / tol_x, Hit(i, PART_LEFT)))
        if within_v and abs(x - b.right) <= tol_x:
            candidates.append((abs(x - b.right) / tol_x, Hit(i, PART_RIGHT)))
        for distance, hit in candidates:
            if best is None or distance < best[0]:
                best = (distance, hit)
    if best is not None:
        return best[1]
    for i, b in enumerate(barriers):
        if b.left < x < b.right and min(0.0, b.height) < v < max(0.0, b.height):
            return Hit(i, PART_BODY)
    return None


# ---------------------------------------------------------------------------
# マウスのドラッグ
# ---------------------------------------------------------------------------
def snap(value: float, step: float) -> float:
    """value を step の整数倍に丸める（round(..., 10) は 0.30000000000000004 のような誤差を消すため）"""
    return round(round(value / step) * step, 10)


def drag(barriers: tuple[Barrier, ...], hit: Hit, dx: float, dv: float) -> tuple[Barrier, ...]:
    """
    ドラッグを反映した新しい並びを返す。barriers はドラッグを始めたときの並び、
    (dx [nm], dv [eV]) はドラッグを始めた点からのマウスの移動量。

    毎回「始めたときの並び + 移動量」から計算するので、途中で制限に当たって止まっても、
    マウスを戻せば元の位置に戻る。値は ENERGY_SNAP・LENGTH_SNAP の刻みに吸着させ、
    幅 MIN_WIDTH・間隔 MIN_GAP より小さくなる手前で止める（隣の障壁は押さない）。
    """
    b = barriers[hit.index]
    prev_right = barriers[hit.index - 1].right if hit.index > 0 else None
    next_left = barriers[hit.index + 1].left if hit.index + 1 < len(barriers) else None

    if hit.part == PART_TOP:
        new = replace(b, height=snap(b.height + dv, ENERGY_SNAP))
    elif hit.part == PART_LEFT:
        low = -float("inf") if prev_right is None else prev_right + MIN_GAP
        new_left = clamp(snap(b.left + dx, LENGTH_SNAP), low, b.right - MIN_WIDTH)
        new = Barrier(new_left, b.right - new_left, b.height)
    elif hit.part == PART_RIGHT:
        high = float("inf") if next_left is None else next_left - MIN_GAP
        new_right = clamp(snap(b.right + dx, LENGTH_SNAP), b.left + MIN_WIDTH, high)
        new = replace(b, width=new_right - b.left)
    elif hit.part == PART_BODY:
        low = -float("inf") if prev_right is None else prev_right + MIN_GAP
        high = float("inf") if next_left is None else next_left - MIN_GAP - b.width
        new = replace(b, left=clamp(snap(b.left + dx, LENGTH_SNAP), low, high))
    else:
        raise ValueError(f"つかんだ場所 part = {hit.part!r} は {PART_TOP}, {PART_LEFT}, {PART_RIGHT}, {PART_BODY} のどれかです")
    return barriers[: hit.index] + (new,) + barriers[hit.index + 1 :]


def clamp(value: float, low: float, high: float) -> float:
    """value を [low, high] の範囲に収める"""
    return min(max(value, low), high)


# ---------------------------------------------------------------------------
# 数値入力・追加・削除
# ---------------------------------------------------------------------------
def set_height(barriers: tuple[Barrier, ...], index: int, height: float) -> tuple[Barrier, ...]:
    """index 番目の障壁の高さを height [eV] にする"""
    return barriers[:index] + (replace(barriers[index], height=height),) + barriers[index + 1 :]


def set_width(barriers: tuple[Barrier, ...], index: int, width: float) -> tuple[Barrier, ...]:
    """
    index 番目の障壁の幅を width [nm] にする。左端は動かさず、右端が動く。
    右側の障壁はすべて同じだけずらすので、ほかの障壁の幅と間隔は変わらない
    （ドラッグと違って、数値で入れた値がそのまま通るように）。
    """
    if width <= 0:
        raise ValueError(f"幅は 0 より大きくしてください（幅 = {width} nm）")
    shift = width - barriers[index].width
    return (
        barriers[:index]
        + (replace(barriers[index], width=width),)
        + tuple(replace(b, left=b.left + shift) for b in barriers[index + 1 :])
    )


def set_gap_before(barriers: tuple[Barrier, ...], index: int, gap: float) -> tuple[Barrier, ...]:
    """
    index 番目の障壁の、左隣との間隔を gap [nm] にする。index 番目より右の障壁がすべて同じだけずれる
    （それより右の幅と間隔は変わらない）。左端の障壁には左隣がないので、間隔は決められない。
    """
    if index == 0:
        raise ValueError("左端の障壁には左隣がないので、間隔は決められません")
    if gap <= 0:
        raise ValueError(f"間隔は 0 より大きくしてください（間隔 = {gap} nm）")
    shift = gap - gap_before(barriers, index)
    return barriers[:index] + tuple(replace(b, left=b.left + shift) for b in barriers[index:])


def append_barrier(barriers: tuple[Barrier, ...]) -> tuple[Barrier, ...]:
    """
    右端に障壁を1つ足す。右端の障壁と同じ高さ・幅で、間隔は最後の間隔と同じ（障壁が1個なら 1 nm）。
    障壁が1つもなければ、高さ 1 eV・幅 0.5 nm の障壁を x = 0 に置く。
    """
    if len(barriers) >= MAX_BARRIERS:
        raise ValueError(f"障壁は {MAX_BARRIERS} 個までです")
    if not barriers:
        return (Barrier(0.0, 0.5, 1.0),)
    last = barriers[-1]
    gap = gap_before(barriers, len(barriers) - 1) if len(barriers) > 1 else 1.0
    return barriers + (replace(last, left=last.right + gap),)


def remove_barrier(barriers: tuple[Barrier, ...], index: int) -> tuple[Barrier, ...]:
    """index 番目の障壁を取り除く（ほかの障壁は動かさない）"""
    return barriers[:index] + barriers[index + 1 :]
