"""
test_barrier_model.py — barrier_model.py（障壁の並びを作り変える部分）が正しく動くかを確かめる

実行: .venv/bin/python -m pytest -v
"""

import numpy as np
import pytest

import barrier_model as bm
import scattering as sc

B = bm.Barrier
TOL = dict(tol_x=0.05, tol_v=0.05)  # マウスの許容距離（テストでは固定値）


def asymmetric():
    """テスト用: 高さ・幅・間隔がバラバラな3つの障壁"""
    return (B(0.0, 0.5, 1.0), B(1.5, 0.3, 0.6), B(2.3, 0.8, -0.4))


# ---------------------------------------------------------------------------
# 1. scattering.py の形への変換
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("count", [0, 1, 2, 3, 5])
def test_uniform_matches_barrier_array(count):
    """ひな形 uniform は、これまでの barrier_array と同じ境界・ポテンシャルになる"""
    boundaries, potentials = bm.to_potential(bm.uniform(count, 0.5, 0.2, 0.5))
    expected = sc.barrier_array(count, 0.5, 0.2, 0.5)
    assert np.allclose(boundaries, expected[0])
    assert potentials == expected[1]


def test_to_potential_asymmetric():
    """バラバラな障壁が「外(0), 障壁, 谷(0), 障壁, ..., 外(0)」の順に並ぶ"""
    boundaries, potentials = bm.to_potential(asymmetric())
    assert np.allclose(boundaries, (0.0, 0.5, 1.5, 1.8, 2.3, 3.1))
    assert potentials == (0.0, 1.0, 0.0, 0.6, 0.0, -0.4, 0.0)


def test_asymmetric_structure_conserves_flux():
    """バラバラな構造（井戸を含む）でも、scattering.py で R + T = 1 になる"""
    boundaries, potentials = bm.to_potential(asymmetric())
    for energy in (0.1, 0.55, 0.6, 1.2, 3.0):
        result = sc.solve_scattering(energy, boundaries, potentials)
        assert result.reflectance + result.transmittance == pytest.approx(1.0, abs=1e-10)


def test_overlap_is_rejected():
    """重なった並びは、理由付きのエラーになる"""
    with pytest.raises(ValueError, match="重なって"):
        bm.to_potential((B(0.0, 1.0, 1.0), B(0.5, 1.0, 1.0)))


# ---------------------------------------------------------------------------
# 2. マウスがどこを指しているか
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "x, v, expected",
    [
        (0.25, 1.0, bm.Hit(0, bm.PART_TOP)),  # 1つ目の上の辺
        (0.0, 0.5, bm.Hit(0, bm.PART_LEFT)),  # 1つ目の左の辺
        (0.5, 0.5, bm.Hit(0, bm.PART_RIGHT)),  # 1つ目の右の辺
        (0.25, 0.5, bm.Hit(0, bm.PART_BODY)),  # 1つ目の中ほど
        (2.7, -0.4, bm.Hit(2, bm.PART_TOP)),  # 井戸の底（高さの辺）
        (2.7, -0.2, bm.Hit(2, bm.PART_BODY)),  # 井戸の中ほど
        (1.0, 0.5, None),  # 障壁の間
        (0.25, 2.0, None),  # 障壁の上の空中
    ],
)
def test_hit_test(x, v, expected):
    assert bm.hit_test(asymmetric(), x, v, **TOL) == expected


def test_hit_test_corner_picks_nearest_edge():
    """角の近くでは、近い方の辺を選ぶ"""
    assert bm.hit_test(asymmetric(), 0.01, 0.97, **TOL) == bm.Hit(0, bm.PART_LEFT)
    assert bm.hit_test(asymmetric(), 0.04, 0.99, **TOL) == bm.Hit(0, bm.PART_TOP)


# ---------------------------------------------------------------------------
# 3. ドラッグ
# ---------------------------------------------------------------------------
def test_drag_top_changes_only_height_with_snap():
    """上の辺: 高さだけが変わり、5 meV 刻みに吸着する"""
    result = bm.drag(asymmetric(), bm.Hit(1, bm.PART_TOP), dx=0.3, dv=0.1234)
    assert result[1] == B(1.5, 0.3, 0.725)
    assert result[0] == asymmetric()[0] and result[2] == asymmetric()[2]


def test_drag_left_edge_keeps_right_edge():
    """左の辺: 右端は動かず、幅が変わる"""
    result = bm.drag(asymmetric(), bm.Hit(1, bm.PART_LEFT), dx=-0.2, dv=0.0)
    assert result[1].left == pytest.approx(1.3)
    assert result[1].right == pytest.approx(1.8)


def test_drag_edges_stop_before_neighbors_and_zero_width():
    """辺は、隣と MIN_GAP、幅が MIN_WIDTH になるところで止まる（隣は押さない）"""
    left = bm.drag(asymmetric(), bm.Hit(1, bm.PART_LEFT), dx=-5.0, dv=0.0)
    assert left[1].left == pytest.approx(0.5 + bm.MIN_GAP)
    shrink = bm.drag(asymmetric(), bm.Hit(1, bm.PART_RIGHT), dx=-5.0, dv=0.0)
    assert shrink[1].width == pytest.approx(bm.MIN_WIDTH)
    right = bm.drag(asymmetric(), bm.Hit(1, bm.PART_RIGHT), dx=5.0, dv=0.0)
    assert right[1].right == pytest.approx(2.3 - bm.MIN_GAP)
    assert right[2] == asymmetric()[2]


def test_drag_body_moves_between_neighbors():
    """中ほど: 幅と高さはそのまま、両隣の間で止まる"""
    moved = bm.drag(asymmetric(), bm.Hit(1, bm.PART_BODY), dx=0.2, dv=0.5)
    assert moved[1] == B(1.7, 0.3, 0.6)
    far = bm.drag(asymmetric(), bm.Hit(1, bm.PART_BODY), dx=10.0, dv=0.0)
    assert far[1].right == pytest.approx(2.3 - bm.MIN_GAP)
    assert far[1].width == pytest.approx(0.3)


def test_drag_outer_barriers_are_free_outside():
    """両端の障壁は、外側には制限なく動かせる"""
    moved = bm.drag(asymmetric(), bm.Hit(0, bm.PART_BODY), dx=-3.0, dv=0.0)
    assert moved[0].left == pytest.approx(-3.0)


def test_drag_never_breaks_structure():
    """どこをどれだけドラッグしても、並びは常に正しい（重ならず、幅が正）"""
    rng = np.random.default_rng(0)
    for _ in range(500):
        hit = bm.Hit(int(rng.integers(3)), str(rng.choice([bm.PART_TOP, bm.PART_LEFT, bm.PART_RIGHT, bm.PART_BODY])))
        bm.validate(bm.drag(asymmetric(), hit, dx=rng.uniform(-5, 5), dv=rng.uniform(-3, 3)))


def test_drag_does_not_modify_input():
    """渡した並びは書き換えない（ドラッグ開始時の状態からやり直せるように）"""
    original = asymmetric()
    bm.drag(original, bm.Hit(0, bm.PART_BODY), dx=1.0, dv=0.0)
    assert original == asymmetric()


# ---------------------------------------------------------------------------
# 4. 数値入力・追加・削除
# ---------------------------------------------------------------------------
def test_set_width_shifts_right_neighbors():
    """幅を変えると右側の障壁がまとめてずれ、ほかの幅と間隔は変わらない"""
    result = bm.set_width(asymmetric(), 0, 1.5)
    assert result[0].width == 1.5
    assert [bm.gap_before(result, i) for i in (1, 2)] == pytest.approx([bm.gap_before(asymmetric(), i) for i in (1, 2)])
    bm.validate(result)


def test_set_gap_before_shifts_right_side():
    """間隔を変えると、その障壁から右がまとめてずれる"""
    result = bm.set_gap_before(asymmetric(), 1, 0.2)
    assert bm.gap_before(result, 1) == pytest.approx(0.2)
    assert bm.gap_before(result, 2) == pytest.approx(bm.gap_before(asymmetric(), 2))
    assert result[0] == asymmetric()[0]


def test_set_invalid_values_are_rejected():
    with pytest.raises(ValueError, match="幅"):
        bm.set_width(asymmetric(), 0, 0.0)
    with pytest.raises(ValueError, match="間隔"):
        bm.set_gap_before(asymmetric(), 1, -0.1)
    with pytest.raises(ValueError, match="左隣"):
        bm.set_gap_before(asymmetric(), 0, 0.5)


def test_append_and_remove():
    """追加は右端に同じ形・同じ間隔で、削除はほかを動かさない"""
    added = bm.append_barrier(asymmetric())
    assert len(added) == 4
    assert added[3].left == pytest.approx(3.1 + bm.gap_before(asymmetric(), 2))
    assert (added[3].width, added[3].height) == (0.8, -0.4)
    assert bm.remove_barrier(asymmetric(), 1) == (asymmetric()[0], asymmetric()[2])
    assert bm.append_barrier(()) == (B(0.0, 0.5, 1.0),)
