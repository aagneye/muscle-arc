import numpy as np

from muscle_arc.geometry.exact_scale import decode_exact_scale


def test_fixed_644x1088_layout() -> None:
    img = np.zeros((644, 1088), dtype=np.uint8)
    s = decode_exact_scale(img, "IMG_00100.tif")
    assert s.ok
    assert abs(s.px_per_cm - 126.1) < 1e-6
    assert abs(s.mm_per_pixel - 10.0 / 126.1) < 1e-9
    assert s.box == (140, 0, 947, 643)


def test_bottom_ruler_853_layout() -> None:
    img = np.zeros((853, 1069), dtype=np.uint8)
    img[-5, 100] = 170
    img[-5, 934] = 170
    s = decode_exact_scale(img, "IMG_00200.tif")
    assert s.ok
    assert abs(s.px_per_cm - (934 - 100) / 5.0) < 1e-9
    assert s.box == (0, 0, 1069, 853)


def test_unknown_format_fails_closed() -> None:
    img = np.zeros((300, 400), dtype=np.uint8)
    s = decode_exact_scale(img, "IMG_00001.tif")
    assert not s.ok
    assert np.isnan(s.mm_per_pixel)
    assert s.box == (0, 0, 400, 300)


def test_siemens_right_ruler_synthetic() -> None:
    img = np.zeros((800, 1200), dtype=np.uint8)
    img[87, 1147:1157] = 175
    # 9 half-cm ticks spaced 68 px at column 1150 -> depth 4.5 cm, 136 px/cm,
    # and the 4.5 cm sector box is 778 px = 5.72 cm wide (passes the width check)
    for k in range(10):
        img[87 + 68 * k, 1148:1153] = 175
    s = decode_exact_scale(img, "IMG_00010.tif")
    assert s.ok
    assert abs(s.px_per_cm - 136.0) < 1e-6
    assert s.box == (211, 91, 989, 87 + 612)
