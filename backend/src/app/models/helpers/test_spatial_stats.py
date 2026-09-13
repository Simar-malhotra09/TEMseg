"""Tests for per-particle nearest-neighbor and border-distance stats.

Particles are drawn as filled rectangles so centroid, edge-to-edge, and
border distances are exact by construction.
"""

import math
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.models.helpers.compute_stats import compute_stats_from_instances


def build_instances(labeled: np.ndarray) -> list[dict]:
    """Build instance dicts from a labeled mask (values 1..n are particle ids)."""
    instances = []
    for label in [int(v) for v in np.unique(labeled) if v != 0]:
        comp = (labeled == label).astype(np.uint8)
        ys, xs = np.nonzero(comp)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        contours, _ = cv.findContours(comp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        largest = max(contours, key=cv.contourArea)
        instances.append(
            {
                "id": label,
                "contour": largest.squeeze().tolist(),
                "bbox": {"x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1},
                "area": float(comp.sum()),
            }
        )
    return instances


def make_stats(labeled: np.ndarray, pixel_size=None, pixel_unit=None) -> dict:
    instances = build_instances(labeled)
    binary = (labeled > 0).astype(np.uint8)
    return compute_stats_from_instances(
        instances,
        binary,
        pixel_size=pixel_size,
        pixel_unit=pixel_unit,
        labeled_mask=labeled,
    )


def draw_rect(labeled: np.ndarray, label: int, x, y, w, h):
    labeled[y : y + h, x : x + w] = label


def test_nearest_neighbor_and_border():
    print("\n=== nearest neighbor + border distance, two rectangles ===")
    h_img, w_img = 512, 512
    labeled = np.zeros((h_img, w_img), dtype=np.uint16)
    draw_rect(labeled, 1, 20, 50, 30, 40)  # centroid (34.5, 69.5)
    draw_rect(labeled, 2, 300, 60, 30, 40)  # centroid (314.5, 79.5)

    stats = make_stats(labeled)
    assert stats["particle_count"] == 2

    expected_nn = math.hypot(314.5 - 34.5, 79.5 - 69.5)  # centroid to centroid
    by_id = {p["id"]: p for p in stats["particles"]}

    for pid in (1, 2):
        nn = by_id[pid]["nearest_neighbor_px"]
        assert nn is not None, f"particle {pid}: expected NN distance"
        assert abs(nn - expected_nn) < 0.01, f"particle {pid}: NN {nn} != {expected_nn}"

    # rect 1: x in [20, 49], y in [50, 89] -> min(20, 511-49, 50, 511-89) = 20
    assert by_id[1]["border_distance_px"] == 20.0
    # rect 2: x in [300, 329], y in [60, 99] -> min(300, 182, 60, 412) = 60
    assert by_id[2]["border_distance_px"] == 60.0

    # no scale given -> no real-unit fields
    assert "nearest_neighbor_real" not in by_id[1]
    assert "border_distance_real" not in by_id[1]

    print(f"  NN: {by_id[1]['nearest_neighbor_px']:.2f} px (expected {expected_nn:.2f})")
    print(f"  Border: {by_id[1]['border_distance_px']}, {by_id[2]['border_distance_px']} (expected 20, 60)")
    print("  ✓ PASSED")


def test_real_units_scale():
    print("\n=== real-unit scaling (0.5 nm/px) ===")
    labeled = np.zeros((512, 512), dtype=np.uint16)
    draw_rect(labeled, 1, 20, 50, 30, 40)
    draw_rect(labeled, 2, 300, 60, 30, 40)

    stats = make_stats(labeled, pixel_size=0.5, pixel_unit="nm")
    assert stats["has_scale"] and stats["unit"] == "nm"

    by_id = {p["id"]: p for p in stats["particles"]}
    assert abs(by_id[1]["border_distance_real"] - 10.0) < 1e-9
    assert abs(by_id[2]["border_distance_real"] - 30.0) < 1e-9
    assert by_id[1]["nearest_neighbor_real"] is not None
    assert abs(by_id[1]["nearest_neighbor_real"] - by_id[1]["nearest_neighbor_px"] * 0.5) < 1e-9

    print("  ✓ PASSED")


def test_single_particle_has_no_neighbor():
    print("\n=== single particle: NN is null, border still computed ===")
    labeled = np.zeros((512, 512), dtype=np.uint16)
    draw_rect(labeled, 1, 100, 100, 30, 30)

    stats = make_stats(labeled)
    assert stats["particle_count"] == 1
    p = stats["particles"][0]
    assert p["nearest_neighbor_px"] is None
    assert p["border_distance_px"] == 100.0

    print("  ✓ PASSED")


def test_three_in_a_row():
    print("\n=== three rectangles in a row: pairwise NN + edge-touching particle ===")
    labeled = np.zeros((512, 512), dtype=np.uint16)
    draw_rect(labeled, 1, 0, 0, 20, 20)    # touches top-left corner, border 0
    draw_rect(labeled, 2, 50, 40, 20, 20)  # NN from particle 1: 50 px
    draw_rect(labeled, 3, 120, 40, 20, 20)  # NN from particle 2: 70 px

    stats = make_stats(labeled)
    by_id = {p["id"]: p for p in stats["particles"]}

    nn12 = math.hypot(50, 40)  # centroids (9.5, 9.5) -> (59.5, 49.5)
    assert abs(by_id[1]["nearest_neighbor_px"] - nn12) < 0.01
    assert abs(by_id[2]["nearest_neighbor_px"] - nn12) < 0.01
    assert by_id[3]["nearest_neighbor_px"] == 70.0
    assert by_id[1]["border_distance_px"] == 0.0
    assert by_id[2]["border_distance_px"] == 40.0
    assert by_id[3]["border_distance_px"] == 40.0

    print("  ✓ PASSED")


def test_fallback_without_labeled_mask():
    print("\n=== fallback path: no labeled mask, polygon centroid + contour border ===")
    mask = np.zeros((512, 512), dtype=np.uint8)
    sq1 = [[20, 50], [40, 50], [40, 70], [20, 70]]  # centroid (30, 60), border 20
    sq2 = [[100, 50], [120, 50], [120, 70], [100, 70]]  # centroid (110, 60), border 50
    for s in (sq1, sq2):
        cv.fillPoly(mask, [np.array(s, dtype=np.int32)], 255)
    instances = [
        {"id": 1, "contour": sq1, "bbox": {"x": 20, "y": 50, "w": 20, "h": 20}, "area": 400.0},
        {"id": 2, "contour": sq2, "bbox": {"x": 100, "y": 50, "w": 20, "h": 20}, "area": 400.0},
    ]

    stats = compute_stats_from_instances(instances, mask, labeled_mask=None)
    by_id = {p["id"]: p for p in stats["particles"]}
    assert abs(by_id[1]["nearest_neighbor_px"] - 80.0) < 0.01
    assert by_id[1]["border_distance_px"] == 20.0
    assert by_id[2]["border_distance_px"] == 50.0

    print("  ✓ PASSED")


def run_all():
    tests = [
        test_nearest_neighbor_and_border,
        test_real_units_scale,
        test_single_particle_has_no_neighbor,
        test_three_in_a_row,
        test_fallback_without_labeled_mask,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"  {passed}/{passed + failed} tests passed")
    if failed:
        print(f"  {failed} FAILED")
    print(f"{'=' * 50}")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
