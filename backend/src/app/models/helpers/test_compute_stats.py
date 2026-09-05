"""
Creates synthetic binary masks with particles of known size, shape, and count,
then checks that compute_stats returns correct values.

"""

import numpy as np
import cv2 as cv
import math
import sys
from pathlib import Path

# allow running from backend/src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.models.helpers.compute_stats import compute_stats, _equivalent_diameter_px


# Helpers to generate synthetic masks

def make_blank(h=512, w=512) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def draw_circle(mask: np.ndarray, cx: int, cy: int, radius: int):
    """Draw a filled circle. Returns the expected area."""
    cv.circle(mask, (cx, cy), radius, 255, -1)
    # actual pixel area won't be exactly pi*r^2 due to rasterization
    # so we count pixels directly as the "truth"
    return None  # we'll count from the mask


def draw_ellipse(mask: np.ndarray, cx: int, cy: int, a: int, b: int, angle: float = 0):
    """Draw a filled ellipse with semi-axes a, b."""
    cv.ellipse(mask, (cx, cy), (a, b), angle, 0, 360, 255, -1)


def draw_rectangle(mask: np.ndarray, x: int, y: int, w: int, h: int):
    """Draw a filled rectangle."""
    cv.rectangle(mask, (x, y), (x + w, y + h), 255, -1)


def count_components(mask: np.ndarray) -> int:
    """Count non-background connected components."""
    n, _ = cv.connectedComponents((mask > 0).astype(np.uint8))
    return n - 1


# Test 1: Single circle — check count, area, diameter, circularity

def test_single_circle():
    print("\n=== Test 1: Single circle (r=50) ===")
    mask = make_blank(512, 512)
    draw_circle(mask, 256, 256, 50)

    # ground truth: count pixels
    true_area = np.count_nonzero(mask)
    true_diameter = _equivalent_diameter_px(true_area)

    stats = compute_stats(mask)

    assert stats["particle_count"] == 1, f"Expected 1 particle, got {stats['particle_count']}"

    measured_area = stats["particles"][0]["area_px"]
    area_err = abs(measured_area - true_area) / true_area
    print(f"  Area: expected={true_area}, got={measured_area:.1f}, error={area_err:.4f}")
    assert area_err < 0.02, f"Area error too high: {area_err:.4f}"

    measured_diam = stats["particles"][0]["diameter_px"]
    diam_err = abs(measured_diam - true_diameter) / true_diameter
    print(f"  Eq. Diameter: expected={true_diameter:.2f}, got={measured_diam:.2f}, error={diam_err:.4f}")
    assert diam_err < 0.02, f"Diameter error too high: {diam_err:.4f}"

    circ = stats["particles"][0]["circularity"]
    print(f"  Circularity: {circ:.4f} (expected ~1.0)")
    assert circ > 0.95, f"Circularity too low for circle: {circ}"

    ar = stats["particles"][0]["aspect_ratio"]
    print(f"  Aspect ratio: {ar:.4f} (expected ~1.0)")
    assert ar < 1.15, f"Aspect ratio too high for circle: {ar}"

    print("  ✓ PASSED")


# Test 2: Multiple circles of same size — check count and consistency

def test_multiple_same_circles():
    print("\n=== Test 2: 5 circles of same size (r=30) ===")
    mask = make_blank(512, 512)
    positions = [(100, 100), (300, 100), (100, 300), (300, 300), (200, 200)]
    for cx, cy in positions:
        draw_circle(mask, cx, cy, 30)

    stats = compute_stats(mask)

    assert stats["particle_count"] == 5, f"Expected 5 particles, got {stats['particle_count']}"

    areas = [p["area_px"] for p in stats["particles"]]
    area_std = np.std(areas)
    area_mean = np.mean(areas)
    cv_area = area_std / area_mean  # coefficient of variation
    print(f"  Count: {stats['particle_count']}")
    print(f"  Areas: mean={area_mean:.1f}, std={area_std:.1f}, CV={cv_area:.4f}")
    assert cv_area < 0.05, f"Area variation too high for identical circles: CV={cv_area}"

    print(f"  Size stats: mean={stats['size_stats']['diameter_mean']:.2f}, std={stats['size_stats']['diameter_std']:.2f}")
    assert stats["size_stats"]["diameter_std"] < 2.0, "Std too high for identical circles"

    print("  ✓ PASSED")


# Test 3: Known pixel scale — check real-unit conversion

def test_pixel_scale_conversion():
    print("\n=== Test 3: Pixel scale conversion (0.5 nm/px) ===")
    mask = make_blank(512, 512)
    draw_circle(mask, 256, 256, 40)

    pixel_size = 0.5  # nm/px

    stats = compute_stats(mask, pixel_size=pixel_size, pixel_unit="nm")

    assert stats["has_scale"] == True
    assert stats["unit"] == "nm"

    area_px = stats["particles"][0]["area_px"]
    area_real = stats["particles"][0]["area_real"]
    expected_area_real = area_px * pixel_size * pixel_size

    err = abs(area_real - expected_area_real) / expected_area_real
    print(f"  Area px: {area_px:.1f}")
    print(f"  Area real: {area_real:.2f} nm² (expected {expected_area_real:.2f})")
    print(f"  Error: {err:.6f}")
    assert err < 1e-6, f"Scale conversion error: {err}"

    diam_px = stats["particles"][0]["diameter_px"]
    diam_real = stats["particles"][0]["diameter_real"]
    expected_diam_real = diam_px * pixel_size
    err_d = abs(diam_real - expected_diam_real) / expected_diam_real
    print(f"  Diameter px: {diam_px:.2f}, real: {diam_real:.2f} nm (expected {expected_diam_real:.2f})")
    assert err_d < 1e-6, f"Diameter scale error: {err_d}"

    print("  ✓ PASSED")


# Test 4: Elongated rectangle — check aspect ratio and shape classification

def test_elongated_rectangle():
    print("\n=== Test 4: Elongated rectangle (20x100) ===")
    mask = make_blank(512, 512)
    draw_rectangle(mask, 200, 200, 20, 100)

    stats = compute_stats(mask)

    assert stats["particle_count"] == 1

    p = stats["particles"][0]
    print(f"  Aspect ratio: {p['aspect_ratio']:.2f} (expected ~5.0)")
    assert p["aspect_ratio"] > 3.0, f"Aspect ratio too low for 20x100 rect: {p['aspect_ratio']}"

    print(f"  Circularity: {p['circularity']:.4f} (expected <0.8)")
    assert p["circularity"] < 0.85, f"Circularity too high for rectangle: {p['circularity']}"

    print(f"  Shape: {p['shape']} (expected 'elongated')")
    assert p["shape"] == "elongated", f"Expected 'elongated', got '{p['shape']}'"

    # area should be close to 20*100 = 2000
    true_area = 20 * 100
    area_err = abs(p["area_px"] - true_area) / true_area
    print(f"  Area: {p['area_px']:.0f} (expected ~{true_area}, error={area_err:.4f})")
    assert area_err < 0.05, f"Area error: {area_err}"

    print("  ✓ PASSED")


# Test 5: Mixed shapes — check shape distribution

def test_shape_distribution():
    print("\n=== Test 5: Mixed shapes (3 circles + 2 rods) ===")
    mask = make_blank(800, 800)

    # 3 circles
    draw_circle(mask, 100, 100, 40)
    draw_circle(mask, 300, 100, 40)
    draw_circle(mask, 500, 100, 40)

    # 2 elongated rectangles
    draw_rectangle(mask, 100, 300, 15, 120)
    draw_rectangle(mask, 300, 300, 15, 120)

    stats = compute_stats(mask)

    print(f"  Count: {stats['particle_count']}")
    assert stats["particle_count"] == 5, f"Expected 5, got {stats['particle_count']}"

    sd = stats["shape_distribution"]
    print(f"  Shape dist: {sd}")

    assert "circular" in sd, "Missing 'circular' in shape distribution"
    assert "elongated" in sd, "Missing 'elongated' in shape distribution"
    assert sd["circular"]["count"] == 3, f"Expected 3 circular, got {sd['circular']['count']}"
    assert sd["elongated"]["count"] == 2, f"Expected 2 elongated, got {sd['elongated']['count']}"

    print("  ✓ PASSED")


# Test 6: Coverage calculation

def test_coverage():
    print("\n=== Test 6: Coverage ===")
    mask = make_blank(100, 100)
    # fill a 10x10 region = 100 pixels out of 10000 = 1% coverage
    draw_rectangle(mask, 0, 0, 10, 10)

    stats = compute_stats(mask)

    expected_coverage = 100 / 10000
    err = abs(stats["coverage"] - expected_coverage)
    print(f"  Coverage: {stats['coverage']:.4f} (expected {expected_coverage:.4f}, err={err:.6f})")
    assert err < 1e-6, f"Coverage error: {err}"

    print("  ✓ PASSED")


# Test 7: Empty mask

def test_empty_mask():
    print("\n=== Test 7: Empty mask ===")
    mask = make_blank(256, 256)

    stats = compute_stats(mask)

    assert stats["particle_count"] == 0
    assert stats["coverage"] == 0.0
    assert len(stats["particles"]) == 0
    print(f"  Count: {stats['particle_count']}, coverage: {stats['coverage']}")

    print("  ✓ PASSED")


# Test 8: Perimeter sanity check (circle perimeter ≈ 2πr)

def test_perimeter():
    print("\n=== Test 8: Perimeter of circle (r=60) ===")
    mask = make_blank(512, 512)
    draw_circle(mask, 256, 256, 60)

    stats = compute_stats(mask)
    p = stats["particles"][0]

    expected_perim = 2 * math.pi * 60
    perim_err = abs(p["perimeter_px"] - expected_perim) / expected_perim
    print(f"  Perimeter: {p['perimeter_px']:.1f} (expected ~{expected_perim:.1f}, error={perim_err:.4f})")
    # rasterization causes perimeter to be slightly larger, allow 5%
    assert perim_err < 0.05, f"Perimeter error: {perim_err}"

    print("  ✓ PASSED")


# Test 9: Distribution fitting sanity

def test_distribution_fitting():
    print("\n=== Test 9: Distribution fitting (20 circles, varied radii) ===")
    mask = make_blank(1024, 1024)
    np.random.seed(42)

    radii = np.random.lognormal(mean=3.0, sigma=0.3, size=20).astype(int)
    radii = np.clip(radii, 10, 80)

    placed = 0
    for r in radii:
        for _ in range(50):  # try random positions
            cx = np.random.randint(r + 5, 1024 - r - 5)
            cy = np.random.randint(r + 5, 1024 - r - 5)
            # check no overlap with existing
            roi = mask[cy - r:cy + r, cx - r:cx + r]
            if np.any(roi):
                continue
            draw_circle(mask, cx, cy, int(r))
            placed += 1
            break

    print(f"  Placed {placed} circles with lognormal radii")

    stats = compute_stats(mask)

    print(f"  Particle count: {stats['particle_count']}")
    assert stats["particle_count"] == placed, f"Expected {placed}, got {stats['particle_count']}"

    fits = stats.get("distribution_fits_diameter") or stats.get("distribution_fits")
    if fits and fits.get("reliable"):
        print(f"  Best fit model: {fits['best_model']}")
        print(f"  Fits: {list(fits['fits'].keys())}")
        for model, f in fits["fits"].items():
            print(f"    {model}: params={f['params']}, KS p={f['ks_pvalue']:.4f}")

        # lognormal should be best or at least a good fit since we generated lognormal
        if "lognormal" in fits["fits"]:
            ln_p = fits["fits"]["lognormal"]["ks_pvalue"]
            print(f"  Lognormal KS p-value: {ln_p:.4f} (should be >0.05)")
    else:
        print(f"  Distribution fitting: {fits}")

    print("  ✓ PASSED")


# Run all

def run_all():
    tests = [
        test_single_circle,
        test_multiple_same_circles,
        test_pixel_scale_conversion,
        test_elongated_rectangle,
        test_shape_distribution,
        test_coverage,
        test_empty_mask,
        test_perimeter,
        test_distribution_fitting,
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
    else:
        print("  All tests passed ✓")
    print(f"{'=' * 50}")

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
