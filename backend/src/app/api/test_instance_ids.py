"""Tests for spatially ordered instance ids in extract_instances."""

import sys
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.api.instances import extract_instances

H = W = 512


def draw_rect(mask: np.ndarray, x, y, w, h):
    cv.rectangle(mask, (x, y), (x + w - 1, y + h - 1), 255, -1)


def ids_by_position(instances: list[dict]) -> dict[tuple[int, int], int]:
    return {(i["bbox"]["x"], i["bbox"]["y"]): i["id"] for i in instances}


def test_connected_component_path():
    print("\n=== spatial ids: ndimage.label path ===")
    mask = np.zeros((H, W), dtype=np.uint8)
    draw_rect(mask, 300, 100, 20, 20)  # top-right
    draw_rect(mask, 50, 300, 20, 20)  # bottom-left
    draw_rect(mask, 50, 100, 20, 20)  # top-left

    instances, labeled = extract_instances(mask, Path("/tmp"), save=False)
    by_pos = ids_by_position(instances)
    assert by_pos[(50, 100)] == 1, f"top-left should be 1, got {by_pos}"
    assert by_pos[(300, 100)] == 2, f"top-right should be 2, got {by_pos}"
    assert by_pos[(50, 300)] == 3, f"bottom-left should be 3, got {by_pos}"

    # labeled mask must carry the same values
    assert set(np.unique(labeled)) <= {0, 1, 2, 3}
    assert np.sum(labeled == 1) == 400

    print("  ids:", by_pos)
    print("  ✓ PASSED")


def test_labels_map_path():
    print("\n=== spatial ids: model labels map path (detection-order ids renumbered) ===")
    mask = np.zeros((H, W), dtype=np.uint8)
    labels = np.zeros((H, W), dtype=np.int32)
    # arbitrary detection-order ids
    draw_rect(mask, 300, 100, 20, 20)  # top-right, detection id 7
    draw_rect(labels, 300, 100, 20, 20)
    labels[100:120, 300:320] = 7
    draw_rect(mask, 50, 300, 20, 20)  # bottom-left, detection id 3
    labels[300:320, 50:70] = 3
    draw_rect(mask, 50, 100, 20, 20)  # top-left, detection id 42
    labels[100:120, 50:70] = 42

    instances, labeled = extract_instances(mask, Path("/tmp"), save=False, labels=labels)
    by_pos = ids_by_position(instances)
    assert by_pos[(50, 100)] == 1, f"top-left should be 1, got {by_pos}"
    assert by_pos[(300, 100)] == 2, f"top-right should be 2, got {by_pos}"
    assert by_pos[(50, 300)] == 3, f"bottom-left should be 3, got {by_pos}"

    print("  ids:", by_pos)
    print("  ✓ PASSED")


def test_single_blob_untouched():
    print("\n=== spatial ids: single blob stays id 1 ===")
    mask = np.zeros((H, W), dtype=np.uint8)
    draw_rect(mask, 200, 200, 20, 20)
    instances, _ = extract_instances(mask, Path("/tmp"), save=False)
    assert [i["id"] for i in instances] == [1]

    print("  ✓ PASSED")


def run_all():
    tests = [
        test_connected_component_path,
        test_labels_map_path,
        test_single_blob_untouched,
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
