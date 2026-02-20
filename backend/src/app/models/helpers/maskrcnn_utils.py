import numpy as np


def iou(boxA, boxB) -> float:
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def merge_boxes(boxes) -> list:
    boxes = np.array(boxes)
    return [boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()]


def merge_masks(masks, indices) -> np.ndarray:
    merged = masks[indices[0]].copy()
    for idx in indices[1:]:
        merged = np.maximum(merged, masks[idx])
    return merged


def group_boxes_and_masks(results, strong_thresh=0.95, min_thresh=0.5,
                           child_thresh=0.5, iou_thresh=0.1) -> dict:
    boxes = np.array(results['boxes'])
    scores = np.array(results['scores'])
    masks = results['masks']

    if len(boxes) == 0:
        return results

    merged_boxes, merged_scores, merged_masks = [], [], []

    for idx in np.where(scores >= min_thresh)[0]:
        group_boxes, group_indices = [boxes[idx]], [idx]

        for c_idx in range(len(boxes)):
            if c_idx == idx or scores[c_idx] < child_thresh:
                continue
            if iou(boxes[idx], boxes[c_idx]) > iou_thresh:
                group_boxes.append(boxes[c_idx])
                group_indices.append(c_idx)

        if len(group_indices) > 1 or scores[idx] >= strong_thresh:
            merged_boxes.append(merge_boxes(group_boxes))
            merged_scores.append(scores[idx])
            merged_masks.append(merge_masks(masks, group_indices))

    return {
        'boxes': np.array(merged_boxes),
        'scores': np.array(merged_scores),
        'masks': np.array(merged_masks) if merged_masks else np.array([]),
    }
