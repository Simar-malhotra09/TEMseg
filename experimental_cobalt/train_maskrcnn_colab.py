"""
Fine-tune the TEMseg MaskRCNN on the experimental_cobalt dataset.

Colab usage:
    !pip install torch==2.5.1 torchvision==0.20.1 opencv-python-headless numpy
    !python train_maskrcnn_colab.py --data-dir /content/experimental_cobalt --output-dir /content/output

Output is a plain state_dict .pth, compatible with backend/src/app/models/impls/maskrcnn.py's
model.load_state_dict(checkpoint) loader — copy it to backend/weights/maskrcnn_best_model.pth.
"""

import argparse
import json
import random
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

NUM_CLASSES = 2  # background + particle
MIN_BOX_AREA = 20.0
MIN_BOX_SIDE = 4.0


class Target(TypedDict):
    boxes: torch.Tensor
    labels: torch.Tensor
    masks: torch.Tensor
    image_id: torch.Tensor
    area: torch.Tensor
    iscrowd: torch.Tensor


def build_model(num_classes: int = NUM_CLASSES) -> torch.nn.Module:
    """Mirrors MaskRCNN._build_model in backend/src/app/models/impls/maskrcnn.py
    so the resulting state_dict loads back into the app unchanged."""
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )
    return model


def load_grayscale_rgb(image_path: Path) -> np.ndarray:
    """Matches MaskRCNN.load_image: grayscale -> 3ch replicate -> float32 [0,1]."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = np.stack([img] * 3, axis=-1)
    return img.astype(np.float32) / 255.0


def polygon_to_mask(polygon: list[float], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array(polygon, dtype=np.int32).reshape(-1, 2)
    cv2.fillPoly(mask, [points], 1)
    return mask


class ParticleDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        images: list[dict],
        annotations_by_image: dict[int, list[dict]],
        train: bool,
    ) -> None:
        self.image_dir = image_dir
        self.images = images
        self.annotations_by_image = annotations_by_image
        self.train = train

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Target]:
        image_info = self.images[idx]
        height, width = image_info["height"], image_info["width"]
        image = load_grayscale_rgb(self.image_dir / image_info["file_name"])

        boxes: list[list[float]] = []
        masks: list[np.ndarray] = []
        for ann in self.annotations_by_image.get(image_info["id"], []):
            x, y, w, h = ann["bbox"]
            if ann["area"] < MIN_BOX_AREA or min(w, h) < MIN_BOX_SIDE:
                continue
            boxes.append([x, y, x + w, y + h])
            masks.append(polygon_to_mask(ann["segmentation"][0], height, width))

        if self.train and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1, :])
            masks = [np.ascontiguousarray(m[:, ::-1]) for m in masks]
            boxes = [[width - x2, y1, width - x1, y2] for x1, y1, x2, y2 in boxes]
        if self.train and random.random() < 0.5:
            image = np.ascontiguousarray(image[::-1, :, :])
            masks = [np.ascontiguousarray(m[::-1, :]) for m in masks]
            boxes = [[x1, height - y2, x2, height - y1] for x1, y1, x2, y2 in boxes]

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        masks_t = (
            torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            if masks
            else torch.zeros((0, height, width), dtype=torch.uint8)
        )
        area_t = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])

        target: Target = {
            "boxes": boxes_t,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "masks": masks_t,
            "image_id": torch.tensor([image_info["id"]]),
            "area": area_t,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        image_t = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        return image_t, target


def collate_fn(batch: list[tuple[torch.Tensor, Target]]) -> tuple[list, list]:
    return tuple(zip(*batch))


def split_dataset(
    images: list[dict], val_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    shuffled = images.copy()
    random.Random(seed).shuffle(shuffled)
    num_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[num_val:], shuffled[:num_val]


def load_annotations(json_path: Path) -> tuple[list[dict], dict[int, list[dict]]]:
    with open(json_path) as f:
        data = json.load(f)
    annotations_by_image: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)
    return data["images"], annotations_by_image


def warmup_lr_scheduler(
    optimizer: torch.optim.Optimizer, warmup_iters: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def f(step: int) -> float:
        if step >= warmup_iters:
            return 1.0
        alpha = step / warmup_iters
        return 0.001 * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
) -> float:
    """optimizer=None runs a no-grad forward pass for validation loss.
    Detection models only compute losses in train() mode, so validation
    keeps the module in train mode but disables gradients."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        context = torch.enable_grad() if optimizer is not None else torch.no_grad()
        with context:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "output"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    annotations_path = args.annotations or (args.data_dir / "instances_val.json")
    images, annotations_by_image = load_annotations(annotations_path)
    train_images, val_images = split_dataset(images, args.val_fraction, args.seed)
    print(f"train images: {len(train_images)}, val images: {len(val_images)}")

    train_dataset = ParticleDataset(
        args.data_dir, train_images, annotations_by_image, train=True
    )
    val_dataset = ParticleDataset(
        args.data_dir, val_images, annotations_by_image, train=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = build_model().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    warmup_iters = min(len(train_loader) - 1, 500)
    warmup = warmup_lr_scheduler(optimizer, warmup_iters) if warmup_iters > 0 else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        active_scheduler = warmup if epoch == 0 else None
        train_loss = run_epoch(model, train_loader, device, optimizer, active_scheduler)
        val_loss = run_epoch(model, val_loader, device, optimizer=None, scheduler=None)
        scheduler.step()

        print(
            f"epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output_dir / "maskrcnn_best_model.pth")
            print(f"  saved new best (val_loss={val_loss:.4f})")

    torch.save(model.state_dict(), args.output_dir / "maskrcnn_last_model.pth")


if __name__ == "__main__":
    main()
