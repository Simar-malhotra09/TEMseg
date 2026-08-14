import os
import sys
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "False"

# allow running as `python cli.py` regardless of CWD, without requiring
# backend/src on PYTHONPATH beforehand
_SRC_DIR = str(Path(__file__).resolve().parents[2])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import argparse  # noqa: E402
import time  # noqa: E402

from app.batch.pipeline import run_batch  # noqa: E402
from app.batch.summary import build_summary_csv  # noqa: E402
from app.batch.types import (  # noqa: E402
    VALID_IMAGE_EXTENSIONS,
    BatchConfig,
    resolve_export_items,
    resolve_model_name,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="temseg-batch",
        description="Batch TEM image segmentation",
    )
    parser.add_argument("--input", "-i", required=True, help="Folder of input images")
    parser.add_argument("--output", "-o", required=True, help="Output folder")
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name: yolosam, maskrcnn, maskrcnn-synthetic",
    )
    parser.add_argument(
        "--export",
        "-e",
        default="full",
        help="Export preset (full, no_png, instances) or comma-separated item list",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress per-image progress output"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    images = [
        p for p in input_dir.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]
    if not images:
        print(f"Error: no valid images found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        resolve_model_name(args.model)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        export_items = resolve_export_items(args.export)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    config = BatchConfig(
        input_dir=input_dir,
        output_dir=Path(args.output),
        model_name=args.model,
        export_items=export_items,
        quiet=args.quiet,
    )

    t_start = time.perf_counter()
    results = run_batch(config)
    elapsed = time.perf_counter() - t_start

    summary_path = config.output_dir / "summary.csv"
    build_summary_csv(results, summary_path)

    ok = sum(1 for r in results if r.success)
    fail = len(results) - ok
    print(
        f"\nDone. {ok}/{len(results)} images processed in {elapsed:.1f}s",
        file=sys.stderr,
    )
    if fail:
        print(f"{fail} failed — see summary.csv for details", file=sys.stderr)
    print(f"Output: {config.output_dir}", file=sys.stderr)

    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
