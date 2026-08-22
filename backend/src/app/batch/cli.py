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
    apply_exclude,
    apply_include,
    normalize_extension,
    resolve_export_items,
    resolve_model_name,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="temseg-batch",
        description=(
            "Segment TEM images in a folder and write masks, instances, "
            "statistics, and COCO annotations to the output folder."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Directory containing the input images to segment.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Directory where results are written (one subfolder per image, plus summary.csv).",
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model to use: yolosam, yolomaskrcnn, or maskrcnn-synthetic.",
    )
    parser.add_argument(
        "--extension",
        "-e",
        default=None,
        help=(
            "Only process files with this extension (e.g. '.tif' or 'tif'). "
            "Default: all supported image types (.tif, .tiff, .jpg, .jpeg, "
            ".png, .npy, .emd)."
        ),
    )
    parser.add_argument(
        "--export",
        "-x",
        default="full",
        help=(
            "Export items to write: a preset (full, no_png, instances) or a "
            "comma-separated item list. Default: full."
        ),
    )
    parser.add_argument(
        "--include",
        default=None,
        help="Comma-separated export items to add to the resolved preset/list.",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="Comma-separated export items to remove from the resolved preset/list.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress per-image progress output.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    extension = normalize_extension(args.extension)

    images = [
        p
        for p in input_dir.iterdir()
        if p.suffix.lower() in VALID_IMAGE_EXTENSIONS
        and (extension is None or p.suffix.lower() == extension)
    ]
    if not images:
        if extension is not None:
            print(
                f"Error: no files with extension '{extension}' found in {input_dir}",
                file=sys.stderr,
            )
        else:
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

    if args.exclude:
        export_items = apply_exclude(
            export_items, [i.strip() for i in args.exclude.split(",") if i.strip()]
        )
    if args.include:
        export_items = apply_include(
            export_items, [i.strip() for i in args.include.split(",") if i.strip()]
        )

    config = BatchConfig(
        input_dir=input_dir,
        output_dir=Path(args.output),
        model_name=args.model,
        export_items=export_items,
        quiet=args.quiet,
        extension=extension,
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
