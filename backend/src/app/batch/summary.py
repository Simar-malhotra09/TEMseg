import csv
from pathlib import Path

from app.batch.types import ImageResult

SUMMARY_HEADER = [
    "filename",
    "status",
    "particle_count",
    "coverage",
    "avg_area",
    "avg_diameter",
    "avg_circularity",
    "avg_aspect_ratio",
    "unit",
    "time_s",
    "error",
]


def build_summary_csv(results: list[ImageResult], output_path: Path) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_HEADER)
        for r in results:
            writer.writerow(
                [
                    r.filename,
                    "OK" if r.success else "FAILED",
                    f"{r.particle_count}",
                    f"{r.coverage:.4f}",
                    f"{r.avg_area:.4f}",
                    f"{r.avg_diameter:.4f}",
                    f"{r.avg_circularity:.4f}",
                    f"{r.avg_aspect_ratio:.4f}",
                    r.unit,
                    f"{r.time_elapsed:.2f}",
                    r.error or "",
                ]
            )
