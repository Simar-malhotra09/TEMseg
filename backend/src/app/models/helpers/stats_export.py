import csv
import io


def _safe_spreadsheet_text(value: str) -> str:
    """Keep a user-defined shape label as text in spreadsheet programs."""
    return f"'{value}" if value.lstrip().startswith(("=", "+", "-", "@")) else value


def build_stats_csv(stats: dict) -> bytes:
    """Export one row per particle with stable, quoted CSV fields."""
    unit = str(stats.get("unit", "px"))
    has_scale = bool(stats.get("has_scale", False))
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "id",
            f"area_{unit}{'²' if unit != 'px' else ''}",
            f"eq_diameter_{unit}",
            f"perimeter_{unit}",
            "circularity",
            "solidity",
            "aspect_ratio",
            "aspect_ratio_pca",
            "aspect_ratio_min_rect",
            "n_vertices",
            "shape",
            f"nearest_neighbor_{unit}",
            "nearest_neighbor_id",
            f"border_distance_{unit}",
            "border_edge",
            "touches_image_edge",
        ]
    )
    for particle in stats.get("particles", []):
        area = (
            particle.get("area_real", particle["area_px"])
            if has_scale
            else particle["area_px"]
        )
        diameter = (
            particle.get("diameter_real", particle["diameter_px"])
            if has_scale
            else particle["diameter_px"]
        )
        perimeter = (
            particle.get("perimeter_real", particle["perimeter_px"])
            if has_scale
            else particle["perimeter_px"]
        )
        nearest = (
            particle.get("nearest_neighbor_real", particle.get("nearest_neighbor_px"))
            if has_scale
            else particle.get("nearest_neighbor_px")
        )
        border = (
            particle.get("border_distance_real", particle.get("border_distance_px"))
            if has_scale
            else particle.get("border_distance_px")
        )
        writer.writerow(
            [
                particle["id"],
                f"{area:.4f}",
                f"{diameter:.4f}",
                f"{perimeter:.4f}",
                f"{particle['circularity']:.4f}",
                f"{particle.get('solidity', 0):.4f}",
                f"{particle['aspect_ratio']:.4f}",
                f"{particle['aspect_ratio_pca']:.4f}"
                if "aspect_ratio_pca" in particle
                else "",
                f"{particle['aspect_ratio_min_rect']:.4f}"
                if "aspect_ratio_min_rect" in particle
                else "",
                particle.get("n_vertices", 0),
                _safe_spreadsheet_text(str(particle["shape"])),
                f"{nearest:.4f}" if nearest is not None else "",
                particle.get("nearest_neighbor_id") or "",
                f"{border:.4f}" if border is not None else "",
                particle.get("border_edge", ""),
                bool(particle.get("touches_image_edge", border == 0)),
            ]
        )
    return output.getvalue().encode("utf-8")
