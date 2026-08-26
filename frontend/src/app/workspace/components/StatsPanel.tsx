"use client";

import { useState, useEffect } from "react";
import {
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  Pencil,
  Check,
  X,
  Ruler,
  Settings,
} from "lucide-react";
import styles from "./StatsPanel.module.css";
import type { StatsResult, Metadata } from "@/lib/api";
import { updatePixelSize } from "@/lib/api";
import ShapeRulesModal from "./ShapeRulesModal";

interface GTScores {
  iou: number;
  dice: number;
  pixel_acc: number;
}

interface Props {
  image: string | null;
  sessionId: string | null;
  metadata: Metadata | null;
  stats: StatsResult | null;
  segDone: boolean;
  groundTruthScore: GTScores | null;
  scaleBarMode: boolean;
  scaleBarPixels: number | null;
  onViewDetails?: () => void;
  onMetadataUpdate?: (metadata: Metadata, stats?: StatsResult) => void;
  onToggleScaleBar?: () => void;
  onScaleBarCancel?: () => void;
}

function fmt(val: number, decimals = 3): string {
  if (Number.isInteger(val) || val > 1000) return val.toFixed(0);
  return val.toFixed(decimals);
}

/** Format the image source as "<parent-dir>/<filename-without-ext>". */
function formatSource(filePath?: string, fileName?: string): string | null {
  if (!filePath) return null;
  const normalized = filePath.replace(/\\/g, "/");
  const parts = normalized.split("/").filter(Boolean);
  const parent = parts.length > 1 ? parts[parts.length - 2] : null;
  const last = parts.length ? parts[parts.length - 1] : "";
  const stem = (fileName && fileName.trim()) || last.replace(/\.[^.]*$/, "");
  if (parent && stem) return `${parent}/${stem}`;
  return stem || normalized;
}

/** Tiny inline histogram using divs */
function MiniHistogram({
  values,
  bins = 12,
}: {
  values: number[];
  bins?: number;
}) {
  if (values.length === 0) return null;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const binWidth = range / bins;

  const counts = new Array(bins).fill(0);
  for (const v of values) {
    const idx = Math.min(Math.floor((v - min) / binWidth), bins - 1);
    counts[idx]++;
  }
  const maxCount = Math.max(...counts);

  return (
    <div className={styles.histogram}>
      {counts.map((c, i) => (
        <div
          key={i}
          className={styles.histBar}
          style={{ height: `${maxCount > 0 ? (c / maxCount) * 100 : 0}%` }}
          title={`${(min + i * binWidth).toFixed(1)}–${(min + (i + 1) * binWidth).toFixed(1)}: ${c}`}
        />
      ))}
    </div>
  );
}

/** Shape distribution bar */
function ShapeBar({
  label,
  fraction,
  count,
}: {
  label: string;
  fraction: number;
  count: number;
}) {
  const colors: Record<string, string> = {
    circular: "#7ee8a2",
    elongated: "#e8c87e",
    irregular: "#e87e7e",
  };
  return (
    <div className={styles.shapeRow}>
      <span className={styles.shapeLabel}>{label}</span>
      <div className={styles.shapeBarOuter}>
        <div
          className={styles.shapeBarInner}
          style={{
            width: `${fraction * 100}%`,
            background: colors[label] ?? "#888",
          }}
        />
      </div>
      <span className={styles.shapeCount}>{count}</span>
    </div>
  );
}

export default function StatsPanel({
  image,
  sessionId,
  metadata,
  stats,
  segDone,
  groundTruthScore,
  scaleBarMode,
  scaleBarPixels,
  onViewDetails,
  onMetadataUpdate,
  onToggleScaleBar,
  onScaleBarCancel,
}: Props) {
  const [sizeOpen, setSizeOpen] = useState(true);
  const [shapeOpen, setShapeOpen] = useState(true);
  const [shapeRulesOpen, setShapeRulesOpen] = useState(false);
  const [editingPixel, setEditingPixel] = useState(false);
  const [editSize, setEditSize] = useState("");
  const [editUnit, setEditUnit] = useState("nm");
  const [pixelBusy, setPixelBusy] = useState(false);
  const [sbLength, setSbLength] = useState("");
  const [sbUnit, setSbUnit] = useState("nm");

  useEffect(() => {
    if (scaleBarPixels != null) {
      setSbLength("");
      setSbUnit(metadata?.pixel_unit ?? "nm");
    }
  }, [scaleBarPixels, metadata?.pixel_unit]);

  const imageSource = formatSource(metadata?.file_path, metadata?.file_name);
  const unit = stats?.unit ?? "px";
  const hasScale = stats?.has_scale ?? false;
  const pixelSize = metadata?.pixel_size;

  const canEditPixel = sessionId != null && metadata != null;
  const pixelSizeMissing = pixelSize == null || pixelSize === "-";

  function startEdit() {
    if (!canEditPixel) return;
    setEditSize(pixelSizeMissing ? "" : String(pixelSize));
    setEditUnit(metadata?.pixel_unit ?? "nm");
    setEditingPixel(true);
  }

  function cancelEdit() {
    setEditingPixel(false);
    setEditSize("");
    setEditUnit("nm");
  }

  async function saveEdit() {
    if (!sessionId) return;
    const val = parseFloat(editSize);
    if (Number.isNaN(val) || val <= 0) return;
    setPixelBusy(true);
    try {
      const result = await updatePixelSize(
        sessionId,
        val,
        editUnit.trim() || "nm",
      );
      if (result.metadata) {
        onMetadataUpdate?.(
          result.metadata as Metadata,
          result.stats as StatsResult | undefined,
        );
      }
      setEditingPixel(false);
    } catch (e) {
      console.error("Failed to update pixel size:", e);
    } finally {
      setPixelBusy(false);
    }
  }

  async function confirmScaleBar() {
    if (!sessionId || scaleBarPixels == null) return;
    const length = parseFloat(sbLength);
    if (Number.isNaN(length) || length <= 0) return;
    const pixelSize = length / scaleBarPixels;
    setPixelBusy(true);
    try {
      const result = await updatePixelSize(
        sessionId,
        pixelSize,
        sbUnit.trim() || "nm",
      );
      if (result.metadata) {
        onMetadataUpdate?.(
          result.metadata as Metadata,
          result.stats as StatsResult | undefined,
        );
      }
      onScaleBarCancel?.();
    } catch (e) {
      console.error("Failed to set pixel size from scale bar:", e);
    } finally {
      setPixelBusy(false);
    }
  }
  return (
    <div className={styles.panel}>
      {/* Image Info */}
      <section className={styles.section}>
        <p className={styles.sectionLabel}>Image Info</p>
        {!image ? (
          <p className={styles.hint}>Upload an image to see info.</p>
        ) : (
          <div className={styles.grid}>
            {sessionId && (
              <div className={styles.row}>
                <span className={styles.label}>Session</span>
                <span className={styles.val}>{sessionId}</span>
              </div>
            )}
            {imageSource && (
              <div className={styles.row}>
                <span className={styles.label}>Source</span>
                <span className={styles.val}>{imageSource}</span>
              </div>
            )}
            {metadata?.original_format && (
              <div className={styles.row}>
                <span className={styles.label}>Format</span>
                <span className={styles.val}>
                  {metadata.original_format.toUpperCase()}
                </span>
              </div>
            )}
            {metadata?.image_shape && (
              <div className={styles.row}>
                <span className={styles.label}>Dimensions</span>
                <span className={styles.val}>
                  {metadata.image_shape[1]}×{metadata.image_shape[0]}
                </span>
              </div>
            )}
            {metadata?.pixel_size != null && (
              <div className={styles.row}>
                <span className={styles.label}>Pixel Size</span>
                {!editingPixel ? (
                  <span
                    className={styles.val}
                    style={{ display: "flex", alignItems: "center", gap: 6 }}
                  >
                    {pixelSize == null || pixelSize === "-"
                      ? "Not Found"
                      : typeof pixelSize === "number"
                        ? pixelSize.toFixed(4)
                        : "Not Found"}
                    {metadata.pixel_unit == null || metadata.pixel_unit === "-"
                      ? ""
                      : " " + metadata.pixel_unit}
                    {canEditPixel && (
                      <button
                        type="button"
                        className={styles.iconBtn}
                        onClick={startEdit}
                        title="Edit pixel size"
                      >
                        <Pencil size={10} />
                      </button>
                    )}
                  </span>
                ) : (
                  <span
                    className={styles.val}
                    style={{ display: "flex", alignItems: "center", gap: 4 }}
                  >
                    <input
                      type="number"
                      step="any"
                      min="0"
                      value={editSize}
                      onChange={(e) => setEditSize(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") saveEdit();
                        if (e.key === "Escape") cancelEdit();
                      }}
                      className={styles.pixelInput}
                      placeholder="0.00"
                      autoFocus
                    />
                    <input
                      type="text"
                      value={editUnit}
                      onChange={(e) => setEditUnit(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") saveEdit();
                        if (e.key === "Escape") cancelEdit();
                      }}
                      className={styles.pixelUnitInput}
                      placeholder="nm"
                    />
                    <button
                      type="button"
                      className={styles.iconBtn}
                      onClick={saveEdit}
                      disabled={pixelBusy}
                      title="Save"
                    >
                      <Check size={10} />
                    </button>
                    <button
                      type="button"
                      className={styles.iconBtn}
                      onClick={cancelEdit}
                      disabled={pixelBusy}
                      title="Cancel"
                    >
                      <X size={10} />
                    </button>
                  </span>
                )}
              </div>
            )}
            {metadata?.axes &&
              metadata.axes.length > 0 &&
              metadata.axes[0]?.units && (
                <div className={styles.row}>
                  <span className={styles.label}>FOV</span>
                  <span className={styles.val}>
                    {(metadata.axes[0].scale * metadata.axes[0].size).toFixed(
                      1,
                    )}{" "}
                    {metadata.axes[0].units}
                  </span>
                </div>
              )}

            {/* Scale bar calibration  */}
            {scaleBarPixels == null && (
              <div className={styles.row}>
                <span
                  className={styles.label}
                  style={{ color: scaleBarMode ? "#7ee8a2" : undefined }}
                >
                  {scaleBarMode ? "Draw line →" : "Scale Bar"}
                </span>
                <span
                  className={styles.val}
                  style={{ display: "flex", alignItems: "center", gap: 4 }}
                >
                  {scaleBarMode ? (
                    <button
                      type="button"
                      className={styles.iconBtn}
                      onClick={onScaleBarCancel}
                      title="Cancel"
                    >
                      <X size={10} />
                    </button>
                  ) : (
                    <button
                      type="button"
                      className={styles.iconBtn}
                      onClick={onToggleScaleBar}
                      title="Measure pixel size from scale bar"
                    >
                      <Ruler size={10} />
                    </button>
                  )}
                </span>
              </div>
            )}
            {scaleBarPixels != null && (
              <div
                className={styles.row}
                style={{
                  flexDirection: "column",
                  alignItems: "stretch",
                  gap: 4,
                }}
              >
                <span className={styles.label} style={{ color: "#7ee8a2" }}>
                  {scaleBarPixels.toFixed(1)} px — enter length:
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <input
                    type="number"
                    step="any"
                    min="0"
                    value={sbLength}
                    onChange={(e) => setSbLength(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") confirmScaleBar();
                      if (e.key === "Escape") onScaleBarCancel?.();
                    }}
                    className={styles.pixelInput}
                    placeholder="0.00"
                    autoFocus
                  />
                  <input
                    type="text"
                    value={sbUnit}
                    onChange={(e) => setSbUnit(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") confirmScaleBar();
                      if (e.key === "Escape") onScaleBarCancel?.();
                    }}
                    className={styles.pixelUnitInput}
                    placeholder="nm"
                  />
                  <button
                    type="button"
                    className={styles.iconBtn}
                    onClick={confirmScaleBar}
                    disabled={pixelBusy}
                    title="Confirm"
                  >
                    <Check size={10} />
                  </button>
                  <button
                    type="button"
                    className={styles.iconBtn}
                    onClick={onScaleBarCancel}
                    disabled={pixelBusy}
                    title="Cancel"
                  >
                    <X size={10} />
                  </button>
                </span>
              </div>
            )}
          </div>
        )}
      </section>

      {/* Shape Rules */}
      <section className={styles.section}>
        <div className={styles.row}>
          <span className={styles.label}>Shape Rules</span>
          <button
            type="button"
            className={styles.detailsBtn}
            style={{ marginTop: 0, width: "auto", padding: "4px 10px" }}
            onClick={() => setShapeRulesOpen(true)}
          >
            Edit
          </button>
        </div>
      </section>

      {/* Overview Stats  */}
      <section className={styles.section}>
        <p className={styles.sectionLabel}>Stats</p>
        {!segDone ? (
          <p className={styles.hint}>
            Run segmentation to see particle statistics.
          </p>
        ) : stats ? (
          <div className={styles.grid}>
            {stats.particle_count > 200 && (
              <div className={styles.warnRow}>
                <AlertTriangle size={12} />
                <span>{stats.particle_count} particles. UI may be laggy.</span>
              </div>
            )}
            <div className={styles.row}>
              <span className={styles.label}>Particles</span>
              <span className={styles.val}>{stats.particle_count}</span>
            </div>
            <div className={styles.row}>
              <span className={styles.label}>Coverage</span>
              <span className={styles.val}>
                {(stats.coverage * 100).toFixed(1)}%
              </span>
            </div>
            <div className={styles.row}>
              <span className={styles.label}>Avg Diameter</span>
              <span className={styles.val}>
                {hasScale && stats.avg_diameter_real != null
                  ? `${fmt(stats.avg_diameter_real)} ${unit}`
                  : `${fmt(stats.avg_diameter_px)} px`}
              </span>
            </div>
            <div className={styles.row}>
              <span className={styles.label}>Avg Area</span>
              <span className={styles.val}>
                {hasScale && stats.avg_area_real != null
                  ? `${fmt(stats.avg_area_real)} ${unit}²`
                  : `${fmt(stats.avg_area_px)} px²`}
              </span>
            </div>
            <div className={styles.row}>
              <span className={styles.label}>Avg Circularity</span>
              <span className={styles.val}>{fmt(stats.avg_circularity)}</span>
            </div>
            <div className={styles.row}>
              <span className={styles.label}>Avg Aspect Ratio</span>
              <span className={styles.val}>{fmt(stats.avg_aspect_ratio)}</span>
            </div>

            {/* GT scores */}
            {groundTruthScore && (
              <>
                <div className={styles.divider} />
                {(["IoU", "Dice", "Pixel Acc"] as const).map((lbl) => {
                  const key =
                    lbl === "Pixel Acc"
                      ? "pixel_acc"
                      : (lbl.toLowerCase() as "iou" | "dice");
                  return (
                    <div className={styles.row} key={lbl}>
                      <span className={styles.label}>{lbl}</span>
                      <span className={styles.val}>
                        {groundTruthScore[key]?.toFixed(3) ?? "—"}
                      </span>
                    </div>
                  );
                })}
              </>
            )}

            {segDone &&
              stats &&
              stats.particles?.length > 0 &&
              onViewDetails && (
                <button
                  type="button"
                  className={styles.detailsBtn}
                  onClick={onViewDetails}
                >
                  View Details →
                </button>
              )}
          </div>
        ) : (
          <p className={styles.hint}>—</p>
        )}
      </section>

      {/* Size Distribution */}
      {segDone && stats && stats.particles.length > 0 && (
        <section className={styles.section}>
          <button
            type="button"
            className={styles.collapseBtn}
            onClick={() => setSizeOpen((o) => !o)}
          >
            {sizeOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            <span className={styles.sectionLabel} style={{ margin: 0 }}>
              Size Distribution
            </span>
          </button>
          {sizeOpen && (
            <>
              <MiniHistogram
                values={stats.particles.map((p: any) =>
                  hasScale && p.diameter_real != null
                    ? p.diameter_real
                    : p.diameter_px,
                )}
              />
              <p className={styles.histLabel}>
                Diameter ({hasScale ? unit : "px"})
              </p>
              <div className={styles.grid}>
                <div className={styles.row}>
                  <span className={styles.label}>Mean ± Std</span>
                  <span className={styles.val}>
                    {fmt(stats.size_stats.diameter_mean)} ±{" "}
                    {fmt(stats.size_stats.diameter_std)}
                  </span>
                </div>
                <div className={styles.row}>
                  <span className={styles.label}>Median</span>
                  <span className={styles.val}>
                    {fmt(stats.size_stats.diameter_median)}
                  </span>
                </div>
                <div className={styles.row}>
                  <span className={styles.label}>Range</span>
                  <span className={styles.val}>
                    {fmt(stats.size_stats.diameter_min)} –{" "}
                    {fmt(stats.size_stats.diameter_max)}
                  </span>
                </div>
              </div>
            </>
          )}
        </section>
      )}

      {/*  Shape Distribution  */}
      {segDone && stats && Object.keys(stats.shape_distribution).length > 0 && (
        <section className={styles.section}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <button
              type="button"
              className={styles.collapseBtn}
              onClick={() => setShapeOpen((o) => !o)}
            >
              {shapeOpen ? (
                <ChevronDown size={12} />
              ) : (
                <ChevronRight size={12} />
              )}
              <span className={styles.sectionLabel} style={{ margin: 0 }}>
                Shape Distribution
              </span>
            </button>
            <button
              type="button"
              className={styles.iconBtn}
              onClick={() => setShapeRulesOpen(true)}
              title="Edit shape rules"
            >
              <Settings size={12} />
            </button>
          </div>
          {shapeOpen && (
            <div className={styles.shapeGrid}>
              {Object.entries(stats.shape_distribution).map(
                ([shape, data]: [string, any]) => (
                  <ShapeBar
                    key={shape}
                    label={shape}
                    fraction={data.fraction}
                    count={data.count}
                  />
                ),
              )}
            </div>
          )}
        </section>
      )}

      <ShapeRulesModal
        open={shapeRulesOpen}
        onClose={() => setShapeRulesOpen(false)}
      />
    </div>
  );
}
