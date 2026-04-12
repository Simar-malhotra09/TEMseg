"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import styles from "./StatsPanel.module.css";
import type { StatsResult } from "@/lib/api";
interface Particle {
  area_px: number;
  area_real?: number;
  perimeter_px: number;
  perimeter_real?: number;
  diameter_px: number;
  diameter_real?: number;
  major_axis_px: number;
  major_axis_real?: number;
  minor_axis_px: number;
  minor_axis_real?: number;
  circularity: number;
  aspect_ratio: number;
  shape: string;
}

interface SizeStats {
  area_mean: number;
  area_std: number;
  area_min: number;
  area_max: number;
  area_median: number;
  diameter_mean: number;
  diameter_std: number;
  diameter_min: number;
  diameter_max: number;
  diameter_median: number;
  unit: string;
}

interface ShapeEntry {
  count: number;
  fraction: number;
}


interface GTScores {
  iou: number;
  dice: number;
  pixel_acc: number;
}

interface Metadata {
  image_shape?: number[];
  original_format?: string;
  pixel_size?: number | null;
  pixel_unit?: string | null;
  axes?: { scale: number; size: number; units: string }[];
}

interface Props {
  image: string | null;
  sessionId: string | null;
  metadata: Metadata | null;
  stats: StatsResult | null;
  segDone: boolean;
  groundTruthScore: GTScores | null;
  onViewDetails?: () => void;

}

function fmt(val: number, decimals = 3): string {
  if (Number.isInteger(val) || val > 1000) return val.toFixed(0);
  return val.toFixed(decimals);
}

/** Tiny inline histogram using divs */
function MiniHistogram({ values, bins = 12 }: { values: number[]; bins?: number }) {
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
function ShapeBar({ label, fraction, count }: { label: string; fraction: number; count: number }) {
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
  onViewDetails,
}: Props) {
  const [sizeOpen, setSizeOpen] = useState(true);
  const [shapeOpen, setShapeOpen] = useState(true);

  const unit = stats?.unit ?? "px";
  const hasScale = stats?.has_scale ?? false;

  return (
    <div className={styles.panel}>

      {/* ── Image Info ─────────────────────────────── */}
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
            {metadata?.image_shape && (
              <div className={styles.row}>
                <span className={styles.label}>Dimensions</span>
                <span className={styles.val}>
                  {metadata.image_shape[1]}×{metadata.image_shape[0]}
                </span>
              </div>
            )}
            {metadata?.original_format && (
              <div className={styles.row}>
                <span className={styles.label}>Format</span>
                <span className={styles.val}>{metadata.original_format.toUpperCase()}</span>
              </div>
            )}
            {metadata?.pixel_size != null && (
              <div className={styles.row}>
                <span className={styles.label}>Pixel Size</span>
                <span className={styles.val}>
                  {metadata.pixel_size.toFixed(4)} {metadata.pixel_unit ?? ""}
                </span>
              </div>
            )}
            {metadata?.axes && metadata.axes.length > 0 && metadata.axes[0]?.units && (
              <div className={styles.row}>
                <span className={styles.label}>FOV</span>
                <span className={styles.val}>
                  {(metadata.axes[0].scale * metadata.axes[0].size).toFixed(1)} {metadata.axes[0].units}
                </span>
              </div>
            )}
          </div>
        )}
      </section>

      {/* ── Overview Stats ──────────────────────────── */}
      <section className={styles.section}>
        <p className={styles.sectionLabel}>Stats</p>
        {!segDone ? (
          <p className={styles.hint}>Run segmentation to see particle statistics.</p>
        ) : stats ? (
          <div className={styles.grid}>
            <div className={styles.row}>
              <span className={styles.label}>Particles</span>
              <span className={styles.val}>{stats.particle_count}</span>
            </div>
            <div className={styles.row}>
              <span className={styles.label}>Coverage</span>
              <span className={styles.val}>{(stats.coverage * 100).toFixed(1)}%</span>
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
                {(["IoU", "Dice", "Pixel Acc"] as const).map(lbl => {
                  const key = lbl === "Pixel Acc" ? "pixel_acc" : lbl.toLowerCase() as "iou" | "dice";
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

            {segDone && stats && stats.particles?.length > 0 && onViewDetails && (
              <button className={styles.detailsBtn} onClick={onViewDetails}>
                View Details →
              </button>
            )}
          </div>
        ) : (
          <p className={styles.hint}>—</p>
        )}
      </section>

      {/* ── Size Distribution ──────────────────────── */}
      {segDone && stats && stats.particles.length > 0 && (
        <section className={styles.section}>
          <button
            className={styles.collapseBtn}
            onClick={() => setSizeOpen(o => !o)}
          >
            {sizeOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            <span className={styles.sectionLabel} style={{ margin: 0 }}>
              Size Distribution
            </span>
          </button>
          {sizeOpen && (
            <>
              <MiniHistogram
                values={stats.particles.map((p:any) =>
                  hasScale && p.diameter_real != null ? p.diameter_real : p.diameter_px
                )}
              />
              <p className={styles.histLabel}>
                Diameter ({hasScale ? unit : "px"})
              </p>
              <div className={styles.grid}>
                <div className={styles.row}>
                  <span className={styles.label}>Mean ± Std</span>
                  <span className={styles.val}>
                    {fmt(stats.size_stats.diameter_mean)} ± {fmt(stats.size_stats.diameter_std)}
                  </span>
                </div>
                <div className={styles.row}>
                  <span className={styles.label}>Median</span>
                  <span className={styles.val}>{fmt(stats.size_stats.diameter_median)}</span>
                </div>
                <div className={styles.row}>
                  <span className={styles.label}>Range</span>
                  <span className={styles.val}>
                    {fmt(stats.size_stats.diameter_min)} – {fmt(stats.size_stats.diameter_max)}
                  </span>
                </div>
              </div>
            </>
          )}
        </section>
      )}

      {/* ── Shape Distribution ─────────────────────── */}
      {segDone && stats && Object.keys(stats.shape_distribution).length > 0 && (
        <section className={styles.section}>
          <button
            className={styles.collapseBtn}
            onClick={() => setShapeOpen(o => !o)}
          >
            {shapeOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            <span className={styles.sectionLabel} style={{ margin: 0 }}>
              Shape Distribution
            </span>
          </button>
          {shapeOpen && (
            <div className={styles.shapeGrid}>
            {Object.entries(stats.shape_distribution).map(([shape, data]: [string, any]) => (
                <ShapeBar
                  key={shape}
                  label={shape}
                  fraction={data.fraction}
                  count={data.count}
                />
              ))}
            </div>
          )}
        </section>
      )}
    </div>
  );
}
