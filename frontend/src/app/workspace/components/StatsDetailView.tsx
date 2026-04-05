"use client";

import { useState, useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, ReferenceLine, CartesianGrid,Line, ComposedChart
} from "recharts";
import { ArrowLeft, ArrowUpDown, ChevronUp, ChevronDown, Crosshair } from "lucide-react";
import styles from "./StatsDetailView.module.css";
import type { StatsResult } from "@/lib/api";

interface Metadata {
  image_shape?: number[];
  original_format?: string;
  pixel_size?: number | null;
  pixel_unit?: string | null;
  axes?: { scale: number; size: number; units: string }[];
}

interface GTScores {
  iou: number;
  dice: number;
  pixel_acc: number;
}

interface Props {
  stats: StatsResult;
  metadata: Metadata | null;
  groundTruthScore: GTScores | null;
  onBack: () => void;
  onLocateParticle?: (particleIndex: number) => void;
}

type SizeMode = "diameter" | "area";
type SortKey = "index" | "diameter" | "area" | "perimeter" | "circularity" | "aspect_ratio" | "shape";
type SortDir = "asc" | "desc";

const SHAPE_COLORS: Record<string, string> = {
  circular: "#7ee8a2",
  elongated: "#e8c87e",
  irregular: "#e87e7e",
};

function fmt(val: number, decimals = 2): string {
  if (Math.abs(val) >= 1000) return val.toFixed(0);
  if (Math.abs(val) >= 1) return val.toFixed(decimals);
  return val.toFixed(Math.min(decimals + 2, 6));
}

/** Build histogram bin data from an array of values */
function buildBins(values: number[], binCount = 20) {
  if (values.length === 0) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const binWidth = range / binCount;

  const bins = Array.from({ length: binCount }, (_, i) => ({
    binStart: min + i * binWidth,
    binEnd: min + (i + 1) * binWidth,
    label: fmt(min + (i + 0.5) * binWidth),
    count: 0,
  }));

  for (const v of values) {
    const idx = Math.min(Math.floor((v - min) / binWidth), binCount - 1);
    bins[idx].count++;
  }

  return bins;
}

/** Custom tooltip for histogram */
function HistTooltip({ active, payload, unit }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className={styles.tooltip}>
      <p>{fmt(d.binStart)} – {fmt(d.binEnd)} {unit}</p>
      <p className={styles.tooltipCount}>{d.count} particles</p>
    </div>
  );
}

/** Compute scaled PDF values to overlay on histogram */
function addFitCurve(
  bins: ReturnType<typeof buildBins>,
  fits: StatsResult["distribution_fits_diameter"],
  totalParticles: number,
) {
  if (!fits?.reliable || !fits.best_model || !fits.fits) return bins;

  const fit = fits.fits[fits.best_model];
  if (!fit) return bins;

  const binWidth = bins.length > 1 ? bins[1].binStart - bins[0].binStart : 1;
  const p = fit.params;

  return bins.map(bin => {
    const x = (bin.binStart + bin.binEnd) / 2;
    let pdf = 0;

    if (fits.best_model === "normal") {
      const z = (x - p.mean) / p.std;
      pdf = Math.exp(-0.5 * z * z) / (p.std * Math.sqrt(2 * Math.PI));
    } else if (fits.best_model === "lognormal") {
      if (x > 0) {
        const lnx = Math.log(x);
        const z = (lnx - p.mu_log) / p.sigma_log;
        pdf = Math.exp(-0.5 * z * z) / (x * p.sigma_log * Math.sqrt(2 * Math.PI));
      }
    } else if (fits.best_model === "weibull") {
      if (x > 0) {
        const c = p.shape;
        const s = p.scale;
        pdf = (c / s) * Math.pow(x / s, c - 1) * Math.exp(-Math.pow(x / s, c));
      }
    }

    // scale PDF to match histogram counts: expected_count = pdf * binWidth * N
    return { ...bin, fit: pdf * binWidth * totalParticles };
  });
}

export default function StatsDetailView({ stats, metadata, groundTruthScore, onBack, onLocateParticle }: Props) {
  const [sizeMode, setSizeMode] = useState<SizeMode>("diameter");
  const [sortKey, setSortKey] = useState<SortKey>("index");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [hoveredRow, setHoveredRow] = useState<number | null>(null);

  const hasScale = stats.has_scale;
  const unit = stats.unit ?? "px";
  const particles = stats.particles ?? [];

  // histogram/pdf 
const histData = useMemo(() => {
    const values = particles.map(p => {
      if (sizeMode === "diameter") {
        return hasScale && p.diameter_real != null ? p.diameter_real : p.diameter_px;
      } else {
        return hasScale && p.area_real != null ? p.area_real : p.area_px;
      }
    });
    const bins = buildBins(values, Math.min(20, Math.max(8, Math.ceil(Math.sqrt(particles.length)))));

    const fits = sizeMode === "diameter"
      ? stats.distribution_fits_diameter
      : stats.distribution_fits_area;

    return addFitCurve(bins, fits, particles.length);
  }, [particles, sizeMode, hasScale, stats.distribution_fits_diameter, stats.distribution_fits_area]);

  const histMean = sizeMode === "diameter"
    ? stats.size_stats.diameter_mean
    : stats.size_stats.area_mean;
  const histMedian = sizeMode === "diameter"
    ? stats.size_stats.diameter_median
    : stats.size_stats.area_median;
  const histUnit = sizeMode === "diameter" ? unit : `${unit}²`;

  // ── shape pie data ──────────────────────────────────────────
  const shapeData = useMemo(() => {
    return Object.entries(stats.shape_distribution).map(([name, data]) => ({
      name,
      value: data.count,
      fraction: data.fraction,
    }));
  }, [stats.shape_distribution]);

  // ── sortable table ──────────────────────────────────────────
  const sortedParticles = useMemo(() => {
    const indexed = particles.map((p, i) => ({ ...p, index: i + 1 }));
    indexed.sort((a, b) => {
      let va: number | string, vb: number | string;
      switch (sortKey) {
        case "index": va = a.index; vb = b.index; break;
        case "diameter": va = a.diameter_real ?? a.diameter_px; vb = b.diameter_real ?? b.diameter_px; break;
        case "area": va = a.area_real ?? a.area_px; vb = b.area_real ?? b.area_px; break;
        case "perimeter": va = a.perimeter_real ?? a.perimeter_px; vb = b.perimeter_real ?? b.perimeter_px; break;
        case "circularity": va = a.circularity; vb = b.circularity; break;
        case "aspect_ratio": va = a.aspect_ratio; vb = b.aspect_ratio; break;
        case "shape": va = a.shape; vb = b.shape; break;
        default: va = a.index; vb = b.index;
      }
      if (typeof va === "string") return sortDir === "asc" ? va.localeCompare(vb as string) : (vb as string).localeCompare(va);
      return sortDir === "asc" ? (va as number) - (vb as number) : (vb as number) - (va as number);
    });
    return indexed;
  }, [particles, sortKey, sortDir, hasScale]);

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir(d => d === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  }

  function SortIcon({ col }: { col: SortKey }) {
    if (sortKey !== col) return <ArrowUpDown size={10} className={styles.sortIconInactive} />;
    return sortDir === "asc"
      ? <ChevronUp size={10} className={styles.sortIconActive} />
      : <ChevronDown size={10} className={styles.sortIconActive} />;
  }

  return (
    <div className={styles.root}>

      {/* header */}
      <header className={styles.header}>
        <button className={styles.backBtn} onClick={onBack}>
          <ArrowLeft size={16} /> Back to workspace
        </button>
        <h1 className={styles.title}>Particle Analysis</h1>
        <span className={styles.subtitle}>
          {stats.particle_count} particles
          {hasScale && ` · ${unit}/px`}
        </span>
      </header>

      {/* top row: charts */}
      <div className={styles.chartsRow}>

        {/* size distribution histogram */}
        <div className={styles.chartCard}>
          <div className={styles.chartHeader}>
            <h2 className={styles.chartTitle}>Size Distribution</h2>
            <div className={styles.toggleGroup}>
              <button
                className={`${styles.toggleBtn} ${sizeMode === "diameter" ? styles.toggleActive : ""}`}
                onClick={() => setSizeMode("diameter")}
              >Equivalent Diameter</button>
              <button
                className={`${styles.toggleBtn} ${sizeMode === "area" ? styles.toggleActive : ""}`}
                onClick={() => setSizeMode("area")}
              >Area</button>
            </div>
          </div>

          <div className={styles.chartWrap}>
            <ResponsiveContainer width="100%" height={280}>
                <ComposedChart data={histData} margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />
                <XAxis
                  dataKey="label"
                  tick={{ fill: "#666", fontSize: 10 }}
                  label={{ value: `${sizeMode === "diameter" ? "Diameter" : "Area"} (${histUnit})`, position: "bottom", offset: 10, fill: "#555", fontSize: 11 }}
                  interval={Math.max(0, Math.floor(histData.length / 8) - 1)}
                />
                <YAxis
                  tick={{ fill: "#666", fontSize: 10 }}
                  label={{ value: "Count", angle: -90, position: "insideLeft", offset: 0, fill: "#555", fontSize: 11 }}
                />
                <Tooltip content={<HistTooltip unit={histUnit} />} cursor={{ fill: "rgba(126, 232, 162, 0.08)" }} />
                <ReferenceLine
                  x={fmt(histMean)}
                  stroke="#7ee8a2"
                  strokeDasharray="4 4"
                  strokeWidth={1.5}
                  label={{ value: "mean", position: "top", fill: "#7ee8a2", fontSize: 10 }}
                />
                <ReferenceLine
                  x={fmt(histMedian)}
                  stroke="#e8c87e"
                  strokeDasharray="4 4"
                  strokeWidth={1.5}
                  label={{ value: "median", position: "top", fill: "#e8c87e", fontSize: 10 }}
                />
                <Bar dataKey="count" fill="#7ee8a2" opacity={0.75} radius={[2, 2, 0, 0]} />
                <Line
                  dataKey="fit"
                  type="monotone"
                  stroke="#e8c87e"
                  strokeWidth={2}
                  dot={false}
                  name="Fitted distribution"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <div className={styles.chartStats}>
            <span>Mean: {fmt(stats.size_stats[sizeMode === "diameter" ? "diameter_mean" : "area_mean"])} {histUnit}</span>
            <span>Std: {fmt(stats.size_stats[sizeMode === "diameter" ? "diameter_std" : "area_std"])} {histUnit}</span>
            <span>Median: {fmt(stats.size_stats[sizeMode === "diameter" ? "diameter_median" : "area_median"])} {histUnit}</span>
            {(() => {
              const fits = sizeMode === "diameter" ? stats.distribution_fits_diameter : stats.distribution_fits_area;
              if (!fits?.reliable || !fits.best_model) return null;
              return <span style={{ color: "#e8c87e" }}>Fit: {fits.best_model}</span>;
            })()}
          </div>
        </div>

        {/* shape distribution */}
        <div className={styles.chartCard}>
          <h2 className={styles.chartTitle}>Shape Distribution</h2>

          <div className={styles.chartWrap}>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={shapeData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={3}
                  dataKey="value"
                  nameKey="name"
                  label={({ name, fraction }) => `${name} ${(fraction * 100).toFixed(0)}%`}
                >
                  {shapeData.map((entry) => (
                    <Cell key={entry.name} fill={SHAPE_COLORS[entry.name] ?? "#888"} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value: number, name: string) => [`${value} particles`, name]}
                  contentStyle={{ background: "#161616", border: "1px solid #2a2a2a", borderRadius: 4, fontSize: 12 }}
                  itemStyle={{ color: "#e8e6e1" }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* legend + stats */}
          <div className={styles.shapeLegend}>
            {shapeData.map(entry => (
              <div key={entry.name} className={styles.legendItem}>
                <span className={styles.legendDot} style={{ background: SHAPE_COLORS[entry.name] ?? "#888" }} />
                <span className={styles.legendLabel}>{entry.name}</span>
                <span className={styles.legendVal}>{entry.value} ({(entry.fraction * 100).toFixed(1)}%)</span>
              </div>
            ))}
          </div>
        </div>
        {/* distribution fit */}
        {(() => {
          const fits = sizeMode === "diameter"
            ? stats.distribution_fits_diameter
            : stats.distribution_fits_area;
          if (!fits?.reliable || !fits.fits) return null;
          return (
            <div className={styles.fitCard}>
              <div className={styles.fitHeader}>
                <h2 className={styles.chartTitle}>
                  Distribution Fit — {sizeMode === "diameter" ? "Eq. Diameter" : "Area"} ({histUnit})
                </h2>
                <span className={styles.fitBest}>
                  Best fit: <strong>{fits.best_model}</strong>
                </span>
              </div>
              <div className={styles.fitGrid}>
                {Object.entries(fits.fits).map(([model, fit]) => {
                  const isBest = model === fits.best_model;
                  return (
                    <div key={model} className={`${styles.fitModel} ${isBest ? styles.fitModelBest : ""}`}>
                      <div className={styles.fitModelHeader}>
                        <span className={styles.fitModelName}>{model}</span>
                        {isBest && <span className={styles.fitBestBadge}>best</span>}
                      </div>
                      <div className={styles.fitParams}>
                        {Object.entries(fit.params).map(([k, v]) => (
                          <div key={k} className={styles.fitParam}>
                            <span className={styles.fitParamKey}>{k}</span>
                            <span className={styles.fitParamVal}>{fmt(v, 4)}</span>
                          </div>
                        ))}
                      </div>
                      <div className={styles.fitGof}>
                        <span>KS p-value: {fit.ks_pvalue < 0.001 ? "<0.001" : fmt(fit.ks_pvalue, 4)}</span>
                        <span className={fit.ks_pvalue >= 0.05 ? styles.fitPass : styles.fitFail}>
                          {fit.ks_pvalue >= 0.05 ? "✓ cannot reject" : "✗ poor fit"}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })()}
      </div>

      {/* summary cards */}
      <div className={styles.cardsRow}>
        {[
          { label: "Particles", value: `${stats.particle_count}` },
          { label: "Coverage", value: `${(stats.coverage * 100).toFixed(1)}%` },
          {
            label: "Avg Eq. Diameter",
            value: hasScale && stats.avg_diameter_real != null
              ? `${fmt(stats.avg_diameter_real)}  ±  ${fmt(stats.size_stats["diameter_std"])} ${unit}`
              : `${fmt(stats.avg_diameter_px)} px`,
          },
          {
            label: "Avg Area",
            value: hasScale && stats.avg_area_real != null
              ? `${fmt(stats.avg_area_real)}  ± ${fmt(stats.size_stats["area_std"])} ${unit}²`
              : `${fmt(stats.avg_area_px)} px²`,
          },
          { label: "Avg Circularity", value: fmt(stats.avg_circularity) },
          { label: "Avg Aspect Ratio", value: fmt(stats.avg_aspect_ratio) },
          ...(groundTruthScore ? [
            { label: "IoU", value: groundTruthScore.iou.toFixed(3) },
            { label: "Dice", value: groundTruthScore.dice.toFixed(3) },
            { label: "Pixel Acc", value: groundTruthScore.pixel_acc.toFixed(3) },
          ] : []),
        ].map(card => (
          <div key={card.label} className={styles.card}>
            <span className={styles.cardLabel}>{card.label}</span>
            <span className={styles.cardVal}>{card.value}</span>
          </div>
        ))}
      </div>

      {/* per-particle table */}
      <div className={styles.tableCard}>
        <div className={styles.tableHeader}>
          <h2 className={styles.chartTitle}>Per-Particle Data</h2>
          {onLocateParticle && (
            <span className={styles.tableHint}>
              <Crosshair size={10} /> Click a row to locate on canvas
            </span>
          )}
        </div>
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                {([
                  ["index", "#"],
                  ["diameter", `Eq. Diameter (${hasScale ? unit : "px"})`],
                  ["area", `Area (${hasScale ? unit + "²" : "px²"})`],
                  ["perimeter", `Perimeter (${hasScale ? unit : "px"})`],
                  ["circularity", "Circularity"],
                  ["aspect_ratio", "Aspect Ratio"],
                  ["shape", "Shape"],
                ] as [SortKey, string][]).map(([key, label]) => (
                  <th key={key} onClick={() => handleSort(key)} className={styles.th}>
                    {label} <SortIcon col={key} />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedParticles.map(p => (
                <tr
                  key={p.index}
                  className={`${styles.tr} ${onLocateParticle ? styles.trClickable : ""} ${hoveredRow === p.index ? styles.trHighlighted : ""}`}
                  onMouseEnter={() => setHoveredRow(p.index)}
                  onMouseLeave={() => setHoveredRow(null)}
                  onClick={() => onLocateParticle?.(p.index - 1)}
                >
                  <td className={styles.td}>
                    <span className={styles.particleId}>{p.index}</span>
                  </td>
                  <td className={styles.td}>{fmt(hasScale && p.diameter_real != null ? p.diameter_real : p.diameter_px)}</td>
                  <td className={styles.td}>{fmt(hasScale && p.area_real != null ? p.area_real : p.area_px)}</td>
                  <td className={styles.td}>{fmt(hasScale && p.perimeter_real != null ? p.perimeter_real : p.perimeter_px)}</td>
                  <td className={styles.td}>{fmt(p.circularity)}</td>
                  <td className={styles.td}>{fmt(p.aspect_ratio)}</td>
                  <td className={styles.td}>
                    <span className={styles.shapeBadge} style={{ borderColor: SHAPE_COLORS[p.shape] ?? "#888", color: SHAPE_COLORS[p.shape] ?? "#888" }}>
                      {p.shape}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
