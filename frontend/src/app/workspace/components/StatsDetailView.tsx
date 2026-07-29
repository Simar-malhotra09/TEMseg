"use client";

import { useState, useMemo, useRef } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, ReferenceLine, CartesianGrid,
  ComposedChart, Line,
  ScatterChart, Scatter, ZAxis,
} from "recharts";
import { ArrowLeft, ArrowUpDown, ChevronUp, ChevronDown, Crosshair } from "lucide-react";
import styles from "./StatsDetailView.module.css";
import type { StatsResult, Metadata } from "@/lib/api";
import { PARTICLE_METRIC_FIELDS } from "@/lib/api";



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
  onLocateShape?:(shape: string)=> void; 
}


type SizeMode = "diameter" | "area";
type SortDir = "asc" | "desc";

// per-particle table columns: "index" (row number) plus every field in the
// shared metric list, so this table and the refine-mode tooltip picker never
// drift apart on what "available fields" means
const TABLE_COLUMN_KEYS = ["index", ...PARTICLE_METRIC_FIELDS] as const;
type SortKey = typeof TABLE_COLUMN_KEYS[number];

function columnLabel(key: SortKey, hasScale: boolean, unit: string): string {
  switch (key) {
    case "index": return "#";
    case "diameter": return `Eq. Diameter (${hasScale ? unit : "px"})`;
    case "area": return `Area (${hasScale ? unit + "²" : "px²"})`;
    case "circularity": return "Circ.";
    case "solidity": return "Solidity";
    case "aspect_ratio": return "Asp. Ratio";
    case "n_vertices": return "Vertices";
    case "shape": return "Shape";
  }
}

const SHAPE_COLORS: Record<string, string> = {
  spherical: "#7ee8a2",
  "quasi-spherical": "#a2d4e8",
  faceted: "#c8a2e8",
  triangular: "#e8d47e",
  elongated: "#e8c87e",
  rod: "#e8a27e",
  irregular: "#e87e7e",
};
 

function fmt(val: number, decimals = 2): string {
  if (Math.abs(val) >= 1000) return val.toFixed(0);
  if (Math.abs(val) >= 1) return val.toFixed(decimals);
  return val.toFixed(Math.min(decimals + 2, 6));
}


// Histogram related  

// Build histogram bin data from an array of values 
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

// Custom tooltip for histogram 
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

// Compute scaled PDF values to overlay on histogram
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


export default function StatsDetailView({ stats, metadata, groundTruthScore, onBack, onLocateParticle, onLocateShape}: Props) {
  const [sizeMode, setSizeMode] = useState<SizeMode>("diameter");
  const [sortKey, setSortKey] = useState<SortKey>("index");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [hoveredRow, setHoveredRow] = useState<number | null>(null);
  const [activeShapes, setActiveShapes] = useState<Set<string>>(new Set(
    Object.keys(stats.shape_distribution)
  ));
  const hasScale = stats.has_scale;
  const unit = stats.unit ?? "px";
  const particles = stats.particles ?? [];
  const chartWrapRef = useRef<HTMLDivElement>(null);

  // serialize the chart's svg onto a canvas, draw the stats line below it, and download as PNG
  const handleExportHistogramPNG = () => {
    const svgEl = chartWrapRef.current?.querySelector("svg");
    if (!svgEl) return;

    const svgString = new XMLSerializer().serializeToString(svgEl);
    const svgUrl = URL.createObjectURL(new Blob([svgString], { type: "image/svg+xml;charset=utf-8" }));
    const { width, height } = svgEl.getBoundingClientRect();
    const lineHeight = 18;
    const statsRowHeight = histStatsLines.length * lineHeight + 8;

    const img = new Image();
    img.onload = () => {
      const scale = 2; // render at 2x for a crisp export
      const canvas = document.createElement("canvas");
      canvas.width = width * scale;
      canvas.height = (height + statsRowHeight) * scale;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.scale(scale, scale);
      ctx.fillStyle = "#111";
      ctx.fillRect(0, 0, width, height + statsRowHeight);
      ctx.drawImage(img, 0, 0, width, height);
      URL.revokeObjectURL(svgUrl);

      ctx.font = "12px monospace";
      ctx.fillStyle = "#ccc";
      histStatsLines.forEach((line, i) => {
        ctx.fillText(line, 10, height + 16 + i * lineHeight);
      });

      canvas.toBlob(blob => {
        if (!blob) return;
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = `size-distribution-${sizeMode}.png`;
        link.click();
        URL.revokeObjectURL(link.href);
      });
    };
    img.src = svgUrl;
  };

  // download the chart's svg directly, with the stats line appended as a text element
  const handleExportHistogramSVG = () => {
    const svgEl = chartWrapRef.current?.querySelector("svg");
    if (!svgEl) return;

    const clone = svgEl.cloneNode(true) as SVGSVGElement;
    const { width, height } = svgEl.getBoundingClientRect();
    const lineHeight = 18;
    const statsRowHeight = histStatsLines.length * lineHeight + 8;

    clone.setAttribute("width", `${width}`);
    clone.setAttribute("height", `${height + statsRowHeight}`);
    clone.setAttribute("viewBox", `0 0 ${width} ${height + statsRowHeight}`);

    const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    bg.setAttribute("x", "0");
    bg.setAttribute("y", `${height}`);
    bg.setAttribute("width", `${width}`);
    bg.setAttribute("height", `${statsRowHeight}`);
    bg.setAttribute("fill", "#fff");
    clone.appendChild(bg);

    histStatsLines.forEach((line, i) => {
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", "10");
      text.setAttribute("y", `${height + 16 + i * lineHeight}`);
      text.setAttribute("font-family", "monospace");
      text.setAttribute("font-size", "12");
      text.setAttribute("fill", "#111");
      text.textContent = line;
      clone.appendChild(text);
    });

    const svgString = new XMLSerializer().serializeToString(clone);
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([svgString], { type: "image/svg+xml;charset=utf-8" }));
    link.download = `size-distribution-${sizeMode}.svg`;
    link.click();
    URL.revokeObjectURL(link.href);
  };
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
  const histStd = sizeMode === "diameter"
    ? stats.size_stats.diameter_std
    : stats.size_stats.area_std;
  const histMedian = sizeMode === "diameter"
    ? stats.size_stats.diameter_median
    : stats.size_stats.area_median;
  const histUnit = sizeMode === "diameter" ? unit : `${unit}²`;

  // fit model name plus its params, eg "normal (mean=1.23, std=0.45)"
  const histFits = sizeMode === "diameter" ? stats.distribution_fits_diameter : stats.distribution_fits_area;
  const histFitLabel = (() => {
    if (!histFits?.reliable || !histFits.best_model || !histFits.fits) return null;
    const fit = histFits.fits[histFits.best_model];
    if (!fit) return null;
    const params = Object.entries(fit.params).map(([k, v]) => `${k}=${fmt(v, 4)}`).join(", ");
    return `${histFits.best_model} (${params})`;
  })();

  // stats summary drawn into the exports, split across two lines so the fit params don't overflow the chart width
  const histStatsLines = [
    `N: ${particles.length}`,
    `Mean: ${fmt(histMean)} ${histUnit}   Std: ${fmt(histStd)} ${histUnit}   Median: ${fmt(histMedian)} ${histUnit}`,
    histFitLabel ? `Fit: ${histFitLabel}` : null,
  ].filter((line): line is string => line !== null);

  //  shape pie data 
  const shapeData = useMemo(() => {
    return Object.entries(stats.shape_distribution).map(([name, data]) => ({
      name,
      value: data.count,
      fraction: data.fraction,
    }));
  }, [stats.shape_distribution]);

  //  sortable table 
  const sortedParticles = useMemo(() => {
    const indexed = particles.map((p, i) => ({ ...p, index: i + 1 }));
    indexed.sort((a, b) => {
      let va: number | string, vb: number | string;
      switch (sortKey) {
        case "index": va = a.index; vb = b.index; break;
        case "diameter": va = a.diameter_real ?? a.diameter_px; vb = b.diameter_real ?? b.diameter_px; break;
        case "area": va = a.area_real ?? a.area_px; vb = b.area_real ?? b.area_px; break;
        case "circularity": va = a.circularity; vb = b.circularity; break;
        case "aspect_ratio": va = a.aspect_ratio; vb = b.aspect_ratio; break;
        case "shape": va = a.shape; vb = b.shape; break;
        case "solidity": va = a.solidity ?? 1; vb = b.solidity ?? 1; break;
        case "n_vertices": va = a.n_vertices ?? 0; vb = b.n_vertices ?? 0; break;
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

  function toggleShape(shape: string) {
    setActiveShapes(prev => {
      const next = new Set(prev);
      if (next.has(shape)) {
        next.delete(shape);
      } else {
        next.add(shape);
      }
      return next;
    });
  }
  const filteredParticles = useMemo(() => {
    return particles.filter(p => activeShapes.has(p.shape));
  }, [particles, activeShapes]);

  function jitter(val: number, amount = 0.02): number {
    return val + (Math.random() - 0.5) * amount;
  }

  const scatterData = useMemo(() => {
      return particles
        .filter(p => activeShapes.has(p.shape))
        .map(p => ({
          ...p,
          aspect_ratio_j: p.aspect_ratio + ((((p.id ?? 0) * 7) % 100) / 100 - 0.5) * 0.05,
          circularity_j: p.circularity + ((((p.id ?? 0) * 13) % 100) / 100 - 0.5) * 0.03,
        }));
  }, [particles, activeShapes]);

  return (
    <div className={styles.root}>

      {/* header */}
      <header className={styles.header}>
        <button type="button" className={styles.backBtn} onClick={onBack}>
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
              <button type="button"
                className={`${styles.toggleBtn} ${sizeMode === "diameter" ? styles.toggleActive : ""}`}
                onClick={() => setSizeMode("diameter")}
              >Equivalent Diameter</button>
              <button type="button" 
                className={`${styles.toggleBtn} ${sizeMode === "area" ? styles.toggleActive : ""}`}
                onClick={() => setSizeMode("area")}
              >Area</button>
            </div>
          </div>

          <div className={styles.chartWrap} ref={chartWrapRef}>
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
          <div className={styles.exportRow}>
            <button type="button" className={styles.exportBtn} onClick={handleExportHistogramPNG}>
              <img src="/download-simple.svg" alt="" className={styles.exportIcon} />
              PNG
            </button>
            <button type="button" className={styles.exportBtn} onClick={handleExportHistogramSVG}>
              <img src="/download-simple.svg" alt="" className={styles.exportIcon} />
              SVG
            </button>
          </div>

          <div className={styles.chartStats}>
            <span>Mean: {fmt(histMean)} {histUnit}</span>
            <span>Std: {fmt(histStd)} {histUnit}</span>
            <span>Median: {fmt(histMedian)} {histUnit}</span>
            {histFits?.reliable && histFits.best_model && (
              <span style={{ color: "#e8c87e" }}>Fit: {histFits.best_model}</span>
            )}
          </div>
        </div>

        {/* shape distribution */}
        <div className={styles.chartCard}>
          <h2 className={styles.chartTitle}>Shape Analysis</h2>
 
          {/* scatter: circularity vs aspect ratio */}
          <div className={styles.chartWrap}>
            <ResponsiveContainer width="100%" height={280}>
              <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />
                <XAxis
                  dataKey="aspect_ratio_j"
                  type="number"
                  domain={['auto', 'auto']}
                  tick={{ fill: "#666", fontSize: 10 }}
                  label={{ value: "Aspect Ratio", position: "bottom", offset: 10, fill: "#555", fontSize: 11 }}
                />
                <YAxis
                  dataKey="circularity_j"
                  type="number"
                  domain={[0, 1]}
                  tick={{ fill: "#666", fontSize: 10 }}
                  label={{ value: "Circularity", angle: -90, position: "insideLeft", offset: 0, fill: "#555", fontSize: 11 }}
                />
                <ZAxis dataKey="solidity" range={[30, 150]} />
                <Tooltip
                  content={({ active, payload }: any) => {
                    if (!active || !payload?.length) return null;
                    const d = payload[0].payload;
                    return (
                      <div className={styles.tooltip}>
                        <p>#{d.id} — {d.shape}</p>
                        <p>Circularity: {(d.circularity ?? 0).toFixed(3)}</p>
                        <p>Aspect Ratio: {(d.aspect_ratio ?? 0).toFixed(3)}</p>
                        <p>Solidity: {(d.solidity ?? 0).toFixed(3)}</p>
                      </div>
                    );
                  }}
                  cursor={{ strokeDasharray: "3 3" }}
                />
                {Object.entries(
                  scatterData.reduce((acc: Record<string, any[]>, p) => {
                    const s = p.shape;
                    if (!acc[s]) acc[s] = [];
                    acc[s].push(p);
                    return acc;
                  }, {})
                ).map(([shapeName, pts]) => (
                  <Scatter
                    key={shapeName}
                    name={shapeName}
                    data={pts}
                    fill={SHAPE_COLORS[shapeName] ?? "#888"}
                    opacity={activeShapes.size === 0 || activeShapes.has(shapeName) ? 0.7 : 0.1}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
 
          {/* shape distribution bars + legend */}
          <div className={styles.shapeLegend}>
            {shapeData.map(entry => {
              const isActive = activeShapes.has(entry.name);
              return (
                <div
                  key={entry.name}
                  className={`${styles.legendItem} ${isActive ? styles.legendItemActive : ""}`}
                  onClick={() => toggleShape(entry.name)}
                  style={{ cursor: "pointer" }}
                >
                  <span
                    className={styles.legendDot}
                    style={{ background: SHAPE_COLORS[entry.name] ?? "#888" }}
                  />
                  <span className={styles.legendLabel}>
                    {entry.name}
                  </span>
                  <span className={styles.legendVal}>
                    {entry.value} ({(entry.fraction * 100).toFixed(1)}%)
                  </span>
                </div>
              );
            })}
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
              <Crosshair size={10} /> Click a row to locate particle or click a shape type to locate all particles of same shape on canvas
            </span>
              
          )}
        </div>
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                {TABLE_COLUMN_KEYS.map(key => (
                  <th key={key} onClick={() => handleSort(key)} className={styles.th}>
                    {columnLabel(key, hasScale, unit)} <SortIcon col={key} />
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
                  <td className={styles.td}>{fmt(p.circularity)}</td>
                  <td className={styles.td}>{fmt(p.solidity ?? 1)}</td>
                  <td className={styles.td}>{fmt(p.aspect_ratio)}</td>
                  <td className={styles.td}>{p.n_vertices ?? "—"}</td>
                  <td className={styles.td}>
                    <span
                      className={`${styles.shapeBadge} ${onLocateShape ? styles.shapeBadgeClickable : ""}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        onLocateShape?.(p.shape);
                      }}
                      style={{ borderColor: SHAPE_COLORS[p.shape] ?? "#888", color: SHAPE_COLORS[p.shape] ?? "#888" }}
                    >
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
