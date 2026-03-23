"use client";

import { useState, useRef, useEffect } from "react";
import { Download, X } from "lucide-react";
import styles from "./ExportPanel.module.css";
import { exportSession, isPyWebView, exportViaPyWebView } from "@/lib/api";

export type ExportItem =
  | "original_image"
  | "seg_mask_png"
  | "seg_mask_npy"
  | "refined_mask_png"
  | "refined_mask_npy"
  | "instances_json"
  | "stats_csv";

interface ExportOption {
  id: ExportItem;
  label: string;
  hint: string;
  available: boolean;
  unavailableReason?: string;
}

interface Props {
  sessionId: string;
  segDone: boolean;
  refineDone: boolean;  // true if user has saved refinements at least once
  hasStats: boolean;
}

export default function ExportPanel({ sessionId, segDone, refineDone, hasStats }: Props) {
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<Set<ExportItem>>(new Set());
  const [downloading, setDownloading] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

  // close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const options: ExportOption[] = [
    {
      id: "original_image",
      label: "Original image",
      hint: "as uploaded",
      available: true,
    },
    {
      id: "seg_mask_png",
      label: "Segmentation mask",
      hint: "PNG",
      available: segDone,
      unavailableReason: "run segmentation first",
    },
    {
      id: "seg_mask_npy",
      label: "Segmentation mask",
      hint: "NPY array",
      available: segDone,
      unavailableReason: "run segmentation first",
    },
    {
      id: "refined_mask_png",
      label: "Refined mask",
      hint: "PNG",
      available: refineDone,
      unavailableReason: "save refinements first",
    },
    {
      id: "refined_mask_npy",
      label: "Refined mask",
      hint: "NPY array",
      available: refineDone,
      unavailableReason: "save refinements first",
    },
    {
      id: "instances_json",
      label: "Instance contours",
      hint: "JSON",
      available: segDone,
      unavailableReason: "run segmentation first",
    },
    {
      id: "stats_csv",
      label: "Stats",
      hint: "CSV",
      available: hasStats,
      unavailableReason: "run segmentation first",
    },
  ];

  function toggle(id: ExportItem, available: boolean) {
    if (!available) return;
    setSelected(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  function selectAll() {
    setSelected(new Set(options.filter(o => o.available).map(o => o.id)));
  }

  async function handleDownload() {
    if (selected.size === 0 || downloading) return;
    setDownloading(true);
    try {
      const items = Array.from(selected);

      if (isPyWebView()) {
        // Native save dialog via PyWebView bridge
        const result = await exportViaPyWebView(sessionId, items);
        if (!result.success && result.error !== "cancelled") {
          console.error("Export failed:", result.error);
        }
      } else {
        // Browser: blob download
        const blob = await exportSession(sessionId, items);
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `temseg_export_${sessionId}.zip`;
        a.click();
        URL.revokeObjectURL(url);
      }

      setOpen(false);
    } catch (err) {
      console.error("Export failed:", err);
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div ref={panelRef} style={{ position: "relative", width: "100%" }}>
      <button
        className={`${styles.triggerBtn} ${!segDone ? styles.triggerBtnDisabled : ""}`}
        onClick={() => segDone && setOpen(o => !o)}
        disabled={!segDone}
      >
        <Download size={14} /> Export
      </button>

      {open && (
        <div className={styles.panel}>
          <div className={styles.panelHeader}>
            <span className={styles.panelTitle}>Export</span>
            <button className={styles.closeBtn} onClick={() => setOpen(false)}>
              <X size={12} />
            </button>
          </div>

          <div className={styles.options}>
            {options.map(opt => (
              <label
                key={opt.id}
                className={`${styles.option} ${!opt.available ? styles.optionDisabled : ""}`}
                title={!opt.available ? opt.unavailableReason : undefined}
              >
                <input
                  type="checkbox"
                  checked={selected.has(opt.id)}
                  disabled={!opt.available}
                  onChange={() => toggle(opt.id, opt.available)}
                  className={styles.checkbox}
                />
                <span className={styles.optionLabel}>{opt.label}</span>
                <span className={styles.optionHint}>{opt.hint}</span>
              </label>
            ))}
          </div>

          <div className={styles.panelFooter}>
            <button className={styles.selectAllBtn} onClick={selectAll}>
              select all
            </button>
            <button
              className={styles.downloadBtn}
              disabled={selected.size === 0 || downloading}
              onClick={handleDownload}
            >
              {downloading ? "Preparing..." : `Download ZIP (${selected.size})`}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
