"use client";

import { useState, useEffect } from "react";
import { Check, X, Pencil, Ruler } from "lucide-react";
import styles from "./CalibrateSection.module.css";
import type { Metadata, StatsResult } from "@/lib/api";
import { updatePixelSize } from "@/lib/api";

interface Props {
  sessionId: string | null;
  metadata: Metadata | null;
  scaleBarMode: boolean;
  scaleBarPixels: number | null;
  onMetadataUpdate?: (metadata: Metadata, stats?: StatsResult) => void;
  onToggleScaleBar: () => void;
  onScaleBarCancel: () => void;
}

export default function CalibrateSection({
  sessionId,
  metadata,
  scaleBarMode,
  scaleBarPixels,
  onMetadataUpdate,
  onToggleScaleBar,
  onScaleBarCancel,
}: Props) {
  const [editingPixel, setEditingPixel] = useState(false);
  const [editSize, setEditSize] = useState("");
  const [editUnit, setEditUnit] = useState("nm");
  const [pixelBusy, setPixelBusy] = useState(false);
  const [sbLength, setSbLength] = useState("");
  const [sbUnit, setSbUnit] = useState("nm");
  const [calibrationError, setCalibrationError] = useState("");

  useEffect(() => {
    if (scaleBarPixels != null) {
      setSbLength("");
      setSbUnit(metadata?.pixel_unit ?? "nm");
    }
  }, [scaleBarPixels, metadata?.pixel_unit]);

  const pixelSize = metadata?.pixel_size;
  const canEditPixel = sessionId != null && metadata != null;
  const pixelSizeMissing = pixelSize == null || pixelSize === "-";

  function startEdit() {
    if (!canEditPixel) return;
    setCalibrationError("");
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
    if (!Number.isFinite(val) || val <= 0) {
      setCalibrationError("Enter a positive pixel size.");
      return;
    }
    setCalibrationError("");
    setPixelBusy(true);
    try {
      const result = await updatePixelSize(
        sessionId,
        { pixel_size: val, pixel_unit: editUnit.trim() || "nm", calibration_source: "manual" },
      );
      if (result.metadata) {
        onMetadataUpdate?.(
          result.metadata as Metadata,
          result.stats as StatsResult | undefined,
        );
      }
      setEditingPixel(false);
    } catch (e) {
      setCalibrationError(e instanceof Error ? e.message : "Could not save pixel size.");
    } finally {
      setPixelBusy(false);
    }
  }

  async function confirmScaleBar() {
    if (!sessionId || scaleBarPixels == null) return;
    const length = parseFloat(sbLength);
    if (!Number.isFinite(length) || length <= 0) {
      setCalibrationError("Enter a positive scale bar length.");
      return;
    }
    const computed = length / scaleBarPixels;
    setCalibrationError("");
    setPixelBusy(true);
    try {
      const result = await updatePixelSize(
        sessionId,
        {
          pixel_size: computed,
          pixel_unit: sbUnit.trim() || "nm",
          calibration_source: "scale_bar",
          scale_bar_pixels: scaleBarPixels,
          scale_bar_length: length,
        },
      );
      if (result.metadata) {
        onMetadataUpdate?.(
          result.metadata as Metadata,
          result.stats as StatsResult | undefined,
        );
      }
      onScaleBarCancel();
    } catch (e) {
      setCalibrationError(e instanceof Error ? e.message : "Could not save scale bar.");
    } finally {
      setPixelBusy(false);
    }
  }

  return (
    <div className={styles.wrap}>
      {metadata && (
        <div className={styles.row}>
          <span className={styles.label}>Pixel Size</span>
          {!editingPixel ? (
            <span className={styles.valFlex}>
              <span className={styles.val}>
                {pixelSizeMissing
                  ? "Not Found"
                  : typeof pixelSize === "number"
                    ? pixelSize.toFixed(4)
                    : "Not Found"}
                {metadata.pixel_unit == null || metadata.pixel_unit === "-"
                  ? ""
                  : " " + metadata.pixel_unit}
              </span>
              {metadata.calibration?.source && (
                <span className={styles.label} title="Source of the current pixel scale">
                  {metadata.calibration.source === "scale_bar"
                    ? "Scale bar"
                    : metadata.calibration.source === "embedded"
                      ? "Image metadata"
                      : "Manual"}
                </span>
              )}
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
            <span className={styles.valFlex}>
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
      {metadata?.calibration_warning && (
        <div className={styles.row}>{metadata.calibration_warning}</div>
      )}
      {calibrationError && <div className={styles.row} role="alert">{calibrationError}</div>}

      {scaleBarPixels == null && (
        <div className={styles.row}>
          <span
            className={styles.label}
            style={{ color: scaleBarMode ? "var(--accent)" : undefined }}
          >
            {scaleBarMode ? "Draw line →" : "Scale Bar"}
          </span>
          <span className={styles.valFlex}>
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
        <div className={styles.confirmRow}>
          <span className={styles.label} style={{ color: "var(--accent)" }}>
            {scaleBarPixels.toFixed(1)} px — enter length:
          </span>
          <span className={styles.valFlex}>
            <input
              type="number"
              step="any"
              min="0"
              value={sbLength}
              onChange={(e) => setSbLength(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") confirmScaleBar();
                if (e.key === "Escape") onScaleBarCancel();
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
                if (e.key === "Escape") onScaleBarCancel();
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
  );
}
