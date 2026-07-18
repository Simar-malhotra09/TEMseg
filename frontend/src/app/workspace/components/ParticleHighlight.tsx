"use client";

import { Instance } from "@/lib/api";
import styles from "./ParticleHighlight.module.css";

interface Props {
  instance: Instance;
  imgWidth: number;
  imgHeight: number;
  viewportWidth: number;
  viewportHeight: number;
}

/**
 * SVG overlay that highlights a single particle with a bright pulsing contour.
 * Positioned absolutely over the image viewport.
 */
export default function ParticleHighlight({
  instance,
  imgWidth,
  imgHeight,
  viewportWidth,
  viewportHeight,
}: Props) {
  if (!instance.contour || instance.contour.length < 3) return null;
  if (imgWidth <= 0 || imgHeight <= 0) return null;

  const points = instance.contour.map(([x, y]) => `${x},${y}`).join(" ");

  // center of bbox for the ID label
  const cx = instance.bbox.x + instance.bbox.w / 2;
  const cy = instance.bbox.y + instance.bbox.h / 2;

  // viewBox is in image-pixel space but gets scaled down to fit the viewport.
  // size stroke/glow/label in screen pixels first, then convert back to
  // viewBox units, so they stay legible regardless of image resolution.
  const scale = Math.min(viewportWidth / imgWidth, viewportHeight / imgHeight) || 1;
  const toImg = (screenPx: number) => screenPx / scale;

  const strokeWidth = toImg(2.5);
  const glowStdDeviation = toImg(3);

  // scale label to particle size (in screen space) so it doesn't dwarf small particles
  const bboxMinScreen = Math.min(instance.bbox.w, instance.bbox.h) * scale;
  const labelR = toImg(Math.max(6, Math.min(12, bboxMinScreen * 0.35)));
  const fontSize = toImg(Math.max(9, Math.min(14, bboxMinScreen * 0.45)));

  return (
    <svg
      className={styles.overlay}
      viewBox={`0 0 ${imgWidth} ${imgHeight}`}
      style={{ width: viewportWidth, height: viewportHeight }}
      preserveAspectRatio="xMidYMid meet"
    >
      {/* glow effect */}
      <defs>
        <filter id="highlight-glow">
          <feGaussianBlur stdDeviation={glowStdDeviation} result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* filled highlight area */}
      <polygon
        points={points}
        fill="#7ee8a2"
        fillOpacity={0.15}
        stroke="none"
      />

      {/* bright contour outline */}
      <polygon
        points={points}
        fill="none"
        stroke="#7ee8a2"
        strokeWidth={strokeWidth}
        filter="url(#highlight-glow)"
        className={styles.pulse}
      />

      {/* ID label */}
      <circle cx={cx} cy={cy} r={labelR} fill="#0d0d0d" fillOpacity={0.8} stroke="#7ee8a2" strokeWidth={1} />
      <text
        x={cx}
        y={cy}
        textAnchor="middle"
        dominantBaseline="central"
        fill="#7ee8a2"
        fontSize={fontSize}
        fontWeight={700}
        fontFamily="monospace"
      >
        {instance.id}
      </text>
    </svg>
  );
}
