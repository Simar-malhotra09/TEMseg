"use client";

import { useRef, useCallback, useMemo, useState, useEffect, useLayoutEffect } from "react";
import { Instance, ParticleStats, ParticleMetricField, StatsResult } from "@/lib/api";
import { ViewBox } from "../hooks/useRefineState";

interface Props {
  // display
  imageSrc: string;
  width: number;
  height: number;
  imgWidth: number;
  imgHeight: number;

  // data is rendered as-is, no local copy
  instances: Instance[];
  selectedId: number | null;
  viewBox: ViewBox;
  splitMode: boolean;
  splitPoints: [number, number][];
  pasteMode?: boolean;
  clipboard?: Instance | null;
  polygonOpacity?: number;
  stats?: StatsResult | null;
  visibleTooltipFields?: ParticleMetricField[];

  // local events ;all business logic handled by parent via useRefineState
  onSelect: (id: number) => void;
  onDeselect: () => void;
  onVertexDragEnd: (instId: number, vi: number, pos: [number, number]) => void;
  onVertexDelete: (instId: number, vi: number) => void;
  onEdgeClick: (instId: number, pos: [number, number]) => void;
  onSplitPointPlace: (pos: [number, number]) => void;
  onPastePlace?: (pos: [number, number]) => void;
  onRotateStart?: () => void;
  onRotateDrag?: (pos: [number, number]) => void;
  onRotateEnd?: () => void;
  onViewBoxChange: (vb: ViewBox) => void;
}

const COLORS = [
  "#ff4444", 
  "#44ff44", "#4488ff", "#ffff44",
  "#ff44ff", "#44ffff", "#ff8844", "#8844ff",
  "#44ff88", "#ff4488",
];

// short-form label for one tooltip field, mirrors the values shown in the
// per-particle stats table
function formatTooltipField(key: ParticleMetricField, p: ParticleStats, hasScale: boolean, unit: string): string {
  switch (key) {
    case "diameter": {
      const v = hasScale && p.diameter_real != null ? p.diameter_real : p.diameter_px;
      return `Diameter: ${v.toFixed(2)} ${hasScale ? unit : "px"}`;
    }
    case "area": {
      const v = hasScale && p.area_real != null ? p.area_real : p.area_px;
      return `Area: ${v.toFixed(2)} ${hasScale ? unit + "²" : "px²"}`;
    }
    case "circularity": return `Circ.: ${p.circularity.toFixed(2)}`;
    case "solidity": return `Solidity: ${p.solidity.toFixed(2)}`;
    case "aspect_ratio": return `Asp. Ratio: ${p.aspect_ratio.toFixed(2)}`;
    case "n_vertices": return `Vertices: ${p.n_vertices}`;
    case "shape": return `Shape: ${p.shape}`;
  }
}

const MIN_ZOOM = 1;
const MAX_ZOOM = 20;
const VERTEX_RADIUS = 5;  // constant screen-space px
const EDGE_HIT_WIDTH = 8; // constant screen-space px

export default function RefineCanvas({
  imageSrc, width, height, imgWidth, imgHeight,
  instances, selectedId, viewBox, splitMode, splitPoints,
  pasteMode = false,
  clipboard = null,
  polygonOpacity = 0.2,
  stats = null,
  visibleTooltipFields = [],
  onSelect, onDeselect, onVertexDragEnd, onVertexDelete,
  onEdgeClick, onSplitPointPlace, onPastePlace,
  onRotateStart, onRotateDrag, onRotateEnd,
  onViewBoxChange,
}: Props) {

  // pure interaction refs 
  const svgRef = useRef<SVGSVGElement>(null);
  const isSpaceDown = useRef(false);
  const panStart = useRef<{ mx: number; my: number; vx: number; vy: number } | null>(null);
  // tracks which vertex is being dragged and its live position during drag
  const draggingVertex = useRef<{ instId: number; vi: number; pos: [number, number] } | null>(null);
  // tracks rotation handle drag
  const rotating = useRef(false);
  // local render-only state for smooth vertex drag 
  const [dragPos, setDragPos] = useLocalDragState();
  // mouse position in image space for paste-mode ghost preview
  const [pasteCursor, setPasteCursor] = useState<[number, number] | null>(null);

  // hover-driven ID tooltip. position is in screen space (px, relative to the
  // svg's own top-left), decoupled from selection so it reflects whatever the
  // cursor is currently over, not what's selected for editing
  const [hover, setHover] = useState<{ id: number; x: number; y: number } | null>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [tooltipPos, setTooltipPos] = useState<{ left: number; top: number } | null>(null);

  // clamp the tooltip so it never spills outside the canvas viewport. 
  // flip to the opposite side of the cursor if the default offset would overflow
  useLayoutEffect(() => {
    if (!hover || !tooltipRef.current) {
      setTooltipPos(null);
      return;
    }
    const tw = tooltipRef.current.offsetWidth;
    const th = tooltipRef.current.offsetHeight;
    const OFFSET = 14;
    let left = hover.x + OFFSET;
    let top = hover.y + OFFSET;
    if (left + tw > width) left = hover.x - OFFSET - tw;
    if (top + th > height) top = hover.y - OFFSET - th;
    left = Math.max(0, Math.min(left, width - tw));
    top = Math.max(0, Math.min(top, height - th));
    setTooltipPos({ left, top });
  }, [hover, width, height]);

  // scale factor: image pixels per screen pixel at current zoom
  // used to keep stroke widths and vertex sizes visually constant
  const s2i = viewBox.w / width;

  // convert screen coords to image-space via SVG matrix
  // immune to any CSS transforms on parent elements
  const toImageSpace = useCallback((clientX: number, clientY: number): [number, number] => {
    const svg = svgRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    const { x, y } = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    return [x, y];
  }, []);

  // keyboard: space = pan, backspace/delete = delete selected
  // note: delete is handled in parent via useRefineState. The canvas just needs space for pan
  useSpaceKey(svgRef, isSpaceDown);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const [mx, my] = toImageSpace(e.clientX, e.clientY);
    const factor = e.deltaY > 0 ? 1.1 : 0.9;
    const newW = Math.min(Math.max(viewBox.w * factor, imgWidth / MAX_ZOOM), imgWidth / MIN_ZOOM);
    const newH = Math.min(Math.max(viewBox.h * factor, imgHeight / MAX_ZOOM), imgHeight / MIN_ZOOM);
    onViewBoxChange({
      w: newW, h: newH,
      x: Math.min(Math.max(mx - (mx - viewBox.x) * (newW / viewBox.w), 0), imgWidth - newW),
      y: Math.min(Math.max(my - (my - viewBox.y) * (newH / viewBox.h), 0), imgHeight - newH),
    });
  }, [viewBox, imgWidth, imgHeight, toImageSpace, onViewBoxChange]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (isSpaceDown.current) {
      e.preventDefault();
      panStart.current = { mx: e.clientX, my: e.clientY, vx: viewBox.x, vy: viewBox.y };
      if (svgRef.current) svgRef.current.style.cursor = "grabbing";
      return;
    }
    const target = e.target as Element;
    if (target.getAttribute("data-rotate-handle") === "true") {
      e.stopPropagation();
      rotating.current = true;
      if (onRotateStart) onRotateStart();
      return;
    }
    if (pasteMode) {
      if (onPastePlace) onPastePlace(toImageSpace(e.clientX, e.clientY));
      return;
    }
    if (splitMode) {
      onSplitPointPlace(toImageSpace(e.clientX, e.clientY));
      return;
    }
    const tag = target.tagName;
    if (tag === "svg" || tag === "image") onDeselect();
  }, [viewBox, pasteMode, splitMode, toImageSpace, onPastePlace, onSplitPointPlace, onDeselect, onRotateStart]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (panStart.current) {
      const { vx, vy, mx, my } = panStart.current;
      onViewBoxChange({
        ...viewBox,
        x: Math.min(Math.max(vx - (e.clientX - mx) / width * viewBox.w, 0), imgWidth - viewBox.w),
        y: Math.min(Math.max(vy - (e.clientY - my) / height * viewBox.h, 0), imgHeight - viewBox.h),
      });
      return;
    }
    if (rotating.current) {
      if (onRotateDrag) onRotateDrag(toImageSpace(e.clientX, e.clientY));
      return;
    }
    if (pasteMode) {
      setPasteCursor(toImageSpace(e.clientX, e.clientY));
      return;
    }
    if (draggingVertex.current) {
      const pos = toImageSpace(e.clientX, e.clientY);
      draggingVertex.current.pos = pos;
      // update local render state for smooth drag — parent not notified until mouseup
      setDragPos({ instId: draggingVertex.current.instId, vi: draggingVertex.current.vi, pos });
    }
  }, [viewBox, width, height, imgWidth, imgHeight, pasteMode, toImageSpace, onViewBoxChange, onRotateDrag, setDragPos]);

  const handleMouseUp = useCallback(() => {
    if (draggingVertex.current) {
      // commit final position to parent only on release
      const { instId, vi, pos } = draggingVertex.current;
      onVertexDragEnd(instId, vi, pos);
      setDragPos(null);
    }
    if (rotating.current) {
      rotating.current = false;
      if (onRotateEnd) onRotateEnd();
    }
    draggingVertex.current = null;
    panStart.current = null;
    if (svgRef.current) svgRef.current.style.cursor = isSpaceDown.current ? "grab" : pasteMode ? "copy" : "default";
  }, [onVertexDragEnd, setDragPos, onRotateEnd, pasteMode]);

  const pointsStr = (contour: [number, number][]) =>
    contour.map(([x, y]) => `${x},${y}`).join(" ");

  // get live vertex position — use drag position during drag, contour otherwise
  const getVertexPos = (instId: number, vi: number, contourPt: [number, number]): [number, number] => {
    if (dragPos && dragPos.instId === instId && dragPos.vi === vi) return dragPos.pos;
    return contourPt;
  };

  // get live points string — applies drag offset to the dragged vertex only
  const getLivePointsStr = (inst: Instance): string => {
    if (!dragPos || dragPos.instId !== inst.id) return pointsStr(inst.contour);
    return inst.contour.map((pt, i) => {
      const [x, y] = getVertexPos(inst.id, i, pt);
      return `${x},${y}`;
    }).join(" ");
  };

  const hoveredInstance = hover ? instances.find(i => i.id === hover.id) : null;
  const particleStatsById = useMemo(
    () => new Map((stats?.particles ?? []).map(p => [p.id, p])),
    [stats]
  );
  const hoveredParticle = hoveredInstance ? particleStatsById.get(hoveredInstance.id) : null;

  return (
    <>
    <svg
      ref={svgRef}
      width={width}
      height={height}
      viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`}
      style={{ display: "block", userSelect: "none", cursor: pasteMode ? "copy" : splitMode ? "crosshair" : "default" }}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      <image
        href={imageSrc}
        x={0} y={0}
        width={imgWidth}
        height={imgHeight}
        style={{ imageRendering: "crisp-edges" }}
      />

      {instances.map((inst) => {
        const color = COLORS[inst.id % COLORS.length];
        const isSelected = inst.id === selectedId;
        const pts = getLivePointsStr(inst);

        return (
          <g key={inst.id}>
            <polygon
              points={pts}
              fill={color}
              fillOpacity={polygonOpacity}
              stroke={color}
              strokeWidth={isSelected ? 2.5 * s2i : 1.5 * s2i}
              opacity={isSelected ? 1 : 0.7}
              style={{ cursor: splitMode ? "crosshair" : "pointer" }}
              onClick={e => {
                e.stopPropagation();
                if (!splitMode) onSelect(inst.id);
              }}
              onMouseMove={e => {
                const rect = svgRef.current!.getBoundingClientRect();
                setHover({ id: inst.id, x: e.clientX - rect.left, y: e.clientY - rect.top });
              }}
              onMouseLeave={() => setHover(h => (h?.id === inst.id ? null : h))}
            />

            {isSelected && !splitMode && (
              <>
                {/* wider invisible stroke over edges — click to insert vertex */}
                <polygon
                  points={pts}
                  fill="none"
                  stroke="transparent"
                  strokeWidth={EDGE_HIT_WIDTH * 2 * s2i}
                  style={{ cursor: "cell" }}
                  onClick={e => {
                    e.stopPropagation();
                    onEdgeClick(inst.id, toImageSpace(e.clientX, e.clientY));
                  }}
                />

                {/* vertex handles */}
                {inst.contour.map((pt, vi) => {
                  const [x, y] = getVertexPos(inst.id, vi, pt);
                  return (
                    <circle
                      key={`v-${inst.id}-${vi}`}
                      cx={x} cy={y}
                      r={VERTEX_RADIUS * s2i}
                      fill={color}
                      stroke="#fff"
                      strokeWidth={1.5 * s2i}
                      style={{ cursor: "move" }}
                      onMouseDown={e => {
                        e.stopPropagation();
                        draggingVertex.current = { instId: inst.id, vi, pos: pt };
                      }}
                      onDoubleClick={e => {
                        e.stopPropagation();
                        onVertexDelete(inst.id, vi);
                      }}
                    />
                  );
                })}

                {/* rotation handle, anchored to centroid */}
                {(() => {
                  const cx = inst.contour.reduce((s, [x]) => s + x, 0) / inst.contour.length;
                  const cy = inst.contour.reduce((s, [, y]) => s + y, 0) / inst.contour.length;
                  // place handle above centroid, scaled by bbox size
                  const handleDist = Math.max(inst.bbox.h, inst.bbox.w) * 0.6 + 20;
                  const hx = cx;
                  const hy = cy - handleDist;
                  return (
                    <g>
                      {/* dashed line from centroid to handle */}
                      <line
                        x1={cx} y1={cy}
                        x2={hx} y2={hy}
                        stroke="#fff"
                        strokeWidth={1 * s2i}
                        strokeDasharray={`${3 * s2i},${2 * s2i}`}
                        opacity={0.6}
                        style={{ pointerEvents: "none" }}
                      />
                      {/* rotation handle circle */}
                      <circle
                        data-rotate-handle="true"
                        cx={hx} cy={hy}
                        r={VERTEX_RADIUS * 1.5 * s2i}
                        fill="#fff"
                        stroke={color}
                        strokeWidth={1.5 * s2i}
                        style={{ cursor: "grab" }}
                      />
                    </g>
                  );
                })()}
              </>
            )}
          </g>
        );
      })}

      {/* paste mode ghost. The polygon follows the cursor  */}
      {pasteMode && clipboard && pasteCursor && (
        <g style={{ pointerEvents: "none" }}>
          {(() => {
            const [px, py] = pasteCursor;
            const cx = clipboard.contour.reduce((s, [x]) => s + x, 0) / clipboard.contour.length;
            const cy = clipboard.contour.reduce((s, [, y]) => s + y, 0) / clipboard.contour.length;
            const ghostPts = clipboard.contour.map(([x, y]) => `${x + (px - cx)},${y + (py - cy)}`).join(" ");
            return (
              <polygon
                points={ghostPts}
                fill="none"
                stroke="#fff"
                strokeWidth={2 * s2i}
                strokeDasharray={`${4 * s2i},${3 * s2i}`}
                opacity={0.8}
              />
            );
          })()}
        </g>
      )}

      {/* split point markers  */}
      {splitMode && splitPoints.map(([x, y], i) => (
        <g key={`sp-${i}`} style={{ pointerEvents: "none" }}>
          <circle cx={x} cy={y} r={10 * s2i} fill="none" stroke="#fff" strokeWidth={2 * s2i} opacity={0.8} />
          <circle cx={x} cy={y} r={4 * s2i} fill="#fff" opacity={0.9} />
          <text x={x + 12 * s2i} y={y + 4 * s2i} fontSize={12 * s2i} fill="#fff">{i + 1}</text>
        </g>
      ))}
    </svg>

    {/* hover ID tooltip. It lives outside SVG image-space so it's never
        squeezed by particle size or zoom; position clamped to the viewport
        in the layout effect above */}
    {hoveredInstance && (
      <div
        ref={tooltipRef}
        style={{
          position: "absolute",
          left: tooltipPos?.left ?? 0,
          top: tooltipPos?.top ?? 0,
          visibility: tooltipPos ? "visible" : "hidden",
          background: "#161616",
          border: "1px solid #2a2a2a",
          borderRadius: 4,
          padding: "4px 8px",
          fontSize: 11,
          fontFamily: "monospace",
          color: "#e8e6e1",
          pointerEvents: "none",
          whiteSpace: "nowrap",
          zIndex: 20,
        }}
      >
        <div>ID: {hoveredInstance.id}</div>
        {hoveredParticle && visibleTooltipFields.map(key => (
          <div key={key}>
            {formatTooltipField(key, hoveredParticle, stats?.has_scale ?? false, stats?.unit ?? "px")}
          </div>
        ))}
      </div>
    )}
    </>
  );
}

// local drag position; smooth vertex rendering during drag, never sent to parent
// parent only receives final position on mouseup via onVertexDragEnd
function useLocalDragState() {
  const [dragPos, setDragPos] = useState<{ instId: number; vi: number; pos: [number, number] } | null>(null);
  return [dragPos, setDragPos] as const;
}

// space key — pan mode cursor management only, no business logic
function useSpaceKey(
  svgRef: React.RefObject<SVGSVGElement | null>,
  isSpaceDown: React.MutableRefObject<boolean>
) {
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && !e.repeat) {
        e.preventDefault();
        isSpaceDown.current = true;
        if (svgRef.current) svgRef.current.style.cursor = "grab";
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        isSpaceDown.current = false;
        if (svgRef.current) svgRef.current.style.cursor = "default";
      }
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, [isSpaceDown, svgRef]);
}
