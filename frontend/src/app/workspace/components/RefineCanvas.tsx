"use client";

import { useRef, useCallback, useState, useEffect } from "react";
import { Instance } from "@/lib/api";
import { ViewBox } from "../hooks/useRefineState";

interface Props {
  // display
  imageSrc: string;
  width: number;
  height: number;
  imgWidth: number;
  imgHeight: number;

  // data — rendered as-is, no local copy
  instances: Instance[];
  selectedId: number | null;
  viewBox: ViewBox;
  splitMode: boolean;
  splitPoints: [number, number][];

  // events — all business logic handled by parent via useRefineState
  onSelect: (id: number) => void;
  onDeselect: () => void;
  onVertexDragEnd: (instId: number, vi: number, pos: [number, number]) => void;
  onVertexDelete: (instId: number, vi: number) => void;
  onEdgeClick: (instId: number, pos: [number, number]) => void;
  onSplitPointPlace: (pos: [number, number]) => void;
  onViewBoxChange: (vb: ViewBox) => void;
}

const COLORS = [
  "#ff4444", "#44ff44", "#4488ff", "#ffff44",
  "#ff44ff", "#44ffff", "#ff8844", "#8844ff",
  "#44ff88", "#ff4488",
];

const MIN_ZOOM = 1;
const MAX_ZOOM = 20;
const VERTEX_RADIUS = 5;  // constant screen-space px
const EDGE_HIT_WIDTH = 8; // constant screen-space px

export default function RefineCanvas({
  imageSrc, width, height, imgWidth, imgHeight,
  instances, selectedId, viewBox, splitMode, splitPoints,
  onSelect, onDeselect, onVertexDragEnd, onVertexDelete,
  onEdgeClick, onSplitPointPlace, onViewBoxChange,
}: Props) {

  // pure interaction refs — no business logic, just mechanics
  const svgRef = useRef<SVGSVGElement>(null);
  const isSpaceDown = useRef(false);
  const panStart = useRef<{ mx: number; my: number; vx: number; vy: number } | null>(null);
  // tracks which vertex is being dragged and its live position during drag
  const draggingVertex = useRef<{ instId: number; vi: number; pos: [number, number] } | null>(null);
  // local render-only state for smooth vertex drag — does not go to parent until mouseup
  const [dragPos, setDragPos] = useLocalDragState();

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
  // note: delete is handled in parent via useRefineState — canvas just needs space for pan
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
    if (splitMode) {
      onSplitPointPlace(toImageSpace(e.clientX, e.clientY));
      return;
    }
    const tag = (e.target as Element).tagName;
    if (tag === "svg" || tag === "image") onDeselect();
  }, [viewBox, splitMode, toImageSpace, onSplitPointPlace, onDeselect]);

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
    if (draggingVertex.current) {
      const pos = toImageSpace(e.clientX, e.clientY);
      draggingVertex.current.pos = pos;
      // update local render state for smooth drag — parent not notified until mouseup
      setDragPos({ instId: draggingVertex.current.instId, vi: draggingVertex.current.vi, pos });
    }
  }, [viewBox, width, height, imgWidth, imgHeight, toImageSpace, onViewBoxChange, setDragPos]);

  const handleMouseUp = useCallback(() => {
    if (draggingVertex.current) {
      // commit final position to parent only on release
      const { instId, vi, pos } = draggingVertex.current;
      onVertexDragEnd(instId, vi, pos);
      setDragPos(null);
    }
    draggingVertex.current = null;
    panStart.current = null;
    if (svgRef.current) svgRef.current.style.cursor = isSpaceDown.current ? "grab" : "default";
  }, [onVertexDragEnd, setDragPos]);

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

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`}
      style={{ display: "block", userSelect: "none", cursor: splitMode ? "crosshair" : "default" }}
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

      {instances.map((inst, i) => {
        const color = COLORS[i % COLORS.length];
        const isSelected = inst.id === selectedId;
        const pts = getLivePointsStr(inst);

        return (
          <g key={inst.id}>
            <polygon
              points={pts}
              fill={`${color}33`}
              stroke={color}
              strokeWidth={isSelected ? 2.5 * s2i : 1.5 * s2i}
              opacity={isSelected ? 1 : 0.7}
              style={{ cursor: splitMode ? "crosshair" : "pointer" }}
              onClick={e => {
                e.stopPropagation();
                if (!splitMode) onSelect(inst.id);
              }}
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
              </>
            )}
          </g>
        );
      })}

      {/* split point markers — on top of everything */}
      {splitMode && splitPoints.map(([x, y], i) => (
        <g key={`sp-${i}`} style={{ pointerEvents: "none" }}>
          <circle cx={x} cy={y} r={10 * s2i} fill="none" stroke="#fff" strokeWidth={2 * s2i} opacity={0.8} />
          <circle cx={x} cy={y} r={4 * s2i} fill="#fff" opacity={0.9} />
          <text x={x + 12 * s2i} y={y + 4 * s2i} fontSize={12 * s2i} fill="#fff">{i + 1}</text>
        </g>
      ))}
    </svg>
  );
}

// local drag position — smooth vertex rendering during drag, never sent to parent
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
