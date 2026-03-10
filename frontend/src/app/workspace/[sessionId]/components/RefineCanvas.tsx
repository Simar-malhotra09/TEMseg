"use client";

import { useRef, useState, useEffect, useCallback } from "react";

interface Instance {
  id: number;
  contour: [number, number][];
  bbox: { x: number; y: number; w: number; h: number };
  area: number;
}

interface Props {
  imageSrc: string;
  width: number;
  height: number;
  imgWidth: number;
  imgHeight: number;
  instances: Instance[];
  onChange: (instances: Instance[]) => void;
}

const COLORS = [
  "#ff4444", "#44ff44", "#4488ff", "#ffff44",
  "#ff44ff", "#44ffff", "#ff8844", "#8844ff",
  "#44ff88", "#ff4488",
];

const MIN_ZOOM = 1;
const MAX_ZOOM = 20;
const VERTEX_RADIUS = 5;   // screen-space px, scaled by s2i for SVG
const EDGE_HIT_WIDTH = 8;  // screen-space px for invisible edge hit area

export default function RefineCanvas({
  imageSrc, width, height, imgWidth, imgHeight, instances, onChange
}: Props) {

  // localInstances is the source of truth while in refine mode.
  // it is initialized from props once, and never re-synced from parent
  // to avoid overwriting in-progress edits.
  // onChange notifies parent after every committed edit.
  const [localInstances, setLocalInstances] = useState<Instance[]>(instances);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [viewBox, setViewBox] = useState({ x: 0, y: 0, w: imgWidth, h: imgHeight });

  // all mutable interaction state lives in refs to avoid stale closures
  const instancesRef = useRef<Instance[]>(instances);  // always current, read in handlers
  const draggingVertex = useRef<{ instId: number; vi: number } | null>(null);
  const panStart = useRef<{ mx: number; my: number; vx: number; vy: number } | null>(null);
  const isSpaceDown = useRef(false);
  const svgRef = useRef<SVGSVGElement>(null);

  // keep instancesRef in sync with state 
  useEffect(() => {
    instancesRef.current = localInstances;
  }, [localInstances]);

  // helper: commit a new instances array to both local state and parent
  const commit = useCallback((updated: Instance[]) => {
    instancesRef.current = updated;  // update ref immediately, before re-render
    setLocalInstances(updated);
    onChange(updated);
  }, [onChange]);

  // keyboard: space = pan mode toggle, backspace/delete = remove selected instance
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space" && !e.repeat) {
        e.preventDefault();
        isSpaceDown.current = true;
        if (svgRef.current) svgRef.current.style.cursor = "grab";
      }
      if ((e.key === "Backspace" || e.key === "Delete") && selectedId !== null) {
        const updated = instancesRef.current.filter(i => i.id !== selectedId);
        setSelectedId(null);
        commit(updated);
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        isSpaceDown.current = false;
        if (svgRef.current) svgRef.current.style.cursor = "default";
      }
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  // only selectedId matters here — instancesRef is always current via ref
  }, [selectedId, commit]);

  // convert screen mouse position to SVG/image-space coordinates
  // uses getScreenCTM so it is immune to CSS transforms on parent elements
  const mouseToImageSpace = useCallback((e: React.MouseEvent): [number, number] => {
    const svg = svgRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const svgPt = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    return [svgPt.x, svgPt.y];
  }, []);

  // scale factor: image pixels per screen pixel at current zoom
  // used to keep vertex radius and stroke width visually constant
  const s2i = viewBox.w / width;

  // wheel zoom: zooms toward cursor position, clamped to MIN/MAX_ZOOM
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const svg = svgRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const { x: mx, y: my } = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    const factor = e.deltaY > 0 ? 1.1 : 0.9;
    setViewBox(vb => {
      const newW = Math.min(Math.max(vb.w * factor, imgWidth / MAX_ZOOM), imgWidth / MIN_ZOOM);
      const newH = Math.min(Math.max(vb.h * factor, imgHeight / MAX_ZOOM), imgHeight / MIN_ZOOM);
      return {
        w: newW, h: newH,
        x: Math.min(Math.max(mx - (mx - vb.x) * (newW / vb.w), 0), imgWidth - newW),
        y: Math.min(Math.max(my - (my - vb.y) * (newH / vb.h), 0), imgHeight - newH),
      };
    });
  }, [imgWidth, imgHeight]);

  // mousedown: start pan (space held) or deselect (click background)
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (isSpaceDown.current) {
      e.preventDefault();
      panStart.current = { mx: e.clientX, my: e.clientY, vx: viewBox.x, vy: viewBox.y };
      if (svgRef.current) svgRef.current.style.cursor = "grabbing";
      return;
    }
    const tag = (e.target as Element).tagName;
    if (tag === "svg" || tag === "image") setSelectedId(null);
  }, [viewBox]);

  // mousemove: pan (if panStart set) or drag vertex (if draggingVertex set)
  // reads instancesRef directly — never closes over localInstances state
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (panStart.current) {
      const { vx, vy, mx, my } = panStart.current;
      setViewBox(vb => ({
        ...vb,
        x: Math.min(Math.max(vx - (e.clientX - mx) / width * vb.w, 0), imgWidth - vb.w),
        y: Math.min(Math.max(vy - (e.clientY - my) / height * vb.h, 0), imgHeight - vb.h),
      }));
      return;
    }
    if (draggingVertex.current) {
      const [ix, iy] = mouseToImageSpace(e);
      const { instId, vi } = draggingVertex.current;
      // read from ref — always current even after previous drags
      const updated = instancesRef.current.map(inst => inst.id !== instId ? inst : {
        ...inst,
        contour: inst.contour.map((pt, i) => i === vi ? [ix, iy] as [number, number] : pt),
      });
      // update ref immediately so next mousemove sees the new positions
      instancesRef.current = updated;
      setLocalInstances(updated);
      // note: we do NOT call onChange during drag — only on mouseup
    }
  }, [width, height, imgWidth, imgHeight, mouseToImageSpace]);

  // mouseup: finalize drag and notify parent
  const handleMouseUp = useCallback(() => {
    if (draggingVertex.current) {
      // commit final position to parent only on release
      onChange(instancesRef.current);
    }
    draggingVertex.current = null;
    panStart.current = null;
    if (svgRef.current) svgRef.current.style.cursor = isSpaceDown.current ? "grab" : "default";
  }, [onChange]);

  // edge click: insert a new vertex at the midpoint of the closest edge
  const handleEdgeClick = useCallback((e: React.MouseEvent, instId: number) => {
    e.stopPropagation();
    const [mx, my] = mouseToImageSpace(e);
    const inst = instancesRef.current.find(i => i.id === instId);
    if (!inst) return;

    const n = inst.contour.length;
    let bestIdx = 0, bestDist = Infinity;

    for (let i = 0; i < n; i++) {
      const [ax, ay] = inst.contour[i];
      const [bx, by] = inst.contour[(i + 1) % n];
      const dx = bx - ax, dy = by - ay;
      const lenSq = dx * dx + dy * dy;
      const t = lenSq > 0 ? Math.max(0, Math.min(1, ((mx - ax) * dx + (my - ay) * dy) / lenSq)) : 0;
      const dist = Math.sqrt((ax + t * dx - mx) ** 2 + (ay + t * dy - my) ** 2);
      if (dist < bestDist) { bestDist = dist; bestIdx = i; }
    }

    const [ax, ay] = inst.contour[bestIdx];
    const [bx, by] = inst.contour[(bestIdx + 1) % n];
    const newContour = [
      ...inst.contour.slice(0, bestIdx + 1),
      [(ax + bx) / 2, (ay + by) / 2] as [number, number],
      ...inst.contour.slice(bestIdx + 1),
    ];
    commit(instancesRef.current.map(i => i.id === instId ? { ...i, contour: newContour } : i));
  }, [mouseToImageSpace, commit]);

  // vertex double-click: delete vertex (minimum 3 vertices enforced)
  const handleVertexDelete = useCallback((instId: number, vi: number) => {
    const inst = instancesRef.current.find(i => i.id === instId);
    if (!inst || inst.contour.length <= 3) return;
    commit(instancesRef.current.map(i => i.id !== instId ? i : {
      ...i,
      contour: i.contour.filter((_, idx) => idx !== vi),
    }));
  }, [commit]);

  const pointsStr = (contour: [number, number][]) =>
    contour.map(([x, y]) => `${x},${y}`).join(" ");

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`}
      style={{ display: "block", cursor: "default", userSelect: "none" }}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* image rendered at full natural resolution — viewBox handles zoom/pan */}
      <image
        href={imageSrc}
        x={0} y={0}
        width={imgWidth}
        height={imgHeight}
        style={{ imageRendering: "crisp-edges" }}
      />

      {localInstances.map((inst, i) => {
        const color = COLORS[i % COLORS.length];
        const isSelected = inst.id === selectedId;
        const pts = pointsStr(inst.contour);

        return (
          <g key={inst.id}>

            {/* visible polygon fill and outline */}
            <polygon
              points={pts}
              fill={`${color}33`}
              stroke={color}
              strokeWidth={isSelected ? 2.5 * s2i : 1.5 * s2i}
              opacity={isSelected ? 1 : 0.7}
              style={{ cursor: "pointer" }}
              onClick={e => { e.stopPropagation(); setSelectedId(inst.id); }}
            />

            {isSelected && (
              <>
                {/* wider invisible stroke over edges — clicking inserts a vertex */}
                <polygon
                  points={pts}
                  fill="none"
                  stroke="transparent"
                  strokeWidth={EDGE_HIT_WIDTH * 2 * s2i}
                  style={{ cursor: "cell" }}
                  onClick={e => handleEdgeClick(e, inst.id)}
                />

                {/* vertex handles — drag to move, double-click to delete */}
                {inst.contour.map(([x, y], vi) => (
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
                      draggingVertex.current = { instId: inst.id, vi };
                    }}
                    onDoubleClick={e => {
                      e.stopPropagation();
                      handleVertexDelete(inst.id, vi);
                    }}
                  />
                ))}
              </>
            )}
          </g>
        );
      })}
    </svg>
  );
}
