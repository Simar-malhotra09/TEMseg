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
const VERTEX_RADIUS = 5;
const EDGE_HIT_WIDTH = 8;

export default function RefineCanvas({
  imageSrc,
  width,
  height,
  imgWidth,
  imgHeight,
  instances,
  onChange
}: Props) {

  const [localInstances, setLocalInstances] = useState<Instance[]>(instances);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  const [viewBox, setViewBox] = useState({ x: 0, y: 0, w: imgWidth, h: imgHeight });

  const draggingVertex = useRef<{ instId: number; vi: number } | null>(null);
  const panStart = useRef<{ mx: number; my: number; vx: number; vy: number } | null>(null);
  const isSpaceDown = useRef(false);
  const svgRef = useRef<SVGSVGElement>(null);
  console.log("RefineCanvas props", { width, height, imgWidth, imgHeight });
  useEffect(() => {
    console.log("[SYNC] parent → local instances:", instances.length);
    setLocalInstances(instances);
  }, [instances]);

  useEffect(() => {

    const onKeyDown = (e: KeyboardEvent) => {
      console.log("[KEYDOWN]", e.code, "selected:", selectedId);

      if (e.code === "Space" && !e.repeat) {
        e.preventDefault();
        isSpaceDown.current = true;
        console.log("[PAN MODE] enabled");
        if (svgRef.current) svgRef.current.style.cursor = "grab";
      }

      if ((e.key === "Backspace" || e.key === "Delete") && selectedId !== null) {
        console.log("[DELETE INSTANCE]", selectedId);

        const updated = localInstances.filter(i => i.id !== selectedId);
        setLocalInstances(updated);
        setSelectedId(null);
        onChange(updated);
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      console.log("[KEYUP]", e.code);

      if (e.code === "Space") {
        isSpaceDown.current = false;
        console.log("[PAN MODE] disabled");
        if (svgRef.current) svgRef.current.style.cursor = "default";
      }
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);

    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };

  }, [selectedId, localInstances, onChange]);


  const viewBoxRef = useRef(viewBox);
  useEffect(() => { viewBoxRef.current = viewBox; }, [viewBox]);

  const mouseToImageSpace = useCallback((e: React.MouseEvent): [number, number] => {
    const svg = svgRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const svgPt = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    return [svgPt.x, svgPt.y];
  }, []); 


  const s2i = viewBox.w / width;


  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const svg = svgRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const svgPt = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    const mx = svgPt.x;
    const my = svgPt.y;
    
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


  const handleMouseDown = useCallback((e: React.MouseEvent) => {

    console.log("[MOUSEDOWN]", {
      tag: (e.target as Element).tagName,
      space: isSpaceDown.current
    });

    if (isSpaceDown.current) {

      e.preventDefault();

      panStart.current = {
        mx: e.clientX,
        my: e.clientY,
        vx: viewBox.x,
        vy: viewBox.y
      };

      console.log("[PAN START]", panStart.current);

      if (svgRef.current) svgRef.current.style.cursor = "grabbing";

      return;
    }

    const tag = (e.target as Element).tagName;

    if (tag === "svg" || tag === "image") {
      console.log("[DESELECT]");
      setSelectedId(null);
    }

  }, [viewBox]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (panStart.current) {
      const { vx, vy, mx, my } = panStart.current;
      setViewBox(vb => {
        const dx = -(e.clientX - mx) / width * vb.w;  // use vb.w not viewBox.w
        const dy = -(e.clientY - my) / height * vb.h;
        return {
          ...vb,
          x: Math.min(Math.max(vx + dx, 0), imgWidth - vb.w),
          y: Math.min(Math.max(vy + dy, 0), imgHeight - vb.h),
        };
      });
      return;
    }
    if (draggingVertex.current) {
      const [ix, iy] = mouseToImageSpace(e);
      const { instId, vi } = draggingVertex.current;
      setLocalInstances(prev => {
        const updated = prev.map(inst => inst.id !== instId ? inst : {
          ...inst,
          contour: inst.contour.map((pt, i) => i === vi ? [ix, iy] as [number, number] : pt),
        });
        onChange(updated);
        return updated;
      });
    }
  }, [width, height, imgWidth, imgHeight, mouseToImageSpace, onChange]);




  const handleMouseUp = useCallback(() => {

    console.log("[MOUSE UP]");

    draggingVertex.current = null;
    panStart.current = null;

    if (svgRef.current)
      svgRef.current.style.cursor = isSpaceDown.current ? "grab" : "default";

  }, []);


  const handleEdgeClick = useCallback((e: React.MouseEvent, instId: number) => {

    e.stopPropagation();

    console.log("[EDGE CLICK]", instId);

    const [mx, my] = mouseToImageSpace(e);

    console.log("[EDGE CLICK POS]", mx, my);

    const inst = localInstances.find(i => i.id === instId);

    if (!inst) return;

    const n = inst.contour.length;

    let bestIdx = 0;
    let bestDist = Infinity;

    for (let i = 0; i < n; i++) {

      const [ax, ay] = inst.contour[i];
      const [bx, by] = inst.contour[(i + 1) % n];

      const dx = bx - ax;
      const dy = by - ay;

      const lenSq = dx * dx + dy * dy;

      const t =
        lenSq > 0
          ? Math.max(
              0,
              Math.min(1, ((mx - ax) * dx + (my - ay) * dy) / lenSq)
            )
          : 0;

      const dist = Math.sqrt(
        (ax + t * dx - mx) ** 2 + (ay + t * dy - my) ** 2
      );

      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }

    const [ax, ay] = inst.contour[bestIdx];
    const [bx, by] = inst.contour[(bestIdx + 1) % n];

    const newContour = [
      ...inst.contour.slice(0, bestIdx + 1),
      [(ax + bx) / 2, (ay + by) / 2] as [number, number],
      ...inst.contour.slice(bestIdx + 1),
    ];

    console.log("[VERTEX INSERTED]", instId, "index", bestIdx + 1);

    const updated = localInstances.map(i =>
      i.id === instId ? { ...i, contour: newContour } : i
    );

    setLocalInstances(updated);
    onChange(updated);

  }, [mouseToImageSpace, localInstances, onChange]);


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
      onClick={(e)=>console.log("[SVG CLICK TARGET]", e.target)}
    >
      <image
        href={imageSrc}
        x={0}
        y={0}
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

            <polygon
              points={pts}
              fill={`${color}33`}
              stroke={color}
              strokeWidth={isSelected ? 2.5 * s2i : 1.5 * s2i}
              opacity={isSelected ? 1 : 0.7}
              style={{ cursor: "pointer" }}
              onClick={e => {
                e.stopPropagation();
                console.log("[SELECT INSTANCE]", inst.id);
                setSelectedId(inst.id);
              }}
            />

            {isSelected && (
              <>

                <polygon
                  points={pts}
                  fill="none"
                  stroke="transparent"
                  strokeWidth={EDGE_HIT_WIDTH * 2 * s2i}
                  style={{ cursor: "cell" }}
                  onClick={e => handleEdgeClick(e, inst.id)}
                />

                {inst.contour.map(([x, y], vi) => (

                  <circle
                    key={`v-${inst.id}-${vi}`}
                    cx={x}
                    cy={y}
                    r={VERTEX_RADIUS * s2i}
                    fill={color}
                    stroke="#fff"
                    strokeWidth={1.5 * s2i}
                    style={{ cursor: "move" }}
                    onMouseDown={e => {
                      e.stopPropagation();
                      console.log("[VERTEX DRAG START]", inst.id, vi);
                      draggingVertex.current = { instId: inst.id, vi };
                    }}
                    onDoubleClick={e => {

                      e.stopPropagation();

                      console.log("[VERTEX DELETE]", inst.id, vi);

                      if (inst.contour.length <= 3) return;

                      const updated = localInstances.map(ins =>
                        ins.id !== inst.id
                          ? ins
                          : {
                              ...ins,
                              contour: ins.contour.filter(
                                (_, idx) => idx !== vi
                              ),
                            }
                      );

                      setLocalInstances(updated);
                      onChange(updated);

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
