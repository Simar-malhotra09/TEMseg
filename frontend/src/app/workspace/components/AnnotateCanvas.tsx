"use client";

import { useState, useEffect, useRef } from "react";
import { Instance } from "@/lib/api";

interface Props {
  imageSrc: string;
  imgWidth: number;
  imgHeight: number;
  viewportWidth: number;
  viewportHeight: number;
  // Already-committed instances on disk  
  existingInstances: Instance[];
  // Not-yet-committed proposals 
  pendingProposals: Instance[];
  // Called when the user closes a polygon with ≥3 vertices. The contour is
  // in image-space coordinates. Caller is responsible for turning it into a
  // pending proposal.
  onPolygonComplete: (contour: [number, number][]) => void;
}

type ViewBox = { x: number; y: number; w: number; h: number };

/**
 * Click to place a vertex, double-click or Enter to close, Backspace to undo
 * the last vertex, Esc to cancel the in-progress polygon. Hold Space to pan,
 * wheel to zoom (cursor stays under the same image pixel). Closed polygons
 * leave the canvas via onPolygonComplete; in-progress vertices live in
 * component state.
 *
 * The SVG viewBox is in image-space and shrinks/translates on zoom/pan, so
 * downstream logic always sees real image coordinates regardless of how the
 * user navigated to them.
 */
export default function AnnotateCanvas({
  imageSrc,
  imgWidth,
  imgHeight,
  viewportWidth,
  viewportHeight,
  existingInstances,
  pendingProposals,
  onPolygonComplete,
}: Props) {
  const [vertices, setVertices] = useState<[number, number][]>([]);
  const [cursor, setCursor] = useState<[number, number] | null>(null);
  const [viewBox, setViewBox] = useState<ViewBox>({
    x: 0,
    y: 0,
    w: imgWidth,
    h: imgHeight,
  });
  const [panHeld, setPanHeld] = useState(false); // space currently down
  const svgRef = useRef<SVGSVGElement>(null);
  // pan-drag origin: cursor client pos + viewBox top-left at drag start
  const panOriginRef = useRef<{
    cx: number;
    cy: number;
    vx: number;
    vy: number;
  } | null>(null);

  function toImageCoords(
    clientX: number,
    clientY: number,
    vb: ViewBox,
  ): [number, number] {
    if (!svgRef.current) return [0, 0];
    const rect = svgRef.current.getBoundingClientRect();
    const x = vb.x + ((clientX - rect.left) / rect.width) * vb.w;
    const y = vb.y + ((clientY - rect.top) / rect.height) * vb.h;
    return [Math.round(x), Math.round(y)];
  }

  function clampViewBox(vb: ViewBox): ViewBox {
    const w = Math.min(vb.w, imgWidth);
    const h = Math.min(vb.h, imgHeight);
    const x = Math.max(0, Math.min(imgWidth - w, vb.x));
    const y = Math.max(0, Math.min(imgHeight - h, vb.y));
    return { x, y, w, h };
  }

  // Spacebar enters pan mode globally (matches refine/Figma convention).
  // preventDefault on space so the browser doesn't scroll the page.
  useEffect(() => {
    function down(e: KeyboardEvent) {
      if (e.code === "Space" && !e.repeat) {
        e.preventDefault();
        setPanHeld(true);
      } else if (e.key === "Escape") {
        e.preventDefault();
        setVertices([]);
        setCursor(null);
      } else if (e.key === "Backspace") {
        e.preventDefault();
        setVertices(prev => prev.slice(0, -1));
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (vertices.length >= 3) {
          onPolygonComplete(vertices);
          setVertices([]);
          setCursor(null);
        }
      }
    }
    function up(e: KeyboardEvent) {
      if (e.code === "Space") {
        e.preventDefault();
        setPanHeld(false);
        panOriginRef.current = null;
      }
    }
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, [vertices, onPolygonComplete]);

  function handleClick(e: React.MouseEvent) {
    // Suppress vertex placement during pan-drag (mouseup → click fires
    // after a drag, which would otherwise drop an unwanted vertex).
    if (panHeld) return;
    setVertices(prev => [
      ...prev,
      toImageCoords(e.clientX, e.clientY, viewBox),
    ]);
  }

  // Two clicks fire before dblclick 
  function handleDoubleClick(e: React.MouseEvent) {
    if (panHeld) return;
    e.preventDefault();
    setVertices(prev => {
      const trimmed = prev.slice(0, -2);
      if (trimmed.length >= 3) {
        onPolygonComplete(trimmed);
        setCursor(null);
        return [];
      }
      return trimmed;
    });
  }

  function handleMouseDown(e: React.MouseEvent) {
    if (panHeld) {
      panOriginRef.current = {
        cx: e.clientX,
        cy: e.clientY,
        vx: viewBox.x,
        vy: viewBox.y,
      };
    }
  }

  function handleMouseMove(e: React.MouseEvent) {
    if (panHeld && panOriginRef.current && svgRef.current) {
      const rect = svgRef.current.getBoundingClientRect();
      const scaleX = viewBox.w / rect.width;
      const scaleY = viewBox.h / rect.height;
      const dx = (e.clientX - panOriginRef.current.cx) * scaleX;
      const dy = (e.clientY - panOriginRef.current.cy) * scaleY;
      setViewBox(prev =>
        clampViewBox({
          ...prev,
          x: panOriginRef.current!.vx - dx,
          y: panOriginRef.current!.vy - dy,
        }),
      );
      return;
    }
    setCursor(toImageCoords(e.clientX, e.clientY, viewBox));
  }

  function handleMouseUp() {
    panOriginRef.current = null;
  }

  // Wheel zoom 
  // React attaches wheel as passive by default; preventDefault wouldn't
  // actually stop page scroll, so we wire a non-passive native listener.
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    function onWheel(e: WheelEvent) {
      e.preventDefault();
      const rect = svg!.getBoundingClientRect();
      const fx = (e.clientX - rect.left) / rect.width;
      const fy = (e.clientY - rect.top) / rect.height;
      const factor = e.deltaY > 0 ? 1.15 : 0.87;
      setViewBox(prev => {
        const newW = Math.max(20, prev.w * factor);
        const newH = Math.max(20, prev.h * factor);
        const cx = prev.x + fx * prev.w;
        const cy = prev.y + fy * prev.h;
        return clampViewBox({
          x: cx - fx * newW,
          y: cy - fy * newH,
          w: newW,
          h: newH,
        });
      });
    }
    svg.addEventListener("wheel", onWheel, { passive: false });
    return () => svg.removeEventListener("wheel", onWheel);
  }, [imgWidth, imgHeight]);

  const verticesStr = vertices.map(([x, y]) => `${x},${y}`).join(" ");
  const lastVertex = vertices.length > 0 ? vertices[vertices.length - 1] : null;

  // Vertex marker radius / stroke widths scale with zoom so they stay visually
  // consistent at any zoom level (since viewBox shrinks but render size stays).
  const zoomScale = viewBox.w / imgWidth;
  const vertexR = 5 * zoomScale;
  const strokeW =  5 * zoomScale;

  return (
    <svg
      ref={svgRef}
      viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`}
      preserveAspectRatio="xMidYMid meet"
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: viewportWidth,
        height: viewportHeight,
        cursor: panHeld ? "grab" : "crosshair",
        // subtle tint so the user knows the mode is active
        background: "rgba(126, 232, 162, 0.04)",
        userSelect: "none",
        pointerEvents: "auto",
        zIndex: 12,
      }}
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => {
        setCursor(null);
        panOriginRef.current = null;
      }}
    >
      {/* Image lives inside the SVG so the viewBox transforms image + overlay
          together. Parent must hide its own <img> while this canvas is active. */}
      <image
        href={imageSrc}
        x={0}
        y={0}
        width={imgWidth}
        height={imgHeight}
        preserveAspectRatio="none"
      />
      {/* committed instances  */}
      {existingInstances.map(inst => {
        if (!inst.contour || inst.contour.length < 3) return null;
        const pts = inst.contour.map(([x, y]) => `${x},${y}`).join(" ");
        return (
          <polygon
            key={`existing-${inst.id}`}
            points={pts}
            fill="rgba(126, 232, 162, 0.10)"
            stroke="#7ee8a2"
            strokeWidth={strokeW * 0.7}
            pointerEvents="none"
          />
        );
      })}
      {/* pending proposals  */}
      {pendingProposals.map(p => {
        if (!p.contour || p.contour.length < 3) return null;
        const pts = p.contour.map(([x, y]) => `${x},${y}`).join(" ");
        return (
          <polygon
            key={`pending-${p.id}`}
            points={pts}
            fill="rgba(255, 209, 102, 0.15)"
            stroke="#ffd166"
            strokeWidth={strokeW}
            pointerEvents="none"
          />
        );
      })}
      {vertices.length >= 2 && (
        <polyline
          points={verticesStr}
          fill="none"
          stroke="#00f0ff"
          strokeWidth={strokeW}
        />
      )}
      {lastVertex && cursor && !panHeld && (
        <line
          x1={lastVertex[0]}
          y1={lastVertex[1]}
          x2={cursor[0]}
          y2={cursor[1]}
          stroke="#00f0ff"
          strokeWidth={strokeW}
          strokeDasharray={`${4 * zoomScale} ${4 * zoomScale}`}
        />
      )}
      {vertices.map(([x, y], i) => (
        <circle
          key={i}
          cx={x}
          cy={y}
          r={vertexR}
          fill="#00f0ff"
          stroke="#0d0d0d"
          strokeWidth={strokeW * 0.7}
        />
      ))}
    </svg>
  );
}
