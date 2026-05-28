"use client";

import { useState, useEffect, useRef } from "react";
import { Instance } from "@/lib/api";

interface Props {
  imageSrc: string;
  imgWidth: number;
  imgHeight: number;
  viewportWidth: number;
  viewportHeight: number;
  busy: boolean;
  // Already-committed instances on disk — rendered as semi-transparent green
  // context so the user can see what's been annotated already.
  existingInstances: Instance[];
  // Not-yet-committed proposals — yellow outlines for context. Reject UI
  // lives in the sidebar; clicks here are reserved for drag-to-box.
  pendingProposals: Instance[];
  // Called on mouseup with the finalized box in image coords. The caller is
  // responsible for sending it to /from-boxes and adding the result to
  // pendingProposals.
  onBoxDrawn: (box: [number, number, number, number]) => void;
}

type ViewBox = { x: number; y: number; w: number; h: number };

/**
 * Drag a rectangle around a particle; the box becomes a SAM prompt. Wheel
 * zooms toward the cursor, Space+drag pans. Esc cancels an in-progress drag.
 * Image renders inside the SVG via <image> so the viewBox transforms image
 * and overlay together.
 */
export default function BoxAnnotateCanvas({
  imageSrc,
  imgWidth,
  imgHeight,
  viewportWidth,
  viewportHeight,
  busy,
  existingInstances,
  pendingProposals,
  onBoxDrawn,
}: Props) {
  const [viewBox, setViewBox] = useState<ViewBox>({
    x: 0,
    y: 0,
    w: imgWidth,
    h: imgHeight,
  });
  const [panHeld, setPanHeld] = useState(false);
  const [drag, setDrag] = useState<{
    x0: number;
    y0: number;
    x1: number;
    y1: number;
  } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
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

  useEffect(() => {
    function down(e: KeyboardEvent) {
      if (e.code === "Space" && !e.repeat) {
        e.preventDefault();
        setPanHeld(true);
      } else if (e.key === "Escape") {
        e.preventDefault();
        setDrag(null);
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
  }, []);

  // Non-passive wheel listener so preventDefault actually stops page scroll.
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

  function handleMouseDown(e: React.MouseEvent) {
    if (busy) return;
    if (panHeld) {
      panOriginRef.current = {
        cx: e.clientX,
        cy: e.clientY,
        vx: viewBox.x,
        vy: viewBox.y,
      };
      return;
    }
    const [x, y] = toImageCoords(e.clientX, e.clientY, viewBox);
    setDrag({ x0: x, y0: y, x1: x, y1: y });
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
    if (drag) {
      const [x, y] = toImageCoords(e.clientX, e.clientY, viewBox);
      setDrag(prev => (prev ? { ...prev, x1: x, y1: y } : prev));
    }
  }

  function handleMouseUp() {
    if (panHeld) {
      panOriginRef.current = null;
      return;
    }
    if (!drag) return;
    const x0 = Math.min(drag.x0, drag.x1);
    const y0 = Math.min(drag.y0, drag.y1);
    const x1 = Math.max(drag.x0, drag.x1);
    const y1 = Math.max(drag.y0, drag.y1);
    setDrag(null);
    if (x1 - x0 >= 4 && y1 - y0 >= 4) {
      onBoxDrawn([x0, y0, x1, y1]);
    }
  }

  const zoomScale = viewBox.w / imgWidth;
  const strokeW = 1.5 * zoomScale;

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
        cursor: busy ? "wait" : panHeld ? "grab" : "crosshair",
        background: "rgba(255, 209, 102, 0.04)",
        userSelect: "none",
        pointerEvents: "auto",
        zIndex: 12,
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => {
        panOriginRef.current = null;
        setDrag(null);
      }}
    >
      <image
        href={imageSrc}
        x={0}
        y={0}
        width={imgWidth}
        height={imgHeight}
        preserveAspectRatio="none"
      />
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
      {drag && (
        <rect
          x={Math.min(drag.x0, drag.x1)}
          y={Math.min(drag.y0, drag.y1)}
          width={Math.abs(drag.x1 - drag.x0)}
          height={Math.abs(drag.y1 - drag.y0)}
          fill="rgba(255, 209, 102, 0.15)"
          stroke="#ffd166"
          strokeWidth={strokeW}
          strokeDasharray={`${4 * zoomScale} ${4 * zoomScale}`}
        />
      )}
    </svg>
  );
}
