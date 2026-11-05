"use client";

import { useEffect, useRef, useState } from "react";
import { Instance } from "@/lib/api";

interface Props {
  imageSrc: string;
  imageFilter: string;
  imgWidth: number;
  imgHeight: number;
  viewportWidth: number;
  viewportHeight: number;
  busy: boolean;
  existingInstances: Instance[];
  pendingProposals: Instance[];
  onPointClick: (point: [number, number]) => void;
  onRejectProposal: (id: number) => void;
  zoomCommand: CanvasZoomCommand;
  onZoomChange: (zoom: number) => void;
}

type ViewBox = { x: number; y: number; w: number; h: number };
export type CanvasZoomCommand = { revision: number; action: "fit" | "in" | "out" };

export default function PointAnnotateCanvas({
  imageSrc,
  imageFilter,
  imgWidth,
  imgHeight,
  viewportWidth,
  viewportHeight,
  busy,
  existingInstances,
  pendingProposals,
  onPointClick,
  onRejectProposal,
  zoomCommand,
  onZoomChange,
}: Props) {
  const [viewBox, setViewBox] = useState<ViewBox>({ x: 0, y: 0, w: imgWidth, h: imgHeight });
  const [panHeld, setPanHeld] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);
  const panOriginRef = useRef<{ cx: number; cy: number; vx: number; vy: number } | null>(null);
  const didPanRef = useRef(false);
  const zoomRevisionRef = useRef(zoomCommand.revision);

  useEffect(() => {
    onZoomChange(viewBox.w / imgWidth);
  }, [viewBox.w, imgWidth, onZoomChange]);

  function clampViewBox(next: ViewBox): ViewBox {
    const w = Math.min(next.w, imgWidth);
    const h = Math.min(next.h, imgHeight);
    return {
      x: Math.max(0, Math.min(imgWidth - w, next.x)),
      y: Math.max(0, Math.min(imgHeight - h, next.y)),
      w,
      h,
    };
  }

  function toImageCoords(clientX: number, clientY: number): [number, number] {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return [0, 0];
    return [
      Math.round(viewBox.x + ((clientX - rect.left) / rect.width) * viewBox.w),
      Math.round(viewBox.y + ((clientY - rect.top) / rect.height) * viewBox.h),
    ];
  }

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.code === "Space" && !event.repeat) {
        event.preventDefault();
        setPanHeld(true);
      }
    }
    function handleKeyUp(event: KeyboardEvent) {
      if (event.code === "Space") {
        event.preventDefault();
        setPanHeld(false);
        panOriginRef.current = null;
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    function handleWheel(event: WheelEvent) {
      event.preventDefault();
      event.stopPropagation();
      const rect = svg!.getBoundingClientRect();
      const fx = (event.clientX - rect.left) / rect.width;
      const fy = (event.clientY - rect.top) / rect.height;
      const factor = event.deltaY > 0 ? 1.15 : 0.87;
      setViewBox((current) => {
        const w = Math.max(20, current.w * factor);
        const h = Math.max(20, current.h * factor);
        const cx = current.x + fx * current.w;
        const cy = current.y + fy * current.h;
        return clampViewBox({ x: cx - fx * w, y: cy - fy * h, w, h });
      });
    }
    svg.addEventListener("wheel", handleWheel, { passive: false });
    return () => svg.removeEventListener("wheel", handleWheel);
  }, [imgWidth, imgHeight]);

  useEffect(() => {
    if (zoomCommand.revision === zoomRevisionRef.current) return;
    zoomRevisionRef.current = zoomCommand.revision;
    setViewBox((current) => {
      if (zoomCommand.action === "fit") {
        return { x: 0, y: 0, w: imgWidth, h: imgHeight };
      }
      const factor = zoomCommand.action === "in" ? 0.9 : 1.1;
      const w = Math.max(20, current.w * factor);
      const h = Math.max(20, current.h * factor);
      const cx = current.x + current.w / 2;
      const cy = current.y + current.h / 2;
      return clampViewBox({ x: cx - w / 2, y: cy - h / 2, w, h });
    });
  }, [zoomCommand, imgWidth, imgHeight, onZoomChange]);

  function handleMouseDown(event: React.MouseEvent<SVGSVGElement>) {
    if (!panHeld) return;
    event.preventDefault();
    didPanRef.current = false;
    panOriginRef.current = {
      cx: event.clientX,
      cy: event.clientY,
      vx: viewBox.x,
      vy: viewBox.y,
    };
  }

  function handleMouseMove(event: React.MouseEvent<SVGSVGElement>) {
    const origin = panOriginRef.current;
    const rect = svgRef.current?.getBoundingClientRect();
    if (!panHeld || !origin || !rect) return;
    const dx = (event.clientX - origin.cx) * (viewBox.w / rect.width);
    const dy = (event.clientY - origin.cy) * (viewBox.h / rect.height);
    if (Math.abs(dx) + Math.abs(dy) > 0) didPanRef.current = true;
    setViewBox((current) => clampViewBox({
      ...current,
      x: origin.vx - dx,
      y: origin.vy - dy,
    }));
  }

  function handleClick(event: React.MouseEvent<SVGSVGElement>) {
    if (busy || panHeld || didPanRef.current) {
      didPanRef.current = false;
      return;
    }
    onPointClick(toImageCoords(event.clientX, event.clientY));
  }

  const zoomScale = viewBox.w / imgWidth;
  const strokeWidth = 1.5 * zoomScale;
  const existingStrokeWidth = 4.2 * zoomScale;

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
        userSelect: "none",
        pointerEvents: "auto",
        zIndex: 12,
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={() => { panOriginRef.current = null; }}
      onMouseLeave={() => { panOriginRef.current = null; }}
      onClick={handleClick}
    >
      <image href={imageSrc} x={0} y={0} width={imgWidth} height={imgHeight} preserveAspectRatio="none" style={{ filter: imageFilter }} />
      <rect x={0} y={0} width={imgWidth} height={imgHeight} fill="none" stroke="rgba(126, 232, 162, 0.5)" strokeWidth={2 * zoomScale} pointerEvents="none" />
      {existingInstances.map((instance) => instance.contour && instance.contour.length >= 3 ? (
        <g key={`existing-${instance.id}`} pointerEvents="none">
          <polygon points={instance.contour.map(([x, y]) => `${x},${y}`).join(" ")} fill="rgba(126, 232, 162, 0.10)" stroke="#7ee8a2" strokeWidth={existingStrokeWidth} />
          {(instance.extra_contours ?? []).map((contour, index) => (
            <polygon key={`extra-${index}`} points={contour.map(([x, y]) => `${x},${y}`).join(" ")} fill="rgba(126, 232, 162, 0.10)" stroke="#7ee8a2" strokeWidth={existingStrokeWidth} />
          ))}
        </g>
      ) : null)}
      {pendingProposals.map((proposal) => proposal.contour && proposal.contour.length >= 3 ? (
        <polygon
          key={`pending-${proposal.id}`}
          points={proposal.contour.map(([x, y]) => `${x},${y}`).join(" ")}
          fill="rgba(255, 209, 102, 0.15)"
          stroke="#ffd166"
          strokeWidth={strokeWidth}
          style={{ cursor: "pointer" }}
          onClick={(event) => { event.stopPropagation(); onRejectProposal(proposal.id); }}
        >
          <title>Click to reject proposal #{proposal.id} (area {proposal.area}px)</title>
        </polygon>
      ) : null)}
    </svg>
  );
}
