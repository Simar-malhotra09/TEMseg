"use client";

import { useState, useRef, useEffect } from "react";
import { Stage, Layer, Line, Circle, Text, Image as KonvaImage } from "react-konva";
import Konva from "konva";

import useImage from "use-image";

export interface Scribble {
  id: string;
  points: number[]; // flat [x1, y1, x2, y2, ...] in image coordinates
  strokeWidth: number;
}

// brush diameter bounds, in image-space px. wide range so a small brush
// works on congested images and a large one works on big isolated ones
const MIN_BRUSH_SIZE = 6;
const MAX_BRUSH_SIZE = 400;

interface Props {
  imageSrc: string;
  imgWidth: number;   // original image width
  imgHeight: number;  // original image height
  width: number;      // viewport width
  height: number;     // viewport height
  initialStrokes?: Scribble[];
  brushSize?: number;  // brush diameter, in image-space px
  onBrushSizeChange?: (size: number) => void; // scroll-to-resize callback
  onChange: (strokes: Scribble[]) => void;
}

export default function ScribbleCanvas({
  imageSrc, imgWidth, imgHeight, width, height, initialStrokes, brushSize = 60, onBrushSizeChange, onChange,
}: Props) {
  const [image] = useImage(imageSrc);
  const [strokes, setStrokes] = useState<Scribble[]>(initialStrokes ?? []);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [drawing, setDrawing] = useState<Scribble | null>(null);
  // viewport-space cursor position which drives the live brush-size preview circle
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null);

  // delete selected stroke on backspace
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Backspace" && selectedId) {
        const updated = strokes.filter(s => s.id !== selectedId);
        setStrokes(updated);
        setSelectedId(null);
        onChange(updated);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [selectedId, strokes]);

  // right-click a stroke to delete it directly, no select-then-backspace needed
  const handleDeleteStroke = (id: string) => {
    const updated = strokes.filter(s => s.id !== id);
    setStrokes(updated);
    if (selectedId === id) setSelectedId(null);
    onChange(updated);
  };

  // map viewport pointer -> image coordinates
  const getPointerPos = (e: Konva.KonvaEventObject<MouseEvent>) => {
    const pos = e.target.getStage()?.getPointerPosition();
    if (!pos) return { x: 0, y: 0 };
    const scaleX = imgWidth / width;
    const scaleY = imgHeight / height;
    return { x: pos.x * scaleX, y: pos.y * scaleY };
  };

  const handleMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (e.target === e.target.getStage() || e.target.getClassName() === "Image") {
      setSelectedId(null);
      const pos = getPointerPos(e);
      setDrawing({ id: crypto.randomUUID(), points: [pos.x, pos.y], strokeWidth: brushSize });
    }
  };

  const handleMouseMove = (e: Konva.KonvaEventObject<MouseEvent>) => {
    const stagePos = e.target.getStage()?.getPointerPosition();
    if (stagePos) setCursorPos(stagePos);
    if (!drawing) return;
    const pos = getPointerPos(e);
    setDrawing(d => d ? { ...d, points: [...d.points, pos.x, pos.y] } : null);
  };

  const handleMouseUp = () => {
    if (!drawing) return;
    if (drawing.points.length >= 4) {
      const updated = [...strokes, drawing];
      setStrokes(updated);
      setSelectedId(drawing.id);
      onChange(updated);
    }
    setDrawing(null);
  };

  // scroll to resize brush. keeps the resize control on-canvas instead of
  // spending sidebar space on a slider; stopPropagation so it doesn't also
  // trigger the page-level pan/zoom wheel handler
  const handleWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    e.evt.stopPropagation();
    if (!onBrushSizeChange) return;
    const factor = e.evt.deltaY > 0 ? 0.9 : 1.1;
    const next = Math.round(Math.min(MAX_BRUSH_SIZE, Math.max(MIN_BRUSH_SIZE, brushSize * factor)));
    onBrushSizeChange(next);
  };

  const toViewportPoints = (points: number[]) =>
    points.map((v, i) => i % 2 === 0 ? v * (width / imgWidth) : v * (height / imgHeight));

  return (
    <Stage
      width={width}
      height={height}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => setCursorPos(null)}
      onWheel={handleWheel}
    >
      <Layer>
        <KonvaImage image={image} width={width} height={height} />
        {strokes.map(stroke => (
          <Line
            key={stroke.id}
            points={toViewportPoints(stroke.points)}
            stroke={stroke.id === selectedId ? "#ffd23f" : "#32a2ff"}
            strokeWidth={stroke.strokeWidth * (width / imgWidth)}
            opacity={0.45}
            lineCap="round"
            lineJoin="round"
            hitStrokeWidth={Math.max(24, stroke.strokeWidth * (width / imgWidth))}
            onClick={() => setSelectedId(stroke.id)}
            onContextMenu={e => {
              e.evt.preventDefault();
              handleDeleteStroke(stroke.id);
            }}
          />
        ))}

        {drawing && (
          <Line
            points={toViewportPoints(drawing.points)}
            stroke="#32a2ff"
            strokeWidth={drawing.strokeWidth * (width / imgWidth)}
            opacity={0.45}
            lineCap="round"
            lineJoin="round"
          />
        )}

        {/* live brush-size preview, derived from the cursor, resizes on scroll */}
        {cursorPos && (() => {
          const radius = (brushSize / 2) * (width / imgWidth);
          return (
            <>
              <Circle
                x={cursorPos.x}
                y={cursorPos.y}
                radius={radius}
                stroke="#fff"
                strokeWidth={1}
                dash={[4, 3]}
                opacity={0.8}
                listening={false}
              />
              <Text
                x={cursorPos.x + radius + 6}
                y={cursorPos.y - 6}
                text={`${brushSize}px`}
                fontSize={11}
                fill="#fff"
                listening={false}
              />
            </>
          );
        })()}
      </Layer>
    </Stage>
  );
}
