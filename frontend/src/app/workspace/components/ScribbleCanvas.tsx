"use client";

import { useState, useRef, useEffect } from "react";
import { Stage, Layer, Line, Image as KonvaImage } from "react-konva";
import Konva from "konva";

import useImage from "use-image";

export interface Scribble {
  id: string;
  points: number[]; // flat [x1, y1, x2, y2, ...] in image coordinates
  strokeWidth: number;
}

interface Props {
  imageSrc: string;
  imgWidth: number;   // original image width
  imgHeight: number;  // original image height
  width: number;      // viewport width
  height: number;     // viewport height
  initialStrokes?: Scribble[];
  brushSize?: number;  // brush diameter, in image-space px
  onChange: (strokes: Scribble[]) => void;
}

export default function ScribbleCanvas({
  imageSrc, imgWidth, imgHeight, width, height, initialStrokes, brushSize = 60, onChange,
}: Props) {
  const [image] = useImage(imageSrc);
  const [strokes, setStrokes] = useState<Scribble[]>(initialStrokes ?? []);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [drawing, setDrawing] = useState<Scribble | null>(null);

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

  const toViewportPoints = (points: number[]) =>
    points.map((v, i) => i % 2 === 0 ? v * (width / imgWidth) : v * (height / imgHeight));

  return (
    <Stage
      width={width}
      height={height}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
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
      </Layer>
    </Stage>
  );
}
