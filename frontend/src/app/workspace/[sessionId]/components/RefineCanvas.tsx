"use client";

import { useRef, useState, useEffect } from "react";
import { Stage, Layer, Image as KonvaImage, Line, Circle } from "react-konva";
import useImage from "use-image";
import { Fragment } from "react";
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

// distinct colors per instance
const COLORS = [
  "#ff4444", "#44ff44", "#4488ff", "#ffff44",
  "#ff44ff", "#44ffff", "#ff8844", "#8844ff",
  "#44ff88", "#ff4488",
];

export default function RefineCanvas({
  imageSrc, width, height, imgWidth, imgHeight, instances, onChange
}: Props) {
  const [image] = useImage(imageSrc);
  const [localInstances, setLocalInstances] = useState<Instance[]>(instances);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const layerRef = useRef<any>(null);

  // sync if parent instances change (e.g. on first load)
  useEffect(() => {
    setLocalInstances(instances);
  }, [instances]);

  // delete selected instance on backspace
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.key === "Backspace" || e.key === "Delete") && selectedId !== null) {
        const updated = localInstances.filter(i => i.id !== selectedId);
        setLocalInstances(updated);
        setSelectedId(null);
        onChange(updated);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [selectedId, localInstances]);

  // ── coordinate scaling ───────────────────────────────────────
  // contours stored in image-space, displayed in viewport-space
  function toViewport(x: number, y: number): [number, number] {
    return [
      (x / imgWidth) * width,
      (y / imgHeight) * height,
    ];
  }

  function toImageSpace(x: number, y: number): [number, number] {
    return [
      (x / width) * imgWidth,
      (y / height) * imgHeight,
    ];
  }

  function getScaledPoints(contour: [number, number][]): number[] {
    return contour.flatMap(([x, y]) => toViewport(x, y));
  }

  // ── vertex drag ──────────────────────────────────────────────
  function handleVertexDrag(instId: number, vertexIdx: number, newX: number, newY: number) {
    const [ix, iy] = toImageSpace(newX, newY);
    const updated = localInstances.map(inst => {
      if (inst.id !== instId) return inst;
      const newContour = inst.contour.map((pt, i) =>
        i === vertexIdx ? [ix, iy] as [number, number] : pt
      );
      return { ...inst, contour: newContour };
    });
    setLocalInstances(updated);
    onChange(updated);
  }

  return (
    <Stage
      width={width}
      height={height}
      onMouseDown={e => {
        // deselect when clicking empty area or image
        if (e.target === e.target.getStage() || e.target.getClassName() === "Image") {
          setSelectedId(null);
        }
      }}
    >
      <Layer ref={layerRef}>
        {/* base image */}
        <KonvaImage image={image} width={width} height={height} />

        {localInstances.map((inst, i) => {
          const color = COLORS[i % COLORS.length];
          const isSelected = inst.id === selectedId;
          const scaledPoints = getScaledPoints(inst.contour);

          return (
            <Fragment key={inst.id}>
              <Line
                points={scaledPoints}
                closed
                fill={`${color}33`}
                stroke={color}
                strokeWidth={isSelected ? 2.5 : 1.5}
                opacity={isSelected ? 1 : 0.7}
                onClick={() => setSelectedId(inst.id)}
                onTap={() => setSelectedId(inst.id)}
              />
              {isSelected && inst.contour.map(([x, y], vi) => {
                const [vx, vy] = toViewport(x, y);
                return (
                  <Circle
                    key={`v-${inst.id}-${vi}`}
                    x={vx}
                    y={vy}
                    radius={5}
                    fill={color}
                    stroke="#fff"
                    strokeWidth={1.5}
                    draggable
                    onDragMove={e =>
                      handleVertexDrag(inst.id, vi, e.target.x(), e.target.y())
                    }
                  />
                );
              })}
            </Fragment>
          );
        })}
      </Layer>
    </Stage>
  );
}
