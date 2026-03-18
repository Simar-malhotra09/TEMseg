"use client";

import { useState, useRef, useEffect } from "react";
import { Stage, Layer, Rect, Transformer, Image as KonvaImage } from "react-konva";
import Konva from "konva";

import useImage from "use-image";

export interface BlackoutRect {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

interface Props {
  imageSrc: string;
  imgWidth: number;   // original image width
  imgHeight: number;  // original image height
  width: number;      // viewport width
  height: number;     // viewport height
  initialRegions?: BlackoutRect[]; // pre render regions if any stored in state
  isInverse: boolean; // true if inverse blackout 
  onChange: (regions: BlackoutRect[]) => void;
}


export default function BlackoutCanvas({ imageSrc, imgWidth, imgHeight, width, height, isInverse, initialRegions, onChange }: Props) {
  const [image] = useImage(imageSrc);
  const [rects, setRects] = useState<BlackoutRect[]>(initialRegions ?? []);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [drawing, setDrawing] = useState<BlackoutRect | null>(null);

  const transformerRef = useRef<Konva.Transformer | null>(null);
  const stageRef = useRef<Konva.Stage | null>(null);
  const layerRef = useRef<Konva.Layer | null>(null);

  // attach transformer to selected rect
  useEffect(() => {
    if (!transformerRef.current || !layerRef.current) return;
    if (selectedId) {
      const node = layerRef.current.findOne(`#${selectedId}`);
      if (node) transformerRef.current.nodes([node]);
    } else {
      transformerRef.current.nodes([]);
    }
    transformerRef.current.getLayer()?.batchDraw();
  }, [selectedId]);

  // delete on backspace
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Backspace" && selectedId) {
        const updated = rects.filter(r => r.id !== selectedId);
        setRects(updated);
        setSelectedId(null);
        onChange(updated);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [selectedId, rects]);

  // map viewport pointer -> image coordinates
  const getPointerPos = (e: Konva.KonvaEventObject<MouseEvent>) => {
    const pos = e.target.getStage()?.getPointerPosition();
    if (!pos) return { x: 0, y: 0 };
    const scaleX = imgWidth / width;
    const scaleY = imgHeight / height;
    return { x: pos.x * scaleX, y: pos.y * scaleY };
  };

  const handleMouseDown = (e: Konva.KonvaEventObject<MouseEvent> ) => {
    if (e.target === e.target.getStage() || e.target.getClassName() === "Image") {
      setSelectedId(null);
      const pos = getPointerPos(e);
      setDrawing({ id: crypto.randomUUID(), x: pos.x, y: pos.y, width: 0, height: 0 });
    }
  };

  const handleMouseMove = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (!drawing) return;
    const pos = getPointerPos(e);
    setDrawing(d => d ? { ...d, width: pos.x - d.x, height: pos.y - d.y } : null);
  };

  const handleMouseUp = () => {
    if (!drawing) return;
    console.log("mouseUp, drawing:", drawing, "size:", drawing.width, drawing.height);
    if (Math.abs(drawing.width) > 5 && Math.abs(drawing.height) > 5) {
      const finalized = {
        ...drawing,
        width: Math.abs(drawing.width),
        height: Math.abs(drawing.height),
        x: drawing.width < 0 ? drawing.x + drawing.width : drawing.x,
        y: drawing.height < 0 ? drawing.y + drawing.height : drawing.y,
      };
      const updated = [...rects, finalized];
      setRects(updated);
      setSelectedId(finalized.id);
      console.log("onChange called with", updated.length, "rects");
      onChange(updated);
    }
    setDrawing(null);
  };

  const scaleToViewport = (rect: BlackoutRect) => ({
    x: rect.x * (width / imgWidth),
    y: rect.y * (height / imgHeight),
    width: rect.width * (width / imgWidth),
    height: rect.height * (height / imgHeight),
  });

  return (
    <Stage
      ref={stageRef}
      width={width}
      height={height}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      <Layer ref={layerRef}>
        <KonvaImage image={image} width={width} height={height} />
        {rects.map(rect => {
          const scaled = scaleToViewport(rect);
          return (
            <Rect
              key={rect.id}
              id={rect.id}
              x={scaled.x} y={scaled.y}
              width={scaled.width} height={scaled.height}
              fill={!isInverse
                ? "rgba(255, 120, 120, 0.4)"
                : "rgba(120, 255, 160, 0.4)"
              }
              stroke={!isInverse? "#ff3232" : " #32CD32"}
              strokeWidth={1.5}
              draggable
              onClick={() => setSelectedId(rect.id)}
              onDragEnd={e => {
                const updated = rects.map(r => r.id === rect.id
                  ? { ...r, x: (e.target.x() / width) * imgWidth, y: (e.target.y() / height) * imgHeight }
                  : r
                );
                setRects(updated);
                onChange(updated);
              }}
              onTransformEnd={e => {
                const node = e.target;
                const updated = rects.map(r => r.id === rect.id
                  ? {
                      ...r,
                      x: (node.x() / width) * imgWidth,
                      y: (node.y() / height) * imgHeight,
                      width: node.width() * node.scaleX() * (imgWidth / width),
                      height: node.height() * node.scaleY() * (imgHeight / height),
                    }
                  : r
                );
                node.scaleX(1); node.scaleY(1);
                setRects(updated);
                onChange(updated);
              }}
            />
          );
        })}

        {drawing && (
          <Rect
            x={drawing.x * (width / imgWidth)}
            y={drawing.y * (height / imgHeight)}
            width={drawing.width * (width / imgWidth)}
            height={drawing.height * (height / imgHeight)}
            fill={!isInverse
                ? "rgba(255, 120, 120, 0.4)"
                : "rgba(120, 255, 160, 0.4)"
              }
            stroke={!isInverse? "#ff3232" : " #32CD32"}
            strokeWidth={1.5}
            dash={[6, 3]}
          />
        )}

        <Transformer
          ref={transformerRef}
          boundBoxFunc={(oldBox, newBox) =>
            newBox.width < 5 || newBox.height < 5 ? oldBox : newBox
          }
        />
      </Layer>
    </Stage>
  );
}
