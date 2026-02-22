"use client";
import { useRef, useState, useEffect } from "react";
import { Stage, Layer, Image as KonvaImage, Rect, Transformer } from "react-konva";
import useImage from "use-image";

interface BlackoutRect {
  id: string;
  x: number; y: number;
  width: number; height: number;
}

interface Props {
  imageSrc: string;
  width: number;
  height: number;
  onChange: (regions: BlackoutRect[]) => void;
}

export default function BlackoutCanvas({ imageSrc, width, height, onChange }: Props) {
  const [image] = useImage(imageSrc);
  const [rects, setRects] = useState<BlackoutRect[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [drawing, setDrawing] = useState<BlackoutRect | null>(null);
  const transformerRef = useRef<any>(null);
  const stageRef = useRef<any>(null);
  const layerRef = useRef<any>(null);

  // attach transformer to selected rect
  useEffect(() => {
    if (!transformerRef.current || !layerRef.current) return;
    if (selectedId) {
      const node = layerRef.current.findOne(`#${selectedId}`);
      transformerRef.current.nodes([node]);
    } else {
      transformerRef.current.nodes([]);
    }
    transformerRef.current.getLayer().batchDraw();
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

  function handleMouseDown(e: any) {
    // clicked on empty area — deselect or start drawing
    if (e.target === e.target.getStage() || e.target.getClassName() === "Image") {
      setSelectedId(null);
      const pos = e.target.getStage().getPointerPosition();
      setDrawing({ id: crypto.randomUUID(), x: pos.x, y: pos.y, width: 0, height: 0 });
    }
  }

  function handleMouseMove(e: any) {
    if (!drawing) return;
    const pos = e.target.getStage().getPointerPosition();
    setDrawing(d => d ? { ...d, width: pos.x - d.x, height: pos.y - d.y } : null);
  }

  function handleMouseUp() {
    if (!drawing) return;
    if (Math.abs(drawing.width) > 5 && Math.abs(drawing.height) > 5) {
      const updated = [...rects, drawing];
      setRects(updated);
      setSelectedId(drawing.id);
      onChange(updated);
    }
    setDrawing(null);
  }

  return (
    <Stage ref={stageRef} width={width} height={height}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      <Layer ref={layerRef}>
        <KonvaImage image={image} width={width} height={height} />

        {rects.map(rect => (
          <Rect
            key={rect.id}
            id={rect.id}
            x={rect.x} y={rect.y}
            width={rect.width} height={rect.height}
            fill="rgba(255,50,50,0.35)"
            stroke="#ff3232"
            strokeWidth={1.5}
            draggable
            onClick={() => setSelectedId(rect.id)}
            onDragEnd={e => {
              const updated = rects.map(r => r.id === rect.id
                ? { ...r, x: e.target.x(), y: e.target.y() } : r);
              setRects(updated);
              onChange(updated);
            }}
            onTransformEnd={e => {
              const node = e.target;
              const updated = rects.map(r => r.id === rect.id ? {
                ...r,
                x: node.x(), y: node.y(),
                width: node.width() * node.scaleX(),
                height: node.height() * node.scaleY(),
              } : r);
              node.scaleX(1); node.scaleY(1);
              setRects(updated);
              onChange(updated);
            }}
          />
        ))}

        {drawing && (
          <Rect x={drawing.x} y={drawing.y}
            width={drawing.width} height={drawing.height}
            fill="rgba(255,50,50,0.25)" stroke="#ff3232"
            strokeWidth={1.5} dash={[6, 3]}
          />
        )}

        <Transformer ref={transformerRef}
          boundBoxFunc={(oldBox, newBox) => newBox.width < 5 || newBox.height < 5 ? oldBox : newBox}
        />
      </Layer>
    </Stage>
  );
}
