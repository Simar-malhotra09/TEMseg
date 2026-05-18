import { useState, useCallback, useRef } from "react";
import { Instance } from "@/lib/api";
import { splitInstances } from "@/lib/api"

export interface ViewBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface Options {
  sessionId: string;
  initialInstances: Instance[];
  onSave: (instances: Instance[]) => Promise<void>;
  onDiscard: () => void;
  imgWidth: number;
  imgHeight: number;
}

export function useRefineState({
  sessionId,
  initialInstances,
  onSave,
  onDiscard,
  imgWidth,
  imgHeight,
}: Options) {

  const [instances, setInstances] = useState<Instance[]>(initialInstances);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [viewBox, setViewBox] = useState<ViewBox>({ x: 0, y: 0, w: imgWidth, h: imgHeight });
  const [splitMode, setSplitMode] = useState(false);
  const [splitPoints, setSplitPoints] = useState<[number, number][]>([]);
  const [splitInstanceId, setSplitInstanceId] = useState<number | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  // copy-paste state
  const [clipboard, setClipboard] = useState<Instance | null>(null);
  const [pasteMode, setPasteMode] = useState(false);

  // rotation state — applies to selected instance or clipboard preview
  const [rotationDeg, setRotationDeg] = useState(0);
  // original contour before rotation starts (for handle-based rotation)
  const [rotateOriginal, setRotateOriginal] = useState<[number, number][] | null>(null);

  // always-current ref — handlers read from this, never from stale closure over state
  const instancesRef = useRef(instances);

  const commit = useCallback((updated: Instance[]) => {
    instancesRef.current = updated;
    setInstances(updated);
  }, []);

  // select / deselect
  const handleSelect = useCallback((id: number) => {
    setSelectedId(id);
    setRotationDeg(0);
  }, []);
  const handleDeselect = useCallback(() => {
    setSelectedId(null);
    setRotationDeg(0);
  }, []);

  // vertex drag end — canvas calls this on mouseup with final position
  const handleVertexDragEnd = useCallback((instId: number, vi: number, pos: [number, number]) => {
    commit(instancesRef.current.map(inst => inst.id !== instId ? inst : {
      ...inst,
      contour: inst.contour.map((pt, i) => i === vi ? pos : pt),
    }));
  }, [commit]);

  // delete vertex — min 3 enforced
  const handleVertexDelete = useCallback((instId: number, vi: number) => {
    const inst = instancesRef.current.find(i => i.id === instId);
    if (!inst || inst.contour.length <= 3) return;
    commit(instancesRef.current.map(i => i.id !== instId ? i : {
      ...i,
      contour: i.contour.filter((_, idx) => idx !== vi),
    }));
  }, [commit]);

  // edge click — find closest edge, insert midpoint vertex
  const handleEdgeClick = useCallback((instId: number, pos: [number, number]) => {
    const [mx, my] = pos;
    const inst = instancesRef.current.find(i => i.id === instId);
    if (!inst) return;

    const n = inst.contour.length;
    let bestIdx = 0, bestDist = Infinity;

    for (let i = 0; i < n; i++) {
      const [ax, ay] = inst.contour[i];
      const [bx, by] = inst.contour[(i + 1) % n];
      const dx = bx - ax, dy = by - ay;
      const lenSq = dx * dx + dy * dy;
      const t = lenSq > 0
        ? Math.max(0, Math.min(1, ((mx - ax) * dx + (my - ay) * dy) / lenSq))
        : 0;
      const dist = Math.sqrt((ax + t * dx - mx) ** 2 + (ay + t * dy - my) ** 2);
      if (dist < bestDist) { bestDist = dist; bestIdx = i; }
    }

    const [ax, ay] = inst.contour[bestIdx];
    const [bx, by] = inst.contour[(bestIdx + 1) % n];
    commit(instancesRef.current.map(i => i.id !== instId ? i : {
      ...i,
      contour: [
        ...inst.contour.slice(0, bestIdx + 1),
        [(ax + bx) / 2, (ay + by) / 2] as [number, number],
        ...inst.contour.slice(bestIdx + 1),
      ],
    }));
  }, [commit]);

  // delete the currently selected instance
  const handleDeleteSelected = useCallback(() => {
    if (selectedId === null) return;
    commit(instancesRef.current.filter(i => i.id !== selectedId));
    setSelectedId(null);
  }, [selectedId, commit]);

  // enter split mode for the selected instance
  const handleEnterSplit = useCallback(() => {
    if (selectedId === null) return;
    setSplitMode(true);
    setSplitInstanceId(selectedId);
    setSplitPoints([]);
  }, [selectedId]);

  const handleCancelSplit = useCallback(() => {
    setSplitMode(false);
    setSplitInstanceId(null);
    setSplitPoints([]);
  }, []);

  const handleSplitPointPlace = useCallback((pos: [number, number]) => {
    setSplitPoints(prev => [...prev, pos]);
  }, []);

  // send split request to backend, replace old instance with results
  const handleConfirmSplit = useCallback(async () => {
    if (!splitInstanceId || splitPoints.length < 2) return;
    try {
      const data = await splitInstances(sessionId, splitInstanceId, splitPoints);
      commit(
        instancesRef.current
          .filter(i => i.id !== splitInstanceId)
          .concat(data.instances)
      );
      setSelectedId(null);
    } finally {
      setSplitMode(false);
      setSplitInstanceId(null);
      setSplitPoints([]);
    }
  }, [splitInstanceId, splitPoints, sessionId, commit]);

  const handleSave = useCallback(async () => {
    setIsSaving(true);
    try { await onSave(instancesRef.current); }
    finally { setIsSaving(false); }
  }, [onSave]);

  // copy selected instance to clipboard
  const handleCopy = useCallback(() => {
    if (selectedId === null) return;
    const inst = instancesRef.current.find(i => i.id === selectedId);
    if (!inst) return;
    setClipboard(inst);
    setPasteMode(false);
  }, [selectedId]);

  // enter paste mode — user will click on canvas to place
  const handleEnterPaste = useCallback(() => {
    if (!clipboard) return;
    setPasteMode(true);
    setSelectedId(null);
  }, [clipboard]);

  const handleCancelPaste = useCallback(() => {
    setPasteMode(false);
  }, []);

  // place copied polygon at clicked position (image-space coords)
  const handlePastePlace = useCallback((pos: [number, number]) => {
    if (!clipboard) return;
    const [px, py] = pos;

    // compute centroid of clipboard contour
    const cx = clipboard.contour.reduce((s, [x]) => s + x, 0) / clipboard.contour.length;
    const cy = clipboard.contour.reduce((s, [, y]) => s + y, 0) / clipboard.contour.length;

    // shift contour so centroid lands at click position
    const shiftedContour = clipboard.contour.map(([x, y]) =>
      [x + (px - cx), y + (py - cy)] as [number, number]
    );

    // compute new bbox
    const xs = shiftedContour.map(p => p[0]);
    const ys = shiftedContour.map(p => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);

    const maxId = instancesRef.current.reduce((m, i) => Math.max(m, i.id), 0);
    const newInstance: Instance = {
      id: maxId + 1,
      contour: shiftedContour,
      bbox: { x: minX, y: minY, w: maxX - minX, h: maxY - minY },
      area: clipboard.area,
    };

    commit([...instancesRef.current, newInstance]);
    setSelectedId(newInstance.id);
    setPasteMode(false);
  }, [clipboard, commit]);

  // start rotation drag — capture original contour
  const handleRotateStart = useCallback(() => {
    if (selectedId === null) return;
    const inst = instancesRef.current.find(i => i.id === selectedId);
    if (!inst) return;
    setRotateOriginal(inst.contour);
  }, [selectedId]);

  // during rotation drag — compute angle from centroid to mouse, rotate original contour
  const handleRotateDrag = useCallback((mousePos: [number, number]) => {
    if (selectedId === null || !rotateOriginal) return;
    const [mx, my] = mousePos;

    const cx = rotateOriginal.reduce((s, [x]) => s + x, 0) / rotateOriginal.length;
    const cy = rotateOriginal.reduce((s, [, y]) => s + y, 0) / rotateOriginal.length;

    const deg = (Math.atan2(my - cy, mx - cx) * 180) / Math.PI - 90;
    setRotationDeg(deg);

    const rad = (deg * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);

    const rotated = rotateOriginal.map(([x, y]) => {
      const dx = x - cx;
      const dy = y - cy;
      return [cx + dx * cos - dy * sin, cy + dx * sin + dy * cos] as [number, number];
    });

    const xs = rotated.map(p => p[0]);
    const ys = rotated.map(p => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);

    commit(
      instancesRef.current.map(i =>
        i.id === selectedId
          ? { ...i, contour: rotated, bbox: { x: minX, y: minY, w: maxX - minX, h: maxY - minY } }
          : i
      )
    );
  }, [selectedId, rotateOriginal, commit]);

  // end rotation drag — clear original
  const handleRotateEnd = useCallback(() => {
    setRotateOriginal(null);
  }, []);

  const handleDiscard = useCallback(() => {
    commit(initialInstances);
    setSelectedId(null);
    setSplitMode(false);
    setSplitPoints([]);
    setSplitInstanceId(null);
    setClipboard(null);
    setPasteMode(false);
    setRotationDeg(0);
    setRotateOriginal(null);
    onDiscard();
  }, [initialInstances, commit, onDiscard]);

  const reinit = useCallback((newInstances: Instance[]) => {
    instancesRef.current = newInstances;
    setInstances(newInstances);
    setSelectedId(null);
    setSplitMode(false);
    setSplitPoints([]);
    setSplitInstanceId(null);
    setClipboard(null);
    setPasteMode(false);
    setRotationDeg(0);
    setRotateOriginal(null);
  }, []);


  return {
    // state
    instances,
    selectedId,
    viewBox,
    splitMode,
    splitPoints,
    splitInstanceId,
    isSaving,
    clipboard,
    pasteMode,
    rotationDeg,
    reinit,
    // handlers
    handleSelect,
    handleDeselect,
    handleVertexDragEnd,
    handleVertexDelete,
    handleEdgeClick,
    handleDeleteSelected,
    handleEnterSplit,
    handleCancelSplit,
    handleSplitPointPlace,
    handleConfirmSplit,
    setViewBox,
    handleSave,
    handleCopy,
    handleEnterPaste,
    handleCancelPaste,
    handlePastePlace,
    handleRotateStart,
    handleRotateDrag,
    handleRotateEnd,
    handleDiscard,
  };
}
