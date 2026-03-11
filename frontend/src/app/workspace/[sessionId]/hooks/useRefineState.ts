import { useState, useCallback, useRef } from "react";
import { Instance } from "@/lib/types";
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

  // always-current ref — handlers read from this, never from stale closure over state
  const instancesRef = useRef(instances);

  const commit = useCallback((updated: Instance[]) => {
    instancesRef.current = updated;
    setInstances(updated);
  }, []);

  // select / deselect
  const handleSelect = useCallback((id: number) => setSelectedId(id), []);
  const handleDeselect = useCallback(() => setSelectedId(null), []);

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

  const handleDiscard = useCallback(() => {
    commit(initialInstances);
    setSelectedId(null);
    setSplitMode(false);
    setSplitPoints([]);
    setSplitInstanceId(null);
    onDiscard();
  }, [initialInstances, commit, onDiscard]);

  const reinit = useCallback((newInstances: Instance[]) => {
    instancesRef.current = newInstances;
    setInstances(newInstances);
    setSelectedId(null);
    setSplitMode(false);
    setSplitPoints([]);
    setSplitInstanceId(null);
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
    handleDiscard,
  };
}
