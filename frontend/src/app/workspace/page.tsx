"use client";
import styles from "./page.module.css";
import { useQuery } from "@tanstack/react-query";
import { useState, useEffect, useRef, useSyncExternalStore } from "react";
// import {Image} from "next/Image"
import {
  Upload, Play, Sliders,
  Eye, EyeOff, Trash2, ChevronDown, AlertTriangle,
} from "lucide-react";

import { BASE_URL, Instance, getModels, uploadImage, getInstances, saveInstances, getSessionMetadata, getStats, fromPoints, fromBoxes, proposeSimilar, rfPropose, Metadata, StatsResult, subscribeToRequestActivity, getActiveRequestCount, PARTICLE_METRIC_FIELDS, ParticleMetricField } from "@/lib/api";
import { nextFreeId } from "@/lib/utils";
import { MousePointerClick, Sparkles, PenTool, BoxSelect } from "lucide-react";

import { BlackoutRect }  from "./components/BlackOutCanvas";
import  BlackoutCanvas  from "./components/BlackOutCanvas";
import ScribbleCanvas, { Scribble } from "./components/ScribbleCanvas";
import RefineCanvas from "./components/RefineCanvas";
import AnnotateCanvas from "./components/AnnotateCanvas";
import BoxAnnotateCanvas from "./components/BoxAnnotateCanvas";
import ExportPanel from "./components/ExportPanel";
import StatsPanel from "./components/StatsPanel";
import StatsDetailView from "./components/StatsDetailView";
import ParticleHighlight from "./components/ParticleHighlight";
import ExpandableHint from "./components/ExpandableHint";

import { useSegmentationState } from "./hooks/useSegmentationState";
import { useRefineState } from "./hooks/useRefineState";

interface ImgSize{
  width: number;
  height: number;
}

type SidebarTab = "segment" | "refine" | "augment";
type AugmentMethod = "click" | "box" | "similar";

// labels for the refine-mode hover tooltip field picker
const TOOLTIP_FIELD_LABELS: Record<ParticleMetricField, string> = {
  diameter: "Diameter", area: "Area", circularity: "Circularity",
  solidity: "Solidity", aspect_ratio: "Aspect Ratio", n_vertices: "Vertices", shape: "Shape",
};

// Horizontal ascii waterfall shown in the status pill while a backend
// request is in flight.
const WATERFALL_LOOP = "=^..^=   U..U  :D  T^T   ╯°□°)╯  ";
const WATERFALL_LENGTH = 20;
const WATERFALL_INTERVAL_MS = 90;

function useAsciiWaterfall(active: boolean): string {
  const offsetRef = useRef(0);
  const [frame, setFrame] = useState(() => WATERFALL_LOOP.slice(0, WATERFALL_LENGTH));

  useEffect(() => {
    if (!active) return;
    const doubled = WATERFALL_LOOP + WATERFALL_LOOP;
    const id = setInterval(() => {
      offsetRef.current = (offsetRef.current + 1) % WATERFALL_LOOP.length;
      setFrame(doubled.slice(offsetRef.current, offsetRef.current + WATERFALL_LENGTH));
    }, WATERFALL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [active]);

  return frame;
}

function AsciiWaterfall({ active }: { active: boolean }) {
  const frame = useAsciiWaterfall(active);
  return (
    <span className={styles.waterfall} aria-hidden="true">{frame}</span>
  );
}

export default function Workspace() {

  // models
  const { data: models = [] } = useQuery({ queryKey: ["models"], queryFn: getModels });
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false);
  useEffect(() => { if (models.length > 0) setSelectedModel(models[0]); }, [models]); // by def load YoloSAM 

  // session 
  const [sessionId, setSessionId] = useState<string | null>(null); // server return sessionId from uploadImage

  // image
  const [image, setImage] = useState<string | null>(null);
  const [imgSize, setImgSize] = useState<ImgSize>({ width: 0, height: 0 }); // set when we load image in client
  const [isDragging, setIsDragging] = useState(false); // to drag and drop files
  const [status, setStatus] = useState("Upload an image to begin."); // def status in our status bar
  const activeRequestCount = useSyncExternalStore(subscribeToRequestActivity, getActiveRequestCount, () => 0);
  const isBlocked = activeRequestCount > 0; // true while any backend request is in flight

  // metadata
  const [metadata, setMetadata] = useState<Metadata | null>(null);

  // stats detail view
  const [showStatsDetail, setShowStatsDetail] = useState(false);

  // sidebar tab: which of the 3 action groups is showing
  const [activeTab, setActiveTab] = useState<SidebarTab>("segment");

  // which method is showing in the Point/Box/Similar sub-window
  const [augmentMethod, setAugmentMethod] = useState<AugmentMethod>("click");

  // highlights particle when it's id gets clicked on the detailed view 
  const [highlightParticleIdx, setHighlightParticleIdx] = useState<number | null>(null);
  const [highlightShape, setHighlightShape] = useState<string | null>(null);

  const [loadedInstances, setLoadedInstances] = useState<Instance[]>([]);


  // viewport
  const viewportRef = useRef<HTMLDivElement>(null); // we want to keep track of the size of the current view: the div where the image is displayed
  const [viewportSize, setViewportSize] = useState<ImgSize>({ width: 0, height: 0 });
  const fileRef = useRef<HTMLInputElement>(null); // org file uploaded
  const gtFileRef = useRef<HTMLInputElement>(null); // gt file uploaded

  useEffect(() => {
    if (!viewportRef.current) return;
    // if the image has been uploded:
    const observer = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      setViewportSize({ width, height });
    });
    observer.observe(viewportRef.current);
    return () => observer.disconnect();
  }, [image]);

  // canvas area: the space available to display the image in, independent of the
  // image's own size. Used to scale small images up so they're easier to annotate.
  const canvasAreaRef = useRef<HTMLDivElement>(null);
  const [canvasSize, setCanvasSize] = useState<ImgSize>({ width: 0, height: 0 });

  useEffect(() => {
    if (!canvasAreaRef.current) return;
    const observer = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      setCanvasSize({ width, height });
    });
    observer.observe(canvasAreaRef.current);
    return () => observer.disconnect();
  }, []);

  // fit the image to the available canvas area, scaling up small images as well as
  // down large ones, so tiny images aren't rendered at their native (hard to annotate) size
  const displayScale = imgSize.width > 0 && imgSize.height > 0 && canvasSize.width > 0 && canvasSize.height > 0
    ? Math.min(canvasSize.width / imgSize.width, canvasSize.height / imgSize.height)
    : 1;
  const displayWidth = imgSize.width * displayScale;
  const displayHeight = imgSize.height * displayScale;

  // Restore session from ?session=... on mount (e.g. after page refresh).
  // Validates by fetching metadata; on 404 we silently clear the query.
  // Also rehydrates seg state (mask + stats) so the UI shows what's on disk.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const restored = params.get("session");
    if (!restored) return;
    (async () => {
      const meta = await getSessionMetadata(restored).catch(() => null);
      if (!meta) {
        window.history.replaceState(null, "", "/workspace");
        setStatus("Previous session no longer available — upload an image to start.");
        return;
      }
      setSessionId(restored);
      setImage(`${BASE_URL}/images/${restored}/preview`);
      setMetadata(meta);

      // Rehydrate seg state from disk in parallel. both calls return null on 404
      // (i.e. no segmentation run yet for this session).
      const [stats, instRes] = await Promise.all([
        getStats(restored).catch(() => null),
        getInstances(restored).catch(() => ({ instances: [] as Instance[] })),
      ]);
      if (stats) {
        seg.setStats(stats);
        seg.setSegDone(true);
        // Cache-bust so the browser doesn't show a stale mask.png.
        seg.setMaskUrl(`${BASE_URL}/images/${restored}/mask?t=${Date.now()}`);
        seg.setMasksVisible(true);
      }
      if (instRes?.instances) setLoadedInstances(instRes.instances);

      setStatus(
        stats
          ? `Restored session ${restored.slice(0, 8)} — ${stats.particle_count} particles.`
          : `Restored session ${restored.slice(0, 8)} — ready.`,
      );
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // zoom/pan
  // only active in `normal` mode. (outside blackout/refine.. modes)
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [panning, setPanning] = useState(false);
  const isPanning = useRef(false);
  const panStart = useRef({ x: 0, y: 0 });

  // scale bar calibration
  const [scaleBarMode, setScaleBarMode] = useState(false);
  const [scaleBarPixels, setScaleBarPixels] = useState<number | null>(null);
  const [scaleBarLineSvg, setScaleBarLineSvg] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);
  const scaleBarStart = useRef<{ x: number; y: number } | null>(null);

  // blackout region refs. written by BlackoutCanvas.onChange, read on seg/gt
  // exclusion and inclusion operations can only be perf one at a time
  const liveRegionsRef = useRef<BlackoutRect[]>([]); // tracks region to be excluded
  const liveInverseRegionsRef = useRef<BlackoutRect[]>([]); // tracks regions to be included

  // RF background scribble mode. user-marked "this is definitely background"
  // strokes, sent with /rf/propose so the RF trains on real ground truth
  // instead of assuming everything far from a known particle is background.
  const [rfBgMode, setRfBgMode] = useState(false);
  const [rfBgScribbles, setRfBgScribbles] = useState<Scribble[]>([]);
  const rfBgScribblesRef = useRef<Scribble[]>([]);
  const [rfBgBrushSize, setRfBgBrushSize] = useState(60); // scroll-to-resize on the scribble canvas

  // refine mode
  const [refineMode, setRefineMode] = useState(false);
  const [refineDone, setRefineDone] = useState(false);

  // bootstrap mode. click points on the image to propose particles via SAM.
  // Each click adds to `pendingProposals` (rendered as a yellow overlay).
  // Accept commits the whole batch via the existing PUT instances endpoint;
  // Discard or per-proposal click-to-reject clears them without a server call.
  const [bootstrapMode, setBootstrapMode] = useState(false);
  const [bootstrapBusy, setBootstrapBusy] = useState(false);
  const [pendingProposals, setPendingProposals] = useState<Instance[]>([]);

  // manual annotation mode. for the zero-detection case where SAM-with-point
  // also can't help (small/clumped/OOD particles). User draws polygons from
  // scratch; each closed polygon becomes a pendingProposal and follows the
  // same Accept/Discard commit flow.
  const [annotateMode, setAnnotateMode] = useState(false);

  // box-prompt annotation mode. drag a tight rectangle around a particle,
  // SAM segments inside via box prompt (far more reliable than point prompt
  // for OOD particles, and much faster than polygon-from-scratch).
  const [boxMode, setBoxMode] = useState(false);

  // Refresh on-disk instances whenever the user enters a canvas mode so the
  // already-committed context renders correctly. loadedInstances is otherwise
  // only populated on session restore / after Accept, so right after a fresh
  // /segment it would be empty even though the disk state is current.
  useEffect(() => {
    if (!sessionId) return;
    if (!annotateMode && !boxMode) return;
    getInstances(sessionId)
      .then(r => setLoadedInstances(r.instances ?? []))
      .catch(() => {});
  }, [annotateMode, boxMode, sessionId]);


  // segmentation hook
  const seg = useSegmentationState({ sessionId, selectedModel });

  // global polygon fill opacity
  const [polygonOpacity, setPolygonOpacity] = useState(0.2);

  // refine hook gets instantiated when refine mode is entered
  // imgSize.width guard ensures we don't init with zero dims
  const refine = useRefineState({
    sessionId: sessionId ?? "",
    initialInstances: [],
    onSave: async (updated: Instance[]) => {
        if (!sessionId) return;
        const result = await saveInstances(sessionId, updated);
        const ts = Date.now();
        seg.setMaskUrl(`${BASE_URL}${result.mask_url}?t=${ts}`);
        if (result.stats) seg.setStats(result.stats);
        setLoadedInstances(updated);  // load newly refined instances 
        setHighlightParticleIdx(null); // clear stale 
        refine.setViewBox({ x: 0, y: 0, w: imgSize.width, h: imgSize.height });
        setZoom(1);
        setPan({ x: 0, y: 0 });
        setRefineDone(true);
        setRefineMode(false);
        setStatus("Edited masks saved.");
    },
    onDiscard: () => {
      refine.setViewBox({ x: 0, y: 0, w: imgSize.width, h: imgSize.height });
      setZoom(1);
      setPan({ x: 0, y: 0 });
      setRefineMode(false);
    },
    imgWidth: imgSize.width || 1,
    imgHeight: imgSize.height || 1,
  });

  // if in refine mode, keyboard shortcuts
  useEffect(() => {
    if (!refineMode) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Backspace" || e.key === "Delete") {
        refine.handleDeleteSelected();
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "c") {
        e.preventDefault();
        refine.handleCopy();
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "v") {
        e.preventDefault();
        refine.handleEnterPaste();
      }
      if (e.key === "Escape") {
        if (refine.pasteMode) refine.handleCancelPaste();
      }
      if (e.key === "[") {
        refine.setPolygonOpacity(Math.max(0.05, refine.polygonOpacity - 0.05));
      }
      if (e.key === "]") {
        refine.setPolygonOpacity(Math.min(1, refine.polygonOpacity + 0.05));
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [refineMode, refine.handleDeleteSelected, refine.handleCopy, refine.handleEnterPaste, refine.handleCancelPaste, refine.pasteMode, refine.polygonOpacity, refine.setPolygonOpacity]);

  // image upload
  async function handleFile(file: File) {
    setStatus("Uploading...");
    seg.reset();
    setRefineMode(false);
    setMetadata(null);
    setZoom(1); // def zoom is the same size as img
    setPan({ x: 0, y: 0 });
    setRfBgMode(false);
    setRfBgScribbles([]);
    rfBgScribblesRef.current = [];
    try {
      const result = await uploadImage(file);
      if (result.error) {
        setStatus(result.error);
        return;
      }
      setSessionId(result.session_id);
      setImage(`${BASE_URL}${result.preview_url}`);
      if (result.image_info) setMetadata(result.image_info);
      // Use query string so /workspace stays a real Next.js route across refreshes.
      window.history.replaceState(null, "", `/workspace?session=${result.session_id}`);
      setStatus(`Loaded: ${file.name} — ready to segment.`);
    } catch (err) {
      console.error("upload failed:", err);
      setStatus(`Upload failed: ${(err as Error).message}`);
    }
  }

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  }

  // ground truth upload
  function onGroundTruthFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const activeRegions = seg.isInvBlackoutMode
      ? liveInverseRegionsRef.current
      : liveRegionsRef.current;
    const blackout = !seg.isInvBlackoutMode && liveRegionsRef.current.length > 0;
    const inverse = seg.isInvBlackoutMode && liveInverseRegionsRef.current.length > 0;
    seg.uploadGT(file, activeRegions, blackout, inverse);
    e.target.value = "";
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }


  // run segmentation reads live region refs, passes values into hook
  async function handleRunSegmentation() {
    if (!sessionId || !selectedModel) return;

    // pass the correct regions 
    const activeRegions = seg.isInvBlackoutMode
      ? liveInverseRegionsRef.current
      : liveRegionsRef.current;

    // only one of these should be true 
    // can add guard 
    const blackout = !seg.isInvBlackoutMode && liveRegionsRef.current.length > 0;
    const inverse = seg.isInvBlackoutMode && liveInverseRegionsRef.current.length > 0;

    setStatus(`Running ${selectedModel}...`);
    const msg = await seg.runSegmentation(activeRegions, blackout, inverse);

    // if gt available, compute scores 
    if (msg) {
      setStatus(msg);
      if (seg.groundTruth) {
        await seg.scoreGroundTruth(activeRegions, blackout, inverse);
      }
    } else {
      setStatus("Segmentation failed.");
    }
  }


  // Bootstrap click → POST /from-points, append returned proposal(s) to
  // pendingProposals. Nothing is committed until the user hits Accept.
  async function handleBootstrapClick(e: React.MouseEvent<HTMLDivElement>) {
    if (!sessionId || bootstrapBusy) return;
    if (imgSize.width === 0 || imgSize.height === 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const imgX = ((e.clientX - rect.left) / rect.width) * imgSize.width;
    const imgY = ((e.clientY - rect.top) / rect.height) * imgSize.height;

    setBootstrapBusy(true);
    setStatus(`Proposing particle at (${imgX.toFixed(0)}, ${imgY.toFixed(0)})...`);
    try {
      const res = await fromPoints(sessionId, [[imgX, imgY]], pendingProposals);
      if (res.proposals.length === 0) {
        const reason = res.rejected[0]?.reason ?? "unknown";
        setStatus(`Click rejected: ${reason}`);
        return;
      }
      const added = res.proposals[0];
      setPendingProposals(prev => [...prev, ...res.proposals]);
      setStatus(
        `Proposed #${added.id} (area ${added.area}px, SAM ${added.sam_score?.toFixed(2)}) — ${pendingProposals.length + res.proposals.length} pending. Accept to commit.`,
      );
    } catch (err) {
      console.error("bootstrap click failed:", err);
      setStatus(`Bootstrap failed: ${(err as Error).message}`);
    } finally {
      setBootstrapBusy(false);
    }
  }

  // Commit all pending proposals: fetch current on-disk instances, renumber
  // pending IDs against the on-disk set (proposals from /from-points and
  // /propose-similar are both numbered server-side without knowing about
  // each other, so naive append can collide), then PUT the combined list.
  // PUT already regenerates mask.png + stats.
  async function handleAcceptProposals() {
    if (!sessionId || pendingProposals.length === 0) return;
    setBootstrapBusy(true);
    setStatus(`Committing ${pendingProposals.length} proposal(s)...`);
    try {
      const existing: Instance[] = await getInstances(sessionId)
        .then(r => r.instances ?? [])
        .catch(() => []);
      const usedIds = existing.map(p => p.id);
      const renumbered = pendingProposals.map(p => {
        const id = nextFreeId(usedIds);
        usedIds.push(id);
        return { ...p, id };
      });
      const combined = [...existing, ...renumbered];
      const result = await saveInstances(sessionId, combined);
      seg.setMaskUrl(`${BASE_URL}${result.mask_url}?t=${Date.now()}`);
      if (result.stats) seg.setStats(result.stats);
      seg.setSegDone(true);
      seg.setMasksVisible(true);
      setLoadedInstances(combined);
      const n = pendingProposals.length;
      setPendingProposals([]);
      setStatus(`Added ${n} particle(s).`);
    } catch (err) {
      console.error("accept proposals failed:", err);
      setStatus(`Accept failed: ${(err as Error).message}`);
    } finally {
      setBootstrapBusy(false);
    }
  }

  function handleDiscardProposals() {
    if (pendingProposals.length === 0) return;
    setPendingProposals([]);
    setStatus("Discarded pending proposals.");
  }

  function handleRejectProposal(id: number) {
    setPendingProposals(prev => prev.filter(p => p.id !== id));
  }

  // Turn a manually-drawn polygon into a pending proposal. IDs are placeholder
  // (renumbered on Accept against the on-disk max), so any unique-within-batch
  // value works. sign a large negative number to make manual entries obvious
  // in logs/debug overlays alongside server-issued positive IDs.
  function handlePolygonComplete(contour: [number, number][]) {
    if (contour.length < 3) return;
    const xs = contour.map(p => p[0]);
    const ys = contour.map(p => p[1]);
    const minX = Math.min(...xs);
    const minY = Math.min(...ys);
    const maxX = Math.max(...xs);
    const maxY = Math.max(...ys);
    // shoelace formula for true polygon area
    let s = 0;
    for (let i = 0; i < contour.length; i++) {
      const [x1, y1] = contour[i];
      const [x2, y2] = contour[(i + 1) % contour.length];
      s += x1 * y2 - x2 * y1;
    }
    const area = Math.round(Math.abs(s) / 2);
    // Placeholder ID. keep stepping down so rejecting a middle polygon
    // can't cause a collision when a later annotation reuses the freed slot.
    // Renumbered against the on-disk max on Accept.
    const minId = pendingProposals.reduce((m, p) => Math.min(m, p.id), 0);
    const placeholderId = minId - 1;
    const proposal: Instance = {
      id: placeholderId,
      contour,
      bbox: {
        x: Math.round(minX),
        y: Math.round(minY),
        w: Math.round(maxX - minX),
        h: Math.round(maxY - minY),
      },
      area,
      source: "manual",
    };
    setPendingProposals(prev => [...prev, proposal]);
    setStatus(`Annotated particle (area ${area}px) — ${pendingProposals.length + 1} pending. Accept to commit.`);
  }

  // Drag-box-to-segment: POST /from-boxes with one box, append the returned
  // proposal to pendingProposals. Same accept/reject flow as the other modes.
  async function handleBoxDrawn(box: [number, number, number, number]) {
    if (!sessionId || bootstrapBusy) return;
    setBootstrapBusy(true);
    setStatus(`Segmenting box (${Math.round(box[2] - box[0])}×${Math.round(box[3] - box[1])})...`);
    try {
      const res = await fromBoxes(sessionId, [box], pendingProposals);
      if (res.proposals.length === 0) {
        const reason = res.rejected[0]?.reason ?? "unknown";
        setStatus(`Box rejected: ${reason}`);
        return;
      }
      const added = res.proposals[0];
      setPendingProposals(prev => [...prev, ...res.proposals]);
      setStatus(
        `Proposed #${added.id} (area ${added.area}px, SAM ${added.sam_score?.toFixed(2)}) — ${pendingProposals.length + res.proposals.length} pending. Accept to commit.`,
      );
    } catch (err) {
      console.error("box drawn failed:", err);
      setStatus(`Box segment failed: ${(err as Error).message}`);
    } finally {
      setBootstrapBusy(false);
    }
  }

  // Call /propose-similar. uses existing on-disk instances as the prior,
  // returns SAM box-sweep candidates. Appended to pendingProposals so the
  // same yellow-overlay accept/reject UX handles them.
  async function handleProposeSimilar() {
    if (!sessionId) return;
    setBootstrapBusy(true);
    setStatus("Searching for similar particles...");
    try {
      const res = await proposeSimilar(sessionId, undefined);
      // const res = await proposeSimilar(sessionId, "cosine");
      if (!res.proposals || res.proposals.length === 0) {
        setStatus(res.message ?? "No similar particles found.");
        return;
      }
      setPendingProposals(prev => [...prev, ...res.proposals]);
      setStatus(
        `Found ${res.proposals.length} candidate(s) from ${res.seed_count ?? "?"} seed(s). Review and accept or discard.`,
      );
    } catch (err) {
      console.error("propose similar failed:", err);
      setStatus(`Find Similar failed: ${(err as Error).message}`);
    } finally {
      setBootstrapBusy(false);
    }
  }

  async function handleRFPropose() {
    if (!sessionId) return;
    setBootstrapBusy(true);
    setStatus("Running RF recovery...");
    try {
      const res = await rfPropose(sessionId, 5, rfBgScribblesRef.current);
      if (res.error) {
        setStatus(res.error);
        return;
      }
      if (!res.proposals || res.proposals.length === 0) {
        setStatus(res.message ?? "RF found no missed regions.");
        return;
      }
      setPendingProposals(prev => [...prev, ...res.proposals]);
      setStatus(
        `RF found ${res.proposals.length} missed region(s). Review and accept or discard.`,
      );
    } catch (err) {
      console.error("RF propose failed:", err);
      setStatus(`RF Recovery failed: ${(err as Error).message}`);
    } finally {
      setBootstrapBusy(false);
    }
  }

  async function enterRefineMode() {
    (document.activeElement as HTMLElement)?.blur();
    if (!sessionId) return;
    setZoom(1);           // reset zoom before entering
    setPan({ x: 0, y: 0 });  
    const res = await getInstances(sessionId);
    refine.reinit(res.instances);  // feed instances into hook
    refine.setViewBox({ x: 0, y: 0, w: imgSize.width, h: imgSize.height });
    setRefineMode(true);
    seg.setGtVisible(false);
    setStatus("Refine mode — drag vertices, space+drag to pan.");
  }


  // blackout apply
  function handleApplyBlackout() {
    const regions = seg.isInvBlackoutMode
      ? liveInverseRegionsRef.current
      : liveRegionsRef.current;
    seg.applyBlackout(regions);
    setStatus("Regions applied — ready to segment.");
  }


  // css zoom/pan handlers. disabled when blackout or refine active
  function handleWheel(e: React.WheelEvent) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(z => {
      const next = Math.min(Math.max(z * delta, 0.5), 5);
      // center the img if zoomed out
      // if (next <= 1) setPan({ x: 0, y: 0 });
      return next;
    });
  }

  function handleMouseDown(e: React.MouseEvent) {
    if (seg.isBlackoutMode || refineMode || bootstrapMode || annotateMode || boxMode || rfBgMode) return;
    setHighlightParticleIdx(null); // clear particle highlight
    setHighlightShape(null);
    e.preventDefault();
    isPanning.current = true;
    setPanning(true);
    panStart.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
  }

  function handleMouseUp() {
    isPanning.current = false;
    setPanning(false);
  }

  function handleMouseMove(e: React.MouseEvent) {
    if (!isPanning.current) return;
    const rawX = e.clientX - panStart.current.x;
    const rawY = e.clientY - panStart.current.y;
    const excessW = (viewportSize.width * (zoom - 1)) / 2;
    const excessH = (viewportSize.height * (zoom - 1)) / 2;
    setPan({
      x: Math.min(excessW, Math.max(-excessW, rawX)),
      y: Math.min(excessH, Math.max(-excessH, rawY)),
    });
  }

  // locate particle give id 
  // used for detail view -> click id-> show particle mapping 
  async function handleLocateParticle(particleIndex: number) {
    if (!sessionId) return;
    setHighlightShape(null);  // clear shape highlight

    // if the last mode the user was in the workspace was refine mode 
    // and the user goes to the stats dashboard and uses ParticleHighlight
    // through the table, it doesn't work as it expects the workspace to be 
    // in the normal mode to add highlights. So we need to set refine mdoe 
    // to false, which 
    // WILL DISCARD ALL CURRENT EDITS. 
    // Need to see what we want to do about this
    setRefineMode(false);  
    let instances = loadedInstances;
    if (instances.length === 0) {
      const res = await getInstances(sessionId); // this should always be up to date
      instances = res.instances;
      setLoadedInstances(instances);
    }

    setHighlightParticleIdx(particleIndex);
    setShowStatsDetail(false); // pan back to workspace
    setZoom(1); // reset so the highlight isn't clipped by a stale pan/zoom
    setPan({ x: 0, y: 0 });
    seg.setMasksVisible(true);
    setStatus(`Highlighting particle id: ${particleIndex+1}`);
  }

  async function handleLocateShape(shape:string) {

    if(!sessionId) return;

    setHighlightParticleIdx(null);  // clear particle id highlight
    // if the last mode the user was in the workspace was refine mode 
    // and the user goes to the stats dashboard and uses ParticleHighlight
    // through the table, it doesn't work as it expects the workspace to be 
    // in the normal mode to add highlights. So we need to set refine mdoe 
    // to false, which 
    // WILL DISCARD ALL CURRENT EDITS. 
    // Need to see what we want to do about this
    setRefineMode(false);
    let instances= loadedInstances;
    if (instances.length === 0) {
      const res = await getInstances(sessionId); // this should always be up to date
      instances = res.instances;
      setLoadedInstances(instances);
    }

    setHighlightShape(shape);
    setShowStatsDetail(false); // pan back to workspace
    setZoom(1); // reset so highlights aren't clipped by a stale pan/zoom
    setPan({ x: 0, y: 0 });
    seg.setMasksVisible(true);
    setStatus(`Highlighting all particles of shape: ${shape}`);
  }

  function handleMetadataUpdate(meta: Metadata, newStats?: StatsResult) {
    setMetadata(meta);
    if (newStats) {
      seg.setStats(newStats);
    }
    setStatus(`Pixel size updated${meta.pixel_size != null && meta.pixel_size !== "-" ? `: ${meta.pixel_size} ${meta.pixel_unit ?? ""}` : ""}`);
  }

  function handleToggleScaleBar() {
    if (scaleBarMode) {
      handleScaleBarCancel();
    } else {
      setScaleBarMode(true);
      setScaleBarPixels(null);
      setScaleBarLineSvg(null);
    }
  }

  function handleScaleBarCancel() {
    setScaleBarMode(false);
    setScaleBarPixels(null);
    setScaleBarLineSvg(null);
    scaleBarStart.current = null;
  }

  function handleScaleBarMouseDown(e: React.MouseEvent<HTMLDivElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const imgX = ((e.clientX - rect.left) / rect.width) * imgSize.width;
    const imgY = ((e.clientY - rect.top) / rect.height) * imgSize.height;
    scaleBarStart.current = { x: imgX, y: imgY };
    setScaleBarLineSvg({ x1: imgX, y1: imgY, x2: imgX, y2: imgY });
  }

  function handleScaleBarMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    if (!scaleBarStart.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const imgX = ((e.clientX - rect.left) / rect.width) * imgSize.width;
    const imgY = ((e.clientY - rect.top) / rect.height) * imgSize.height;
    setScaleBarLineSvg({ x1: scaleBarStart.current.x, y1: scaleBarStart.current.y, x2: imgX, y2: imgY });
  }

  function handleScaleBarMouseUp(e: React.MouseEvent<HTMLDivElement>) {
    if (!scaleBarStart.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const imgX = ((e.clientX - rect.left) / rect.width) * imgSize.width;
    const imgY = ((e.clientY - rect.top) / rect.height) * imgSize.height;
    const line = { x1: scaleBarStart.current.x, y1: scaleBarStart.current.y, x2: imgX, y2: imgY };
    const pixels = Math.sqrt((line.x2 - line.x1) ** 2 + (line.y2 - line.y1) ** 2);
    scaleBarStart.current = null;
    if (pixels < 5) {
      setScaleBarLineSvg(null);
      return;
    }
    setScaleBarLineSvg(line);
    setScaleBarPixels(pixels);
  }

  // track which regions are to be used
  const activeRegions = seg.isInvBlackoutMode
    ? seg.invBlackoutRegions
    : seg.blackoutRegions;
  const hasRegions = activeRegions.length > 0;

  // Switching tabs abandons any in-progress work in the tab being left, same
  // rule already used when jumping to the stats dashboard mid-refine (see
  // handleLocateParticle/handleLocateShape).
  function switchTab(next: SidebarTab) {
    if (next === activeTab) return;
    if (refineMode) refine.handleDiscard();
    if (pendingProposals.length > 0) handleDiscardProposals();
    setBootstrapMode(false);
    setBoxMode(false);
    setAnnotateMode(false);
    setRfBgMode(false);
    seg.setIsBlackoutMode(false);
    setActiveTab(next);
  }

  // Switching method within the Point/Box/Similar sub-window exits whichever
  // of Click/Box was active, same idea as switchTab but nothing to discard.
  function switchAugmentMethod(next: AugmentMethod) {
    if (next === augmentMethod) return;
    setBootstrapMode(false);
    setBoxMode(false);
    setAugmentMethod(next);
  }

  return (
    <>
      {/*If in stats dashboard hide the workspace*/}
      {showStatsDetail && seg.stats && sessionId && (
        <StatsDetailView
          stats={seg.stats}
          metadata={metadata}
          groundTruthScore={seg.groundTruthScore as any}
          sessionId={sessionId}
          onBack={() => setShowStatsDetail(false)}
          onLocateParticle={handleLocateParticle}
          onLocateShape={handleLocateShape}
        />
      )}

      {/* Main workspace */}
      <div className={styles.workspaceRoot}style={{display: showStatsDetail? "none" : "flex" }}>

        {/* topbar */}
        <header className={styles.topbar}>
          <div className={styles.topbarLeft}>
            <div
              onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); window.location.href = "/workspace/"; }}
              title="New workspace"
              style={{ cursor: "pointer" }}
            >
              <span className={styles.logo}>TEM<span className={styles.logoAccent}>seg</span></span>
            </div>
            <span className={styles.sessionTag}>
              session · {sessionId ? sessionId.slice(0, 8) : "upload image to start"}
            </span>
          </div>
          <div className={styles.topbarRight}>
            {seg.regionsOutOfSync && (
              <span className={styles.warnPill}>
                <AlertTriangle size={11} /> blackout regions changed — re-run seg
              </span>
            )}

            {/*Display zoom size and reset to normal on click*/}
            <span
              className={`${styles.statusPill} ${isBlocked ? styles.statusPillBlocked : ""}`}
              key={isBlocked ? "blocked" : status}
            >
              {isBlocked ? <AsciiWaterfall active /> : status}
            </span>
            {(zoom !== 1 || pan.x !== 0 || pan.y !== 0) && (
              <button type="button" className={styles.zoomReset} onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}>
                {Math.round(zoom * 100)}% ✕
              </button>
            )}
          </div>
        </header>

        {/* Main workspace */}
        <div className={styles.workspaceBody}>

          {/* 
              left sidebar, handles 
                a. model selection dropdown 
                b. run seg button 
                c. hide/show masks
                d. enter/exit refine mode
                e. exclude/isolate regions 
                f. upload/compute gt
                g. export (modified) masks
            */}
          <aside className={styles.sidebar}>

            {/* model selector */}
            <section className={styles.sidebarSection}>
              <p className={styles.sidebarLabel}>Model</p>
              <div className={styles.dropdownWrap}>
                <button type="button" className={styles.dropdownBtn} onClick={() => setModelDropdownOpen(o => !o)}>
                  {selectedModel} <ChevronDown size={14} />
                </button>
                {modelDropdownOpen && (
                  <ul className={styles.dropdownList}>
                    {models.map(m => (
                      <li key={m}
                        className={`${styles.dropdownItem} ${m === selectedModel ? styles.dropdownItemActive : ""}`}
                        onClick={() => { setSelectedModel(m); setModelDropdownOpen(false); }}
                      >{m}</li>
                    ))}
                  </ul>
                )}
              </div>
            </section>

            {/* tab bar. switchTab discards any in-progress work in the tab
                being left (unsaved refine edits, pending augment proposals). */}
            <div className={styles.tabBar}>
              <button type="button"
                className={`${styles.tabBtn} ${activeTab === "segment" ? styles.tabBtnActive : ""}`}
                onClick={() => switchTab("segment")}
              >Segment</button>
              <button type="button"
                className={`${styles.tabBtn} ${activeTab === "refine" ? styles.tabBtnActive : ""}`}
                onClick={() => switchTab("refine")}
                disabled={!seg.segDone}
              >Refine</button>
              <button type="button"
                className={`${styles.tabBtn} ${activeTab === "augment" ? styles.tabBtnActive : ""}`}
                onClick={() => switchTab("augment")}
                disabled={!sessionId}
              >Augment</button>
            </div>

            {activeTab === "segment" && (
              <>
                <section className={styles.sidebarSection}>
                  <p className={styles.sidebarLabel}>Actions</p>

                  {/*[ACTION]- run seg */}
                  <button
                    type="button"
                    className={`${styles.actionBtn} ${styles.actionBtnPrimary}`}
                    onClick={handleRunSegmentation}
                    disabled={!image || seg.isSegmenting}
                  >
                    <Play size={14} /> {seg.isSegmenting ? "Running..." : "Run Segmentation"}
                  </button>
                </section>
                <section>
                  {/*[ACTION]- show/hide masks */}
                  <button type="button" className={styles.actionBtn} disabled={!seg.segDone}
                    onClick={() => seg.setMasksVisible(v => !v)}>
                    {seg.masksVisible ? <EyeOff size={14} /> : <Eye size={14} />}
                    {seg.masksVisible ? "Hide Masks" : "Show Masks"}
                  </button>
                </section>
                {seg.segDone && seg.masksVisible && (
                  <div style={{ marginBottom: 8 }}>
                    <p className={styles.sidebarHint}>Fill mask opacity: {(polygonOpacity * 100).toFixed(0)}%</p>
                    <input
                      type="range"
                      min={5}
                      max={100}
                      value={Math.round(polygonOpacity * 100)}
                      onChange={e => setPolygonOpacity(Number(e.target.value) / 100)}
                      style={{ width: "100%", accentColor: "#7ee8a2" }}
                    />
                  </div>
                )}

                {/* [ACTION] blackout regions */}
                <section className={styles.sidebarSection}>
                  <p className={styles.sidebarLabel}>Blackout Regions</p>
                  <div style={{ display: "flex", gap: 4, marginBottom: 6 }}>
                    <button type="button"
                      className={`${styles.actionBtn} ${!seg.isInvBlackoutMode ? styles.actionBtnPrimary : ""}`}
                      onClick={() => seg.setIsInvBlackoutMode(false)}
                      disabled={!image}
                    >Exclude</button>
                    <button type="button"
                      className={`${styles.actionBtn} ${seg.isInvBlackoutMode ? styles.actionBtnPrimary : ""}`}
                      onClick={() => seg.setIsInvBlackoutMode(true)}
                      disabled={!image}
                    >Isolate</button>
                  </div>
                  <button type="button" className={styles.actionBtn} disabled={!image}
                    onClick={() => { seg.setMasksVisible(false); seg.setIsBlackoutMode(b => !b); }}>
                    <Trash2 size={14} /> {seg.isBlackoutMode ? "Exit" : "Draw Regions"}
                  </button>
                  {seg.isBlackoutMode && (
                    <button type="button" className={styles.actionBtn} onClick={handleApplyBlackout}>
                      Apply
                    </button>
                  )}
                  {hasRegions && !seg.isBlackoutMode && (
                    <button type="button" className={styles.actionBtn} onClick={() => {
                      seg.clearRegions();
                      if (seg.isInvBlackoutMode) liveInverseRegionsRef.current = [];
                      else liveRegionsRef.current = [];
                      setStatus("Regions cleared.");
                    }}>
                      Clear Regions
                    </button>
                  )}
                  <p className={styles.sidebarHint}>
                    {seg.isInvBlackoutMode
                      ? "Isolate: model only sees selected regions."
                      : "Exclude: model ignores selected regions."}
                  </p>
                </section>

                {/* [ACTION] ground truth */}
                <section className={styles.sidebarSection}>
                  <p className={styles.sidebarLabel}>Ground Truth</p>
                  <button type="button" className={styles.actionBtn}
                    onClick={() => gtFileRef.current?.click()}
                    disabled={!sessionId}>
                    <Upload size={14} /> Upload GT
                  </button>
                  <input ref={gtFileRef} type="file" accept=".npy,.png,.tiff,.tif,.json"
                    hidden onChange={onGroundTruthFileChange} />
                  <p className={styles.sidebarHint}>
                    {seg.groundTruth ? seg.groundTruthStatus : "Upload a ground truth mask to compute accuracy scores."}
                  </p>
                  {seg.gtUrl && (
                    <button type="button" className={styles.actionBtn} onClick={() => seg.setGtVisible(v => !v)}>
                      {seg.gtVisible ? <EyeOff size={14} /> : <Eye size={14} />}
                      {seg.gtVisible ? "Hide GT" : "Show GT"}
                    </button>
                  )}
                </section>
              </>
            )}

            {activeTab === "refine" && (
              <section>
                {/*[ACTION]: refine masks */}
                <button type="button" className={styles.actionBtn} disabled={!seg.segDone}
                  onClick={
                    refineMode
                      ? () => {
                          refine.handleSave();
                          setTimeout(() => {
                            seg.setMasksVisible(b => !b);
                          }, 300);

                        }
                      : () => {
                          enterRefineMode();

                          setTimeout(() => {
                            seg.setMasksVisible(b => !b);
                          }, 300);
                        }
                  }
                  >
                  <Sliders size={14} />
                  {refineMode
                    ? (refine.isSaving ? "Saving..." : "Save Refinements")
                    : "Refine Masks"}
                </button>

                {/* discard all changes made while refining masks*/}
                {refineMode && (
                  <button type="button" className={styles.actionBtn} onClick={refine.handleDiscard}>
                    Discard
                  </button>
                )}

                {/* refine mode controls  */}
                {refineMode && (
                  <section className={styles.sidebarSection}>
                    <p className={styles.sidebarLabel}>Refine</p>

                    <div style={{ marginBottom: 8 }}>
                      <p className={styles.sidebarHint}>Fill mask opacity: {(refine.polygonOpacity * 100).toFixed(0)}%</p>
                      <input
                        type="range"
                        min={5}
                        max={100}
                        value={Math.round(refine.polygonOpacity * 100)}
                        onChange={e => refine.setPolygonOpacity(Number(e.target.value) / 100)}
                        style={{ width: "100%", accentColor: "#7ee8a2" }}
                      />
                    </div>

                    <div style={{ marginBottom: 8 }}>
                      <p className={styles.sidebarHint}>Hover tooltip fields</p>
                      {PARTICLE_METRIC_FIELDS.map(field => (
                        <label key={field} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11 }}>
                          <input
                            type="checkbox"
                            checked={refine.tooltipFields.includes(field)}
                            onChange={e => refine.setTooltipFields(
                              e.target.checked
                                ? [...refine.tooltipFields, field]
                                : refine.tooltipFields.filter((f: ParticleMetricField) => f !== field)
                            )}
                          />
                          {TOOLTIP_FIELD_LABELS[field]}
                        </label>
                      ))}
                    </div>

                    {refine.selectedId !== null && !refine.splitMode && !refine.pasteMode && (
                      <>
                        <button type="button" className={styles.actionBtn} onClick={refine.handleEnterSplit}>
                          Split Instance
                        </button>
                        <button type="button" className={styles.actionBtn} onClick={refine.handleCopy}>
                          Copy (⌘C)
                        </button>
                      </>
                    )}

                    {refine.clipboard && !refine.pasteMode && !refine.splitMode && (
                      <button type="button" className={styles.actionBtn} onClick={refine.handleEnterPaste}>
                        Paste (⌘V)
                      </button>
                    )}

                    {refine.pasteMode && (
                      <>
                        <p className={styles.sidebarHint}>
                          Click on image to place copied polygon
                        </p>
                        <button type="button" className={styles.actionBtn} onClick={refine.handleCancelPaste}>
                          Cancel (Esc)
                        </button>
                      </>
                    )}

                    {refine.splitMode && (
                      <>
                        <p className={styles.sidebarHint}>
                          {refine.splitPoints.length} point{refine.splitPoints.length !== 1 ? "s" : ""} placed
                        </p>
                        <p className={styles.sidebarHint}>
                          Click each particle to place a seed point
                        </p>
                        <button type="button"
                          className={`${styles.actionBtn} ${styles.actionBtnPrimary}`}
                          disabled={refine.splitPoints.length < 2}
                          onClick={refine.handleConfirmSplit}
                        >
                          Confirm Split
                        </button>
                        <button type="button" className={styles.actionBtn} onClick={refine.handleCancelSplit}>
                          Cancel Split
                        </button>
                      </>
                    )}
                  </section>
                )}
              </section>
            )}

            {activeTab === "augment" && (
              <section className={styles.sidebarSection}>
                <div className={styles.subWindow}>
                  <p className={styles.subWindowTitle}>
                    <button type="button"
                      className={`${styles.tabBtn} ${styles.titleTab} ${augmentMethod === "click" ? styles.tabBtnActive : ""}`}
                      onClick={() => switchAugmentMethod("click")}
                    >Point</button>
                    {"/"}
                    <button type="button"
                      className={`${styles.tabBtn} ${styles.titleTab} ${augmentMethod === "box" ? styles.tabBtnActive : ""}`}
                      onClick={() => switchAugmentMethod("box")}
                    >Box</button>
                    {"/"}
                    <button type="button"
                      className={`${styles.tabBtn} ${styles.titleTab} ${augmentMethod === "similar" ? styles.tabBtnActive : ""}`}
                      onClick={() => switchAugmentMethod("similar")}
                    >Similar</button>
                  </p>

                  {augmentMethod === "click" && (
                    <>
                      {/* bootstrap mode. click on the image to *propose* particles via
                          SAM point-prompt. Requires a prior /segment run so the SAM
                          image embedding is cached server-side (the backend 400s
                          otherwise, but the disabled state is the clean UX). */}
                      <button type="button"
                        className={styles.actionBtn}
                        disabled={!sessionId || refineMode || annotateMode || boxMode || !seg.segDone}
                        onClick={() => setBootstrapMode(b => !b)}
                        title={!seg.segDone ? "Run segmentation first to cache the SAM embedding" : undefined}
                      >
                        <MousePointerClick size={14} />
                        {bootstrapMode ? "Stop Clicking" : "Click to Add Particles"}
                      </button>
                      <ExpandableHint summary="Point prompt tips">
                        <p className={styles.sidebarHint}>
                          Click a missed particle so SAM segments it from that point using a point prompt, best for particles that sit clearly apart from their neighbors.
                        </p>
                      </ExpandableHint>
                    </>
                  )}

                  {augmentMethod === "box" && (
                    <>
                      {/* box-prompt annotation. drag a tight box, SAM segments inside.
                          Preferred over click-to-add for OOD particles where point
                          prompts bleed; preferred over polygon for speed. */}
                      <button type="button"
                        className={styles.actionBtn}
                        disabled={!sessionId || refineMode || annotateMode || bootstrapMode || !seg.segDone}
                        onClick={() => setBoxMode(m => !m)}
                        title={!seg.segDone ? "Run segmentation first to cache the SAM embedding" : undefined}
                      >
                        <BoxSelect size={14} />
                        {boxMode ? "Stop Boxing" : "Box to Add Particles"}
                      </button>
                      <ExpandableHint summary="Box prompt tips">
                        <p className={styles.sidebarHint}>
                          Drag a box around a missed particle so SAM segments it from that box using a box prompt, best when particles sit close together and a point would bleed into neighbors.
                        </p>
                      </ExpandableHint>
                      {boxMode && (
                        <p className={styles.sidebarShortcuts}>
                          [Space]+drag pan · [wheel] zoom · [Esc] cancel
                        </p>
                      )}
                    </>
                  )}

                  {augmentMethod === "similar" && (
                    <>
                      {/* propose-similar: build a SAM-embedding prior from existing
                          annotations and find more candidates across the image. Needs
                          at least one annotated particle to construct the prior. */}
                      <button type="button"
                        className={styles.actionBtn}
                        disabled={
                          !sessionId ||
                          refineMode ||
                          annotateMode ||
                          boxMode ||
                          rfBgMode ||
                          !seg.segDone ||
                          !(seg.stats?.particle_count ?? 0) ||
                          bootstrapBusy
                        }
                        onClick={handleProposeSimilar}
                        title={
                          !(seg.stats?.particle_count ?? 0)
                            ? "Annotate at least one particle first to build the prior"
                            : undefined
                        }
                      >
                        <Sparkles size={14} />
                        {bootstrapBusy ? "Searching..." : "Find Similar Particles"}
                      </button>
                      <ExpandableHint summary="Similar search tips">
                        <p className={styles.sidebarHint}>
                          Search the rest of the image for particles like the ones you already annotated, once you have a few examples to build from.
                        </p>
                      </ExpandableHint>
                    </>
                  )}
                </div>

                <div className={styles.subWindow}>
                  <p className={styles.subWindowTitle}>RF Recovery</p>

                  {/* RF background scribble. mark patches that are definitely
                      background so the RF trains on real ground truth instead of
                      assuming everything far from a known particle is background
                      (which mislabels particles the model missed). Comes first —
                      RF Recover Missed needs this to have anything to train on. */}
                  <button type="button"
                    className={`${styles.actionBtn} ${rfBgMode ? styles.actionBtnPrimary : ""}`}
                    disabled={
                      !sessionId ||
                      refineMode ||
                      annotateMode ||
                      boxMode ||
                      !seg.segDone ||
                      bootstrapBusy
                    }
                    onClick={() => setRfBgMode(v => !v)}
                  >
                    <PenTool size={14} /> {rfBgMode ? "Done Marking" : "Mark RF Background"}
                  </button>
                  {rfBgScribbles.length > 0 && !rfBgMode && (
                    <button type="button" className={styles.actionBtn} onClick={() => {
                      rfBgScribblesRef.current = [];
                      setRfBgScribbles([]);
                      setStatus("RF background scribbles cleared.");
                    }}>
                      Clear Background Marks
                    </button>
                  )}
                  {rfBgMode ? (
                    <>
                      <ExpandableHint summary="Background marking tips">
                        <p className={styles.sidebarHint}>
                          Mark patches that definitely contain no particle so the classifier learns real background instead of guessing.
                        </p>
                      </ExpandableHint>
                      <p className={styles.sidebarShortcuts}>
                        · [scroll]: resize brush 
                      </p>
                    </>
                  ) : (
                    <ExpandableHint summary="Why mark background">
                      <p className={styles.sidebarHint}>
                        Mark a few patches that definitely contain no particle, since RF Recover below needs real background examples and cannot guess them safely on its own.
                      </p>
                    </ExpandableHint>
                  )}

                  {/* RF recovery. requires prior /segment so SAM embedding is cached,
                      and requires marked background since the RF trains on your annotated
                      particles as foreground and your marks as background, and has
                      nothing reliable to learn from otherwise. */}
                  <button type="button"
                    className={styles.actionBtn}
                    disabled={
                      !sessionId ||
                      refineMode ||
                      annotateMode ||
                      boxMode ||
                      rfBgMode ||
                      !seg.segDone ||
                      rfBgScribbles.length === 0 ||
                      bootstrapBusy
                    }
                    onClick={handleRFPropose}
                    title={
                      !seg.segDone
                        ? "Run segmentation first"
                        : rfBgScribbles.length === 0
                          ? "Mark background above first so the RF has both classes to train on"
                          : undefined
                    }
                  >
                    <Sparkles size={14} />
                    {bootstrapBusy ? "Running..." : "RF Recover Missed"}
                  </button>
                  <ExpandableHint summary="RF recovery tips">
                    <p className={styles.sidebarHint}>
                      Train a classifier on your annotated particles as foreground and your marked background as the negative class to catch particles missed everywhere in the image.
                    </p>
                  </ExpandableHint>
                </div>

                <div className={styles.subWindow}>
                  <p className={styles.subWindowTitle}>Manual Annotation</p>

                  {/* manual annotation. fallback when both YOLO and SAM-with-point
                      fail (small / clumped / OOD particles). Doesn't require a
                      prior /segment run since it doesn't use any model output. */}
                  <button type="button"
                    className={styles.actionBtn}
                    disabled={!sessionId || refineMode || bootstrapMode || boxMode}
                    onClick={() => setAnnotateMode(m => !m)}
                  >
                    <PenTool size={14} />
                    {annotateMode ? "Stop Annotating" : "Annotate Manually"}
                  </button>
                  {annotateMode && (
                    <>
                      <ExpandableHint summary="Manual trace tips">
                        <p className={styles.sidebarHint}>
                          Trace a particle by hand when SAM cannot separate it from its neighbors, since this does not depend on any model.
                        </p>
                      </ExpandableHint>
                      <p className={styles.sidebarShortcuts}>
                        [Enter] or double-click close<br />
                        [Backspace]: undo vertex<br />
                        [Space]+drag: pan<br />
                        [Esc]: cancel
                      </p>
                    </>
                  )}
                </div>

                {pendingProposals.length > 0 && (
                  <div className={styles.proposalsBar}>
                    <button type="button"
                      className={`${styles.actionBtn} ${styles.actionBtnPrimary}`}
                      onClick={handleAcceptProposals}
                      disabled={bootstrapBusy}
                    >
                      Accept {pendingProposals.length} Proposal{pendingProposals.length === 1 ? "" : "s"}
                    </button>
                    <button type="button"
                      className={styles.actionBtn}
                      onClick={handleDiscardProposals}
                      disabled={bootstrapBusy}
                    >
                      Discard
                    </button>
                    <p className={styles.sidebarHint}>
                      Click a proposal on the canvas to reject it individually.
                    </p>
                  </div>
                )}
              </section>
            )}

            <section>
              {sessionId && (
                <ExportPanel
                  sessionId={sessionId}
                  segDone={seg.segDone}
                  refineDone={refineDone}
                  hasStats={!!seg.stats}
                />
              )}
            </section>

          </aside>

          {/* canvas */}
          <main className={styles.canvasArea} ref={canvasAreaRef}>
            {!image ? (
              <div
                className={`${styles.dropZone} ${isDragging ? styles.dropZoneDragging : ""}`}
                onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={onDrop}
                onClick={() => fileRef.current?.click()}
              >
                <Upload size={32} strokeWidth={1.5} />
                <p className={styles.dropLabel}>Drop image here or click to upload</p>
                <p className={styles.dropHint}>EMD, TIF, TIFF, JPEG, PNG, NPY supported</p>
              </div>
            ) : (
              <div
                className={styles.imageViewport}
                ref={viewportRef}
                onWheel={handleWheel}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                style={{
                  position: "relative",
                  display: "inline-block",
                  lineHeight: 0,
                  transform: seg.isBlackoutMode || refineMode || bootstrapMode || annotateMode || boxMode || scaleBarMode || rfBgMode
                    ? "none"
                    : `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
                  transformOrigin: "center center",
                  transition: isPanning.current ? "none" : "transform 0.05s ease-out",
                  cursor: seg.isBlackoutMode || refineMode || bootstrapMode || annotateMode || boxMode || scaleBarMode || rfBgMode
                    ? "default"
                    : panning ? "grabbing" : "grab",
                }}
              >
                {/* base image. hidden when blackout/refine canvas is active (they render their own) */}
                <img
                  src={image}
                  alt="TEM input"
                  className={styles.temImage}
                  style={{
                    display: "block",
                    visibility: seg.isBlackoutMode || refineMode || annotateMode || boxMode || rfBgMode ? "hidden" : "visible",
                    ...(displayWidth > 0 && displayHeight > 0
                      ? { width: displayWidth, height: displayHeight }
                      : {}),
                  }}
                  onLoad={e => setImgSize({
                    width: e.currentTarget.naturalWidth,
                    height: e.currentTarget.naturalHeight,
                  })}
                />

                {/* blackout canvas */}
                {seg.isBlackoutMode && imgSize.width > 0 && (
                  <div style={{ position: "absolute", top: 0, left: 0 }}>
                    <BlackoutCanvas
                      imageSrc={image} 
                      width={viewportSize.width}
                      height={viewportSize.height}
                      imgWidth={imgSize.width}
                      imgHeight={imgSize.height}
                      isInverse={seg.isInvBlackoutMode}
                      initialRegions={seg.isInvBlackoutMode ? seg.invBlackoutRegions : seg.blackoutRegions}
                      onChange={regions => {
                        if (seg.isInvBlackoutMode) {
                          liveInverseRegionsRef.current = regions;
                          seg.setInvBlackoutRegions(regions);
                        } else {
                          liveRegionsRef.current = regions;
                          seg.setBlackoutRegions(regions);
                        }
                      }}
                    />
                  </div>
                )}

                {/* RF background scribble canvas. Freehand brush strokes */}
                {rfBgMode && imgSize.width > 0 && (
                  <div style={{ position: "absolute", top: 0, left: 0 }}>
                    <ScribbleCanvas
                      imageSrc={image}
                      width={viewportSize.width}
                      height={viewportSize.height}
                      imgWidth={imgSize.width}
                      imgHeight={imgSize.height}
                      initialStrokes={rfBgScribbles}
                      brushSize={rfBgBrushSize}
                      onBrushSizeChange={setRfBgBrushSize}
                      onChange={scribbles => {
                        rfBgScribblesRef.current = scribbles;
                        setRfBgScribbles(scribbles);
                      }}
                    />
                  </div>
                )}

                {/* refine canvas */}
                {refineMode && imgSize.width > 0 && (
                  <div style={{ position: "absolute", top: 0, left: 0 }}>
                    <RefineCanvas
                      imageSrc={image}
                      width={viewportSize.width}
                      height={viewportSize.height}
                      imgWidth={imgSize.width}
                      imgHeight={imgSize.height}
                      instances={refine.instances}
                      selectedId={refine.selectedId}
                      viewBox={refine.viewBox}
                      splitMode={refine.splitMode}
                      splitPoints={refine.splitPoints}
                      pasteMode={refine.pasteMode}
                      clipboard={refine.clipboard}
                      polygonOpacity={refine.polygonOpacity}
                      stats={seg.stats}
                      visibleTooltipFields={refine.tooltipFields}
                      onSelect={refine.handleSelect}
                      onDeselect={refine.handleDeselect}
                      onVertexDragEnd={refine.handleVertexDragEnd}
                      onVertexDelete={refine.handleVertexDelete}
                      onEdgeClick={refine.handleEdgeClick}
                      onSplitPointPlace={refine.handleSplitPointPlace}
                      onPastePlace={refine.handlePastePlace}
                      onRotateStart={refine.handleRotateStart}
                      onRotateDrag={refine.handleRotateDrag}
                      onRotateEnd={refine.handleRotateEnd}
                      onViewBoxChange={refine.setViewBox}
                    />
                  </div>
                )}

                {/* seg mask overlay */}
                {seg.segDone && seg.masksVisible && seg.maskUrl && (
                  <img src={seg.maskUrl} style={{
                    position: "absolute", top: 0, left: 0,
                    width: "100%", height: "100%",
                    opacity: polygonOpacity, mixBlendMode: "screen",
                    pointerEvents: "none",
                  }} />
                )}

                {/* manual annotation canvas,SVG overlay, in image-space */}
                {annotateMode && !refineMode && image && imgSize.width > 0 && (
                  <AnnotateCanvas
                    imageSrc={image}
                    imgWidth={imgSize.width}
                    imgHeight={imgSize.height}
                    viewportWidth={viewportSize.width}
                    viewportHeight={viewportSize.height}
                    existingInstances={loadedInstances}
                    pendingProposals={pendingProposals}
                    onPolygonComplete={handlePolygonComplete}
                  />
                )}

                {/* box-prompt annotation canvas. drag rectangles, SAM segments */}
                {boxMode && !refineMode && image && imgSize.width > 0 && (
                  <BoxAnnotateCanvas
                    imageSrc={image}
                    imgWidth={imgSize.width}
                    imgHeight={imgSize.height}
                    viewportWidth={viewportSize.width}
                    viewportHeight={viewportSize.height}
                    busy={bootstrapBusy}
                    existingInstances={loadedInstances}
                    pendingProposals={pendingProposals}
                    onBoxDrawn={handleBoxDrawn}
                  />
                )}

                {/* bootstrap click capture.sits on top, transparent, catches clicks */}
                {bootstrapMode && !refineMode && imgSize.width > 0 && (
                  <div
                    onClick={handleBootstrapClick}
                    style={{
                      position: "absolute", top: 0, left: 0,
                      width: "100%", height: "100%",
                      cursor: bootstrapBusy ? "wait" : "crosshair",
                      // subtle tint so the user knows the mode is active
                      background: "rgba(108, 99, 255, 0.06)",
                      pointerEvents: "auto",
                    }}
                  />
                )}

                {/* scale bar line. rendered under the mouse capture overlay */}
                {scaleBarMode && scaleBarLineSvg && imgSize.width > 0 && (
                  <svg
                    viewBox={`0 0 ${imgSize.width} ${imgSize.height}`}
                    preserveAspectRatio="xMidYMid meet"
                    style={{
                      position: "absolute", top: 0, left: 0,
                      width: "100%", height: "100%",
                      pointerEvents: "none",
                      zIndex: 14,
                    }}
                  >
                    <line
                      x1={scaleBarLineSvg.x1} y1={scaleBarLineSvg.y1}
                      x2={scaleBarLineSvg.x2} y2={scaleBarLineSvg.y2}
                      stroke="#7ee8a2"
                      strokeWidth={2}
                      strokeDasharray={scaleBarPixels ? undefined : "8 4"}
                    />
                    <circle cx={scaleBarLineSvg.x1} cy={scaleBarLineSvg.y1} r={5} fill="#7ee8a2" />
                    <circle cx={scaleBarLineSvg.x2} cy={scaleBarLineSvg.y2} r={5} fill="#7ee8a2" />
                  </svg>
                )}

                {/* scale bar mouse capture to draw the measurement line */}
                {scaleBarMode && !scaleBarPixels && imgSize.width > 0 && (
                  <div
                    onMouseDown={handleScaleBarMouseDown}
                    onMouseMove={handleScaleBarMouseMove}
                    onMouseUp={handleScaleBarMouseUp}
                    onMouseLeave={e => { if (scaleBarStart.current) handleScaleBarMouseUp(e); }}
                    style={{
                      position: "absolute", top: 0, left: 0,
                      width: "100%", height: "100%",
                      cursor: "crosshair",
                      background: "rgba(126, 232, 162, 0.04)",
                      zIndex: 15,
                    }}
                  />
                )}

                {/* pending proposals overlay: yellow outlines, click to reject.
                    Sits above the bootstrap click-capture so reject clicks
                    aren't swallowed as new proposals. Suppressed while a
                    canvas mode is active and those canvases render their own
                    in-image-space proposal overlay so the viewBox transforms
                    them together with the image. */}
                {pendingProposals.length > 0 && !refineMode && !seg.isBlackoutMode && !annotateMode && !boxMode && imgSize.width > 0 && (
                  <svg
                    viewBox={`0 0 ${imgSize.width} ${imgSize.height}`}
                    preserveAspectRatio="xMidYMid meet"
                    style={{
                      position: "absolute", top: 0, left: 0,
                      width: "100%", height: "100%",
                      pointerEvents: "none",
                      zIndex: 11,
                    }}
                  >
                    {pendingProposals.map(p => {
                      if (!p.contour || p.contour.length < 3) return null;
                      const pts = p.contour.map(([x, y]) => `${x},${y}`).join(" ");
                      return (
                        <g key={p.id}>
                          <polygon
                            points={pts}
                            fill="#ffd166"
                            fillOpacity={0.18}
                            stroke="#ffd166"
                            strokeWidth={2}
                            style={{ pointerEvents: "auto", cursor: "pointer" }}
                            onClick={(ev) => {
                              ev.stopPropagation();
                              handleRejectProposal(p.id);
                            }}
                          >
                            <title>Click to reject proposal #{p.id} (area {p.area}px)</title>
                          </polygon>
                        </g>
                      );
                    })}
                  </svg>
                )}

                {/* GT overlay */}
                {seg.gtVisible && seg.gtUrl && (
                  <img src={seg.gtUrl} style={{
                    position: "absolute", top: 0, left: 0,
                    width: "100%", height: "100%",
                    opacity: 0.5, mixBlendMode: "screen",
                    filter: "hue-rotate(200deg)",
                    pointerEvents: "none",
                  }} />
                )}

                {/* particle highlight from stats table */}
                {highlightParticleIdx !== null && loadedInstances.length > highlightParticleIdx && !refineMode && !seg.isBlackoutMode && (
                  <ParticleHighlight
                    instance={loadedInstances[highlightParticleIdx]}
                    imgWidth={imgSize.width}
                    imgHeight={imgSize.height}
                    viewportWidth={viewportSize.width}
                    viewportHeight={viewportSize.height}
                  />
                )}

                {/* highlights all particles of that shape selected from stats table */}
                {highlightShape !== null && !refineMode && !seg.isBlackoutMode &&
                  loadedInstances
                    .filter(inst => {
                      // find this instance's shape from stats
                      const particle = seg.stats?.particles?.find((p: any) => p.id === inst.id);
                      return particle?.shape === highlightShape;
                    })
                    .map(inst => (
                      <ParticleHighlight
                        key={inst.id}
                        instance={inst}
                        imgWidth={imgSize.width}
                        imgHeight={imgSize.height}
                        viewportWidth={viewportSize.width}
                        viewportHeight={viewportSize.height}
                      />
                    ))
                }

              </div>
            )}
            <input ref={fileRef} type="file" accept=".emd,.tif,.tiff,.jpg,.jpeg,.png,.npy"
              hidden onChange={onFileChange} />
          </main>

          {/* right panel */}

          <StatsPanel
            image={image}
            sessionId={sessionId}
            metadata={metadata}
            stats={seg.stats}
            segDone={seg.segDone}
            groundTruthScore={seg.groundTruthScore as any}
            scaleBarMode={scaleBarMode}
            scaleBarPixels={scaleBarPixels}
            onViewDetails={() => setShowStatsDetail(true)}
            onMetadataUpdate={handleMetadataUpdate}
            onToggleScaleBar={handleToggleScaleBar}
            onScaleBarCancel={handleScaleBarCancel}
          />

        </div>
      </div>
    </>
  );
}
