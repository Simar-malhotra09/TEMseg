"use client";
import styles from "./page.module.css";
import { useQuery } from "@tanstack/react-query";
import { useState, useEffect, useRef } from "react";
// import {Image} from "next/Image"
import {
  Upload, Play, Sliders,
  Eye, EyeOff, Trash2, ChevronDown, AlertTriangle,
} from "lucide-react";

import { BASE_URL, Instance, getModels, uploadImage, getInstances, saveInstances, getSessionMetadata, getStats, fromPoints, fromBoxes, proposeSimilar } from "@/lib/api";
import { MousePointerClick, Sparkles, PenTool, BoxSelect } from "lucide-react";

import { BlackoutRect }  from "./components/BlackOutCanvas";
import  BlackoutCanvas  from "./components/BlackOutCanvas";
import RefineCanvas from "./components/RefineCanvas";
import AnnotateCanvas from "./components/AnnotateCanvas";
import BoxAnnotateCanvas from "./components/BoxAnnotateCanvas";
import ExportPanel from "./components/ExportPanel";
import StatsPanel from "./components/StatsPanel";
import StatsDetailView from "./components/StatsDetailView";
import ParticleHighlight from "./components/ParticleHighlight";

import { useSegmentationState } from "./hooks/useSegmentationState";
import { useRefineState } from "./hooks/useRefineState";

interface ImgSize{
  width: number;
  height: number;
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

  // metadata
  const [metadata, setMetadata] = useState<Record<string, any> | null>(null);

  // stats detail view
  const [showStatsDetail, setShowStatsDetail] = useState(false);

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

      // Rehydrate seg state from disk in parallel — both calls return null on 404
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

  // blackout region refs — written by BlackoutCanvas.onChange, read on seg/gt
  // exclusion and inclusion operations can only be perf one at a time
  const liveRegionsRef = useRef<BlackoutRect[]>([]); // tracks region to be excluded
  const liveInverseRegionsRef = useRef<BlackoutRect[]>([]); // tracks regions to be included 

  // refine mode
  const [refineMode, setRefineMode] = useState(false);
  const [refineDone, setRefineDone] = useState(false);

  // bootstrap mode — click points on the image to propose particles via SAM.
  // Each click adds to `pendingProposals` (rendered as a yellow overlay).
  // Accept commits the whole batch via the existing PUT instances endpoint;
  // Discard or per-proposal click-to-reject clears them without a server call.
  const [bootstrapMode, setBootstrapMode] = useState(false);
  const [bootstrapBusy, setBootstrapBusy] = useState(false);
  const [pendingProposals, setPendingProposals] = useState<Instance[]>([]);

  // manual annotation mode — for the zero-detection case where SAM-with-point
  // also can't help (small/clumped/OOD particles). User draws polygons from
  // scratch; each closed polygon becomes a pendingProposal and follows the
  // same Accept/Discard commit flow.
  const [annotateMode, setAnnotateMode] = useState(false);

  // box-prompt annotation mode — drag a tight rectangle around a particle,
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
    const result = await uploadImage(file);
    setSessionId(result.session_id);
    setImage(`${BASE_URL}${result.preview_url}`);
    if (result.image_info) setMetadata(result.image_info);
    // Use query string so /workspace stays a real Next.js route across refreshes.
    window.history.replaceState(null, "", `/workspace?session=${result.session_id}`);
    setStatus(`Loaded: ${file.name} — ready to segment.`);
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
  // pending IDs to live above the on-disk max (proposals from /from-points
  // and /propose-similar are both numbered server-side without knowing about
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
      const maxExistingId = existing.reduce((m, p) => Math.max(m, p.id), 0);
      const renumbered = pendingProposals.map((p, i) => ({
        ...p,
        id: maxExistingId + 1 + i,
      }));
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
  // value works — using a large negative number to make manual entries obvious
  // in logs/debug overlays alongside server-issued positive IDs.
  function handlePolygonComplete(contour: [number, number][]) {
    if (contour.length < 3) return;
    const xs = contour.map(p => p[0]);
    const ys = contour.map(p => p[1]);
    const minX = Math.min(...xs);
    const minY = Math.min(...ys);
    const maxX = Math.max(...xs);
    const maxY = Math.max(...ys);
    // shoelace formula — true polygon area
    let s = 0;
    for (let i = 0; i < contour.length; i++) {
      const [x1, y1] = contour[i];
      const [x2, y2] = contour[(i + 1) % contour.length];
      s += x1 * y2 - x2 * y1;
    }
    const area = Math.round(Math.abs(s) / 2);
    // Placeholder ID — keep stepping down so rejecting a middle polygon
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

  // Call /propose-similar — uses existing on-disk instances as the prior,
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
        `Found ${res.proposals.length} candidate(s) from ${res.seed_count ?? "?"} seed(s) — review and Accept.`,
      );
    } catch (err) {
      console.error("propose similar failed:", err);
      setStatus(`Find Similar failed: ${(err as Error).message}`);
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


  // css zoom/pan handlers — disabled when blackout or refine active
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
    if (seg.isBlackoutMode || refineMode || bootstrapMode || annotateMode || boxMode) return;
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
    let instances = loadedInstances;
    if (instances.length === 0) {
      const res = await getInstances(sessionId); // this should always be up to date
      instances = res.instances;
      setLoadedInstances(instances);
    }

    setHighlightParticleIdx(particleIndex);
    setShowStatsDetail(false); // pan back to workspace
    seg.setMasksVisible(true);
    setStatus(`Highlighting particle id: ${particleIndex+1}`);
  }

  async function handleLocateShape(shape:string) {

    if(!sessionId) return;

    setHighlightParticleIdx(null);  // clear particle id highlight
    let instances= loadedInstances;
    if (instances.length === 0) {
      const res = await getInstances(sessionId); // this should always be up to date
      instances = res.instances;
      setLoadedInstances(instances);
    }

    setHighlightShape(shape);
    setShowStatsDetail(false); // pan back to workspace
    seg.setMasksVisible(true);
    setStatus(`Highlighting all particles of shape: ${shape}`);
  }

  // track which regions are to be used
  const activeRegions = seg.isInvBlackoutMode
    ? seg.invBlackoutRegions
    : seg.blackoutRegions;
  const hasRegions = activeRegions.length > 0;

  return (
    <>
      {/*If in stats dashboard hide the workspace*/}
      {showStatsDetail && seg.stats && (
        <StatsDetailView
          stats={seg.stats}
          metadata={metadata}
          groundTruthScore={seg.groundTruthScore as any}
          onBack={() => setShowStatsDetail(false)}
          onLocateParticle={handleLocateParticle}
          onLocateShape={handleLocateShape}
        />
      )}

      {/* Main workspace */}
      <div className={styles.workspaceRoot} style={{display: showStatsDetail? "none" : "flex" }}>

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
            <span className={styles.statusPill} key={status}>{status}</span>
            {(zoom !== 1 || pan.x !== 0 || pan.y !== 0) && (
              <button className={styles.zoomReset} onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}>
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
                <button className={styles.dropdownBtn} onClick={() => setModelDropdownOpen(o => !o)}>
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

            {/* actions */}
            <section className={styles.sidebarSection}>
              <p className={styles.sidebarLabel}>Actions</p>

              {/*[ACTION]- run seg */}
              <button
                className={`${styles.actionBtn} ${styles.actionBtnPrimary}`}
                onClick={handleRunSegmentation}
                disabled={!image || seg.isSegmenting}
              >
                <Play size={14} /> {seg.isSegmenting ? "Running..." : "Run Segmentation"}
              </button>
            </section>
            <section>
              {/*[ACTION]- show/hide masks */}
              <button className={styles.actionBtn} disabled={!seg.segDone}
                onClick={() => seg.setMasksVisible(v => !v)}>
                {seg.masksVisible ? <EyeOff size={14} /> : <Eye size={14} />}
                {seg.masksVisible ? "Hide Masks" : "Show Masks"}
              </button>

              {/*[ACTION]- refine masks */}
              <button className={styles.actionBtn} disabled={!seg.segDone}
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
                <button className={styles.actionBtn} onClick={refine.handleDiscard}>
                  Discard
                </button>
              )}

              {/* bootstrap mode — click on the image to *propose* particles via
                  SAM point-prompt. Requires a prior /segment run so the SAM
                  image embedding is cached server-side (the backend 400s
                  otherwise, but the disabled state is the clean UX). */}
              <button
                className={styles.actionBtn}
                disabled={!sessionId || refineMode || annotateMode || boxMode || !seg.segDone}
                onClick={() => setBootstrapMode(b => !b)}
                title={!seg.segDone ? "Run segmentation first to cache the SAM embedding" : undefined}
              >
                <MousePointerClick size={14} />
                {bootstrapMode ? "Stop Clicking" : "Click to Add Particles"}
              </button>

              {/* box-prompt annotation — drag a tight box, SAM segments inside.
                  Preferred over click-to-add for OOD particles where point
                  prompts bleed; preferred over polygon for speed. */}
              <button
                className={styles.actionBtn}
                disabled={!sessionId || refineMode || annotateMode || bootstrapMode || !seg.segDone}
                onClick={() => setBoxMode(m => !m)}
                title={!seg.segDone ? "Run segmentation first to cache the SAM embedding" : undefined}
              >
                <BoxSelect size={14} />
                {boxMode ? "Stop Boxing" : "Box to Add Particles"}
              </button>
              {boxMode && (
                <p className={styles.sidebarHint}>
                  Drag a tight rectangle around a particle · Space+drag pans · wheel zooms · Esc cancels.
                </p>
              )}

              {/* manual annotation — fallback when both YOLO and SAM-with-point
                  fail (small / clumped / OOD particles). Doesn't require a
                  prior /segment run since it doesn't use any model output. */}
              <button
                className={styles.actionBtn}
                disabled={!sessionId || refineMode || bootstrapMode || boxMode}
                onClick={() => setAnnotateMode(m => !m)}
              >
                <PenTool size={14} />
                {annotateMode ? "Stop Annotating" : "Annotate Manually"}
              </button>
              {annotateMode && (
                <p className={styles.sidebarHint}>
                  Click to place vertices · Enter or double-click to close · Backspace undoes · Esc cancels.
                </p>
              )}

              {/* propose-similar: build a SAM-embedding prior from existing
                  annotations and find more candidates across the image. Needs
                  at least one annotated particle to construct the prior. */}
              <button
                className={styles.actionBtn}
                disabled={
                  !sessionId ||
                  refineMode ||
                  annotateMode ||
                  boxMode ||
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

              {pendingProposals.length > 0 && (
                <>
                  <button
                    className={`${styles.actionBtn} ${styles.actionBtnPrimary}`}
                    onClick={handleAcceptProposals}
                    disabled={bootstrapBusy}
                  >
                    Accept {pendingProposals.length} Proposal{pendingProposals.length === 1 ? "" : "s"}
                  </button>
                  <button
                    className={styles.actionBtn}
                    onClick={handleDiscardProposals}
                    disabled={bootstrapBusy}
                  >
                    Discard
                  </button>
                  <p className={styles.sidebarHint}>
                    Click a proposal on the canvas to reject it individually.
                  </p>
                </>
              )}
          </section>
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

            {/* refine controls — only shown in refine mode */}
            {refineMode && (
              <section className={styles.sidebarSection}>
                <p className={styles.sidebarLabel}>Refine</p>

                <div style={{ marginBottom: 8 }}>
                  <p className={styles.sidebarHint}>Fill opacity {(refine.polygonOpacity * 100).toFixed(0)}%</p>
                  <input
                    type="range"
                    min={5}
                    max={100}
                    value={Math.round(refine.polygonOpacity * 100)}
                    onChange={e => refine.setPolygonOpacity(Number(e.target.value) / 100)}
                    style={{ width: "100%", accentColor: "#7ee8a2" }}
                  />
                </div>

                {refine.selectedId !== null && !refine.splitMode && !refine.pasteMode && (
                  <>
                    <button className={styles.actionBtn} onClick={refine.handleEnterSplit}>
                      Split Instance
                    </button>
                    <button className={styles.actionBtn} onClick={refine.handleCopy}>
                      Copy (⌘C)
                    </button>
                  </>
                )}

                {refine.clipboard && !refine.pasteMode && !refine.splitMode && (
                  <button className={styles.actionBtn} onClick={refine.handleEnterPaste}>
                    Paste (⌘V)
                  </button>
                )}

                {refine.pasteMode && (
                  <>
                    <p className={styles.sidebarHint}>
                      Click on image to place copied polygon
                    </p>
                    <button className={styles.actionBtn} onClick={refine.handleCancelPaste}>
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
                    <button
                      className={`${styles.actionBtn} ${styles.actionBtnPrimary}`}
                      disabled={refine.splitPoints.length < 2}
                      onClick={refine.handleConfirmSplit}
                    >
                      Confirm Split
                    </button>
                    <button className={styles.actionBtn} onClick={refine.handleCancelSplit}>
                      Cancel Split
                    </button>
                  </>
                )}
              </section>
            )}

            {/* [ACTION] blackout regions */}
            <section className={styles.sidebarSection}>
              <p className={styles.sidebarLabel}>Blackout Regions</p>
              <div style={{ display: "flex", gap: 4, marginBottom: 6 }}>
                <button
                  className={`${styles.actionBtn} ${!seg.isInvBlackoutMode ? styles.actionBtnPrimary : ""}`}
                  onClick={() => seg.setIsInvBlackoutMode(false)}
                  disabled={!image}
                >Exclude</button>
                <button
                  className={`${styles.actionBtn} ${seg.isInvBlackoutMode ? styles.actionBtnPrimary : ""}`}
                  onClick={() => seg.setIsInvBlackoutMode(true)}
                  disabled={!image}
                >Isolate</button>
              </div>
              <button className={styles.actionBtn} disabled={!image}
                onClick={() => { seg.setMasksVisible(false); seg.setIsBlackoutMode(b => !b); }}>
                <Trash2 size={14} /> {seg.isBlackoutMode ? "Exit" : "Draw Regions"}
              </button>
              {seg.isBlackoutMode && (
                <button className={styles.actionBtn} onClick={handleApplyBlackout}>
                  Apply
                </button>
              )}
              {hasRegions && !seg.isBlackoutMode && (
                <button className={styles.actionBtn} onClick={() => {
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
              <button className={styles.actionBtn}
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
                <button className={styles.actionBtn} onClick={() => seg.setGtVisible(v => !v)}>
                  {seg.gtVisible ? <EyeOff size={14} /> : <Eye size={14} />}
                  {seg.gtVisible ? "Hide GT" : "Show GT"}
                </button>
              )}
            </section>

          </aside>

          {/* canvas */}
          <main className={styles.canvasArea}>
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
                  transform: seg.isBlackoutMode || refineMode || bootstrapMode || annotateMode || boxMode
                    ? "none"
                    : `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
                  transformOrigin: "center center",
                  transition: isPanning.current ? "none" : "transform 0.05s ease-out",
                  cursor: seg.isBlackoutMode || refineMode || bootstrapMode || annotateMode || boxMode
                    ? "default"
                    : panning ? "grabbing" : "grab",
                }}
              >
                {/* base image — hidden when blackout/refine canvas is active (they render their own) */}
                <img
                  src={image}
                  alt="TEM input"
                  className={styles.temImage}
                  style={{
                    display: "block",
                    visibility: seg.isBlackoutMode || refineMode || annotateMode || boxMode ? "hidden" : "visible",
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
                    opacity: 0.5, mixBlendMode: "screen",
                    pointerEvents: "none",
                  }} />
                )}

                {/* manual annotation canvas — SVG overlay, in image-space */}
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

                {/* box-prompt annotation canvas — drag rectangles, SAM segments */}
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

                {/* bootstrap click capture — sits on top, transparent, catches clicks */}
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

                {/* pending proposals overlay — yellow outlines, click to reject.
                    Sits above the bootstrap click-capture so reject clicks
                    aren't swallowed as new proposals. Suppressed while a
                    canvas mode is active — those canvases render their own
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
            onViewDetails={() => setShowStatsDetail(true)}
          />

        </div>
      </div>
    </>
  );
}
