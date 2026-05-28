"use client";
import styles from "./page.module.css";
import { useQuery } from "@tanstack/react-query";
import { useState, useEffect, useRef } from "react";
// import {Image} from "next/Image"
import {
  Upload, Play, Sliders,
  Eye, EyeOff, Trash2, ChevronDown, AlertTriangle,
} from "lucide-react";

import { BASE_URL, Instance, getModels, uploadImage, getInstances, saveInstances,Metadata } from "@/lib/api";

import { BlackoutRect }  from "./components/BlackOutCanvas";
import  BlackoutCanvas  from "./components/BlackOutCanvas";
import RefineCanvas from "./components/RefineCanvas";
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
  const [metadata, setMetadata] = useState<Metadata | null>(null);

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
    window.history.replaceState(null, "", `/workspace/${result.session_id}`);
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
    if (seg.isBlackoutMode || refineMode) return;
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
            <span className={styles.statusPill} key={status}>{status}</span>
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

            {/* actions */}
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

              {/*[ACTION]- refine masks */}
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
                  transform: seg.isBlackoutMode || refineMode
                    ? "none"
                    : `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
                  transformOrigin: "center center",
                  transition: isPanning.current ? "none" : "transform 0.05s ease-out",
                  cursor: seg.isBlackoutMode || refineMode
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
                    visibility: seg.isBlackoutMode || refineMode ? "hidden" : "visible",
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
