"use client";

import styles from "./page.module.css";
import { useQuery } from "@tanstack/react-query";
import { useState, useEffect, useRef, useCallback } from "react";
import {
  Upload, Play, Download, Sliders,
  Eye, EyeOff, Trash2, ChevronDown, AlertTriangle,
} from "lucide-react";
import { BASE_URL, getModels, uploadImage, getInstances, saveInstances } from "@/lib/api";
import BlackoutCanvas from "./components/BlackOutCanvas";
import RefineCanvas from "./components/RefineCanvas";
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
  const liveRegionsRef = useRef<any[]>([]); // tracks region to be excluded
  const liveInverseRegionsRef = useRef<any[]>([]); // tracks regions to be included 

  // refine mode
  const [refineMode, setRefineMode] = useState(false);

  // segmentation hook
  const seg = useSegmentationState({ sessionId, selectedModel });

  // refine hook — instantiated when refine mode is entered
  // imgSize.width guard ensures we don't init with zero dims
  const refine = useRefineState({
    sessionId: sessionId ?? "",
    initialInstances: [],
    onSave: async (updated) => {
      if (!sessionId) return;
      const result = await saveInstances(sessionId, updated);
      seg.setMaskUrl(`${BASE_URL}${result.mask_url}?t=${Date.now()}`);
      setRefineMode(false);
      setStatus("Edited masks saved.");
    },
    onDiscard: () => setRefineMode(false),
    imgWidth: imgSize.width || 1,
    imgHeight: imgSize.height || 1,
  });

  // delete key — only active in refine mode
  useEffect(() => {
    if (!refineMode) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Backspace" || e.key === "Delete") {
        refine.handleDeleteSelected();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [refineMode, refine.handleDeleteSelected]);

  // image upload
  async function handleFile(file: File) {
    setStatus("Uploading...");
    seg.reset();
    setRefineMode(false);
    setZoom(1); // def zoom is the same size as img
    setPan({ x: 0, y: 0 });
    const result = await uploadImage(file);
    setSessionId(result.session_id);
    setImage(`${BASE_URL}${result.preview_url}`);
    window.history.replaceState(null, "", `/workspace/${result.session_id}`);
    setStatus(`Loaded: ${file.name} — ready to segment.`);
  }

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  // run segmentation — reads live region refs, passes values into hook
  async function handleRunSegmentation() {
    if (!sessionId || !selectedModel) return;
    const activeRegions = seg.inverseBlackoutMode
      ? liveInverseRegionsRef.current
      : liveRegionsRef.current;
    const blackout = !seg.inverseBlackoutMode && liveRegionsRef.current.length > 0;
    const inverse = seg.inverseBlackoutMode && liveInverseRegionsRef.current.length > 0;

    setStatus(`Running ${selectedModel}...`);
    const msg = await seg.runSegmentation(activeRegions, blackout, inverse);
    if (msg) {
      setStatus(msg);
      if (seg.groundTruth) {
        await seg.scoreGroundTruth(activeRegions, blackout, inverse);
      }
    } else {
      setStatus("Segmentation failed.");
    }
  }

  // enter refine mode — load instances then activate
  async function enterRefineMode() {
    if (!sessionId) return;
    const res = await getInstances(sessionId);
    // reset refine hook with fresh instances
    // we do this by passing instances to a wrapper that re-inits
    refine.handleDiscard(); // clears state
    // inject fresh instances — handled below via key trick
    setRefineInstances(res.instances);
    setRefineMode(true);
    // setStatus("Refine mode — drag vertices, space+drag to pan.");
  }

  // separate state to hand fresh instances to refine hook on enter
  const [refineInstances, setRefineInstances] = useState<any[]>([]);

  // re-init refine hook when instances change
  // this is the clean way to reset a hook with new data
  const refineKey = refineInstances.length > 0 ? refineInstances[0].id : "empty";

  // blackout apply
  function handleApplyBlackout() {
    const regions = seg.inverseBlackoutMode
      ? liveInverseRegionsRef.current
      : liveRegionsRef.current;
    seg.applyBlackout(regions);
    setStatus("Regions applied — ready to segment.");
  }

  // ground truth upload
  function onGroundTruthFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const activeRegions = seg.inverseBlackoutMode
      ? liveInverseRegionsRef.current
      : liveRegionsRef.current;
    const blackout = !seg.inverseBlackoutMode && liveRegionsRef.current.length > 0;
    const inverse = seg.inverseBlackoutMode && liveInverseRegionsRef.current.length > 0;
    seg.uploadGT(file, activeRegions, blackout, inverse);
    e.target.value = "";
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
    if (seg.blackoutMode || refineMode) return;
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

  const activeRegions = seg.inverseBlackoutMode
    ? seg.inverseBlackoutRegions
    : seg.blackoutRegions;
  const hasRegions = activeRegions.length > 0;

  return (
    <div className={styles.workspaceRoot}>

      {/* topbar */}
      <header className={styles.topbar}>
        <div className={styles.topbarLeft}>
          <div
            onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); window.location.href = "/workspace/new"; }}
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
          <span className={styles.statusPill}>{status}</span>
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

            {/*[ACTION]- show/hide masks */}
            <button className={styles.actionBtn} disabled={!seg.segDone}
              onClick={() => seg.setMasksVisible(v => !v)}>
              {seg.masksVisible ? <EyeOff size={14} /> : <Eye size={14} />}
              {seg.masksVisible ? "Hide Masks" : "Show Masks"}
            </button>

            {/*[ACTION]- refine masks */}
            <button className={styles.actionBtn} disabled={!seg.segDone}
              onClick={refineMode ? refine.handleSave : enterRefineMode}>
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
            <button className={styles.actionBtn} disabled={!seg.segDone}>
              <Download size={14} /> Export
            </button>
          </section>

          {/* split controls — only shown in refine mode */}
          {refineMode && (
            <section className={styles.sidebarSection}>
              <p className={styles.sidebarLabel}>Refine</p>

              {refine.selectedId !== null && !refine.splitMode && (
                <button className={styles.actionBtn} onClick={refine.handleEnterSplit}>
                  Split Instance
                </button>
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
                className={`${styles.actionBtn} ${!seg.inverseBlackoutMode ? styles.actionBtnPrimary : ""}`}
                onClick={() => seg.setInverseBlackoutMode(false)}
                disabled={!image}
              >Exclude</button>
              <button
                className={`${styles.actionBtn} ${seg.inverseBlackoutMode ? styles.actionBtnPrimary : ""}`}
                onClick={() => seg.setInverseBlackoutMode(true)}
                disabled={!image}
              >Isolate</button>
            </div>
            <button className={styles.actionBtn} disabled={!image}
              onClick={() => { seg.setMasksVisible(false); seg.setBlackoutMode(b => !b); }}>
              <Trash2 size={14} /> {seg.blackoutMode ? "Exit" : "Draw Regions"}
            </button>
            {seg.blackoutMode && (
              <button className={styles.actionBtn} onClick={handleApplyBlackout}>
                Apply
              </button>
            )}
            {hasRegions && !seg.blackoutMode && (
              <button className={styles.actionBtn} onClick={() => {
                seg.clearRegions();
                if (seg.inverseBlackoutMode) liveInverseRegionsRef.current = [];
                else liveRegionsRef.current = [];
                setStatus("Regions cleared.");
              }}>
                Clear Regions
              </button>
            )}
            <p className={styles.sidebarHint}>
              {seg.inverseBlackoutMode
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
              <p className={styles.dropHint}>TIF, TIFF, JPEG, PNG, NPY supported</p>
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
                transform: seg.blackoutMode || refineMode
                  ? "none"
                  : `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
                transformOrigin: "center center",
                transition: isPanning.current ? "none" : "transform 0.05s ease-out",
                cursor: seg.blackoutMode || refineMode
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
                  display: "block", width: "100%", height: "100%",
                  visibility: seg.blackoutMode || refineMode ? "hidden" : "visible",
                }}
                onLoad={e => setImgSize({
                  width: e.currentTarget.naturalWidth,
                  height: e.currentTarget.naturalHeight,
                })}
              />

              {/* blackout canvas */}
              {seg.blackoutMode && imgSize.width > 0 && (
                <div style={{ position: "absolute", top: 0, left: 0 }}>
                  <BlackoutCanvas
                    imageSrc={image} 
                    width={viewportSize.width}
                    height={viewportSize.height}
                    imgWidth={imgSize.width}
                    imgHeight={imgSize.height}
                    isInverse={seg.inverseBlackoutMode}
                    initialRegions={seg.inverseBlackoutMode ? seg.inverseBlackoutRegions : seg.blackoutRegions}
                    onChange={regions => {
                      if (seg.inverseBlackoutMode) {
                        liveInverseRegionsRef.current = regions;
                        seg.setInverseBlackoutRegions(regions);
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
                    onSelect={refine.handleSelect}
                    onDeselect={refine.handleDeselect}
                    onVertexDragEnd={refine.handleVertexDragEnd}
                    onVertexDelete={refine.handleVertexDelete}
                    onEdgeClick={refine.handleEdgeClick}
                    onSplitPointPlace={refine.handleSplitPointPlace}
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
            </div>
          )}
          <input ref={fileRef} type="file" accept=".tif,.tiff,.jpg,.jpeg,.png,.npy"
            hidden onChange={onFileChange} />
        </main>

        {/* stats panel */}
        <aside className={styles.statsPanel}>
          <p className={styles.sidebarLabel}>Stats</p>
          {!seg.segDone ? (
            <p className={styles.sidebarHint}>Run segmentation to see particle statistics.</p>
          ) : (
            <div className={styles.statsGrid}>
              {(["Particles", "Avg. Size", "Avg. Circularity", "Coverage"] as const).map(label => {
                const key = ({
                  "Particles": "particle_count",
                  "Avg. Size": "avg_size",
                  "Avg. Circularity": "avg_circularity",
                  "Coverage": "coverage",
                } as const)[label];
                return (
                  <div className={styles.statRow} key={label}>
                    <span className={styles.statLabel}>{label}</span>
                    <span className={styles.statVal}>
                      {seg.stats && key in seg.stats
                        ? (seg.stats as any)[key].toFixed(3)
                        : "—"}
                    </span>
                  </div>
                );
              })}
              <div className={styles.statDivider} />
              {(["IoU", "Dice", "Pixel Acc"] as const).map(label => {
                const key = label === "Pixel Acc" ? "pixel_acc" : label.toLowerCase() as "iou" | "dice";
                return (
                  <div className={styles.statRow} key={label}>
                    <span className={styles.statLabel}>{label}</span>
                    <span className={styles.statVal}>
                      {seg.groundTruthScore?.[key]?.toFixed(3) ?? "—"}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </aside>

      </div>
    </div>
  );
}
