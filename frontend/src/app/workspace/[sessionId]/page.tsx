"use client";

import styles from "./page.module.css";
import { useQuery } from "@tanstack/react-query";
import { useState, useEffect, useRef } from "react";
import {
  Upload,
  Play,
  Download,
  Sliders,
  Eye,
  EyeOff,
  Trash2,
  ChevronDown,
  AlertTriangle,
} from "lucide-react";
import { BASE_URL, getModels, uploadImage, segmentImage, uploadGroundTruth, computeGTScore, getInstances,saveInstances } from "@/lib/api";
import { useRouter } from "next/navigation";
import BlackoutCanvas from "./components/BlackOutCanvas";
import RefineCanvas from "./components/RefineCanvas.tsx";

export default function Workspace({ params }: { params: { session_id: string } }) {
  const router = useRouter();
  const { data: models = [] } = useQuery({
    queryKey: ["models"],
    queryFn: getModels,
  });

  const STAT_MAP: Record<string, string> = {
    "Particles":        "particle_count",
    "Avg. Size":        "avg_size",
    "Avg. Circularity": "avg_circularity",
    "Coverage":         "coverage",
  };

  const liveRegionsRef = useRef<any[]>([]);
  const liveInverseRegionsRef = useRef<any[]>([]);
  const isPanning = useRef(false);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const panStart = useRef({ x: 0, y: 0 });

  const [panning, setPanning] = useState(false);
  // ── image / session ──────────────────────────────────────────
  const [image,       setImage]      = useState<string | null>(null);
  const [imgSize,     setImgSize]    = useState({ width: 0, height: 0 });
  const [sessionId,   setSessionId]  = useState<string | null>(null);
  const [isDragging,  setIsDragging] = useState(false);
  const [status,      setStatus]     = useState("Upload an image to begin.");

  const fileRef        = useRef<HTMLInputElement>(null);
  const gtFileRef      = useRef<HTMLInputElement>(null);
  const viewportRef    = useRef<HTMLDivElement>(null);
  const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 });

  // ── model ────────────────────────────────────────────────────
  const [selectedModel,      setSelectedModel]      = useState<string | null>(null);
  const [modelDropdownOpen,  setModelDropdownOpen]  = useState(false);

  // ── segmentation ─────────────────────────────────────────────
  const [segDone,      setSegDone]      = useState(false);
  const [maskUrl,      setMaskUrl]      = useState<string | null>(null);
  const [masksVisible, setMasksVisible] = useState(true);
  const [stats,        setStats]        = useState<Record<string, number> | null>(null);

  // ── blackout ─────────────────────────────────────────────────
  const [blackoutMode,      setBlackoutMode]      = useState(false);
  const [blackoutRegions,   setBlackoutRegions]   = useState<any[]>([]); // live, being edited
  const [committedRegions,  setCommittedRegions]  = useState<any[]>([]); // locked to last seg run

  // ── inverse blackout ─────────────────────────────────────────────────
  const [inverseBlackoutMode,      setInverseBlackoutMode]      = useState(false);
  const [inverseBlackoutRegions,   setInverseBlackoutRegions]   = useState<any[]>([]); // live, being edited
  const [inverseCommittedRegions,  setInverseCommittedRegions]  = useState<any[]>([]); // locked to last seg run

  // make sure regions stay in sync
  const regionsOutOfSync = segDone && (
    JSON.stringify(blackoutRegions) !== JSON.stringify(committedRegions) ||
    JSON.stringify(inverseBlackoutRegions) !== JSON.stringify(inverseCommittedRegions)
  );

  // ── ground truth ─────────────────────────────────────────────
  const [groundTruth,       setGroundTruth]       = useState(false);
  const [groundTruthStatus, setGroundTruthStatus] = useState("Upload ground truth for current image!");
  const [groundTruthScore,  setGroundTruthScore]  = useState<{
    iou?: number; dice?: number; pixel_acc?: number;
  } | null>(null);
  const [gtUrl, setGtUrl] = useState<string | null>(null);
  const [gtVisible, setGtVisible] = useState(false);

  // - refine masks 
  const [refineMode, setRefineMode] = useState(false);
  const [instances, setInstances] = useState<Instance[]>([]);

  // ── effects ──────────────────────────────────────────────────
  useEffect(() => {
    if (models.length > 0) setSelectedModel(models[0]);
  }, [models]);

  useEffect(() => {
    if (!viewportRef.current) return;
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setViewportSize({ width, height });
    });
    observer.observe(viewportRef.current);
    return () => observer.disconnect();
  }, [image]);

  // ── handlers ─────────────────────────────────────────────────
  function handleWheel(e: React.WheelEvent) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(z => {
      const next = Math.min(Math.max(z * delta, 0.5), 5);
      if (next <= 1) setPan({ x: 0, y: 0 }); // reset pan when fully zoomed out
      return next;
    });
  }

  function handleMouseDown(e: React.MouseEvent) {
    if (blackoutMode || refineMode) return;
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

    // how much extra size the image gains at current zoom
    const excessW = (viewportSize.width * (zoom - 1)) / 2;
    const excessH = (viewportSize.height * (zoom - 1)) / 2;

    // clamp so image edges can't go past canvas edges
    setPan({
      x: Math.min(excessW, Math.max(-excessW, rawX)),
      y: Math.min(excessH, Math.max(-excessH, rawY)),
    });
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  async function handleFile(file: File) {
    setStatus("Uploading...");
    const result = await uploadImage(file);
    setSessionId(result.session_id);
    setImage(`${BASE_URL}${result.preview_url}`); // use preview_url instead of createObjectURL
    window.history.replaceState(null, "", `/workspace/${result.session_id}`);
    setStatus(`Loaded: ${file.name} — ready to segment.`)
  }

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  }

  async function handleRunSegmentation() {
    if (!sessionId || !selectedModel) return;
    console.log("blackout:", blackoutRegions);
    console.log("inverse blackout:", inverseBlackoutRegions);
    // regions to act on 
    const activeRegions = inverseBlackoutMode 
      ? liveInverseRegionsRef.current 
      : liveRegionsRef.current;

    console.log("regions being sent:", activeRegions);
    
    try {
      setStatus(`Running ${selectedModel}...`);

      const result = await segmentImage(
        sessionId,
        selectedModel,
        activeRegions,
        !inverseBlackoutMode && liveRegionsRef.current.length > 0,
        inverseBlackoutMode && liveInverseRegionsRef.current.length > 0,
      );

      setCommittedRegions(blackoutRegions);
      setInverseCommittedRegions(inverseBlackoutRegions);

      setMaskUrl(`${BASE_URL}${result.mask_url}?t=${Date.now()}`);
      setStats(result.stats);
      setSegDone(true);
      setMasksVisible(true);
      setStatus(`Segmentation completed in ${result.time_elapsed.toFixed(2)}s. Refine masks or export.`);


      if (groundTruth) {
        const scored = await computeGTScore(
          sessionId!, 
          activeRegions, 
          // ensure mutual exclusiveness
          !inverseBlackoutMode && blackoutRegions.length > 0,   // blackout
          inverseBlackoutMode && inverseBlackoutRegions.length > 0, // inverse
        );

        setGroundTruthScore(scored.scores);
        setGroundTruthStatus("GT score computed.");
        setGtUrl(`${BASE_URL}/gt/${sessionId}/preview?t=${Date.now()}`);
        console.log(`GT preview url: ${BASE_URL}/gt/${sessionId}/preview`)
      }
    } catch (err) {
      console.error("Segmentation failed:", err);
      setStatus("Segmentation failed.");
    }
 }

  async function applyBlackout() {
    const regions = inverseBlackoutMode 
      ? liveInverseRegionsRef.current 
      : liveRegionsRef.current;
    console.log("applyBlackout called, regions:", regions.length, "inverseMode:", inverseBlackoutMode);
    if (inverseBlackoutMode) {
      setInverseBlackoutRegions(regions);
    } else {
      setBlackoutRegions(regions);
    }
    setBlackoutMode(false);
    setStatus("Regions applied — ready to segment.");
  }

  async function handleGroundTruth(file: File) {
    setGroundTruthStatus("Uploading ground truth...");
    const res = await uploadGroundTruth(sessionId!, file);
    setGroundTruth(true);

    if (res.warnings?.length > 0) setGroundTruthStatus(`Warning: ${res.warnings[0]}`);

    if (segDone) {
      const activeCommitted = inverseBlackoutMode
        ? liveInverseRegionsRef.current
        : liveRegionsRef.current;

      const scored = await computeGTScore(
        sessionId!,
        activeCommitted,
        !inverseBlackoutMode && liveRegionsRef.current.length > 0,
        inverseBlackoutMode && liveInverseRegionsRef.current.length > 0,
      );
      setGroundTruthScore(scored.scores);
      setGroundTruthStatus("GT score computed.");
      setGtUrl(`${BASE_URL}/gt/${sessionId}/preview`);
    } else {
      setGroundTruthStatus("GT uploaded — run segmentation to compute score.");
    }
  }

  function onGroundTruthFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleGroundTruth(file);
    e.target.value = "";
  }

  async function enterRefineMode() {
    const res = await getInstances(sessionId!);
    setInstances(res.instances);
    setRefineMode(true);
  }

  async function saveRefinements() {
    const result = await saveInstances(sessionId!, instances);
    setMaskUrl(`${BASE_URL}${result.mask_url}?t=${Date.now()}`);
    setRefineMode(false);
  }
  // ── render ───────────────────────────────────────────────────
  return (
    <div className={styles.workspaceRoot}>

      {/* Top Bar */}
      <header className={styles.topbar}>
        <div className={styles.topbarLeft}>
        <div 
          onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); window.location.href = '/workspace/new'; }}
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
          {regionsOutOfSync && (
            <span className={styles.warnPill}>
              <AlertTriangle size={11} /> blackout regions changed — re-run seg
            </span>
          )}
          <span className={styles.statusPill}>{status}</span>

          {(zoom !== 1 || pan.x !== 0 || pan.y !== 0) && (
            <button className={styles.zoomReset} 
              onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}>
              {Math.round(zoom * 100)}% ✕
            </button>
          )}
        </div>
      </header>

      <div className={styles.workspaceBody}>

        {/* Left Sidebar */}
        <aside className={styles.sidebar}>

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

          <section className={styles.sidebarSection}>
            <p className={styles.sidebarLabel}>Actions</p>
            <button className={`${styles.actionBtn} ${styles.actionBtnPrimary}`}
              onClick={handleRunSegmentation} disabled={!image}>
              <Play size={14} /> Run Segmentation
            </button>
            <button className={styles.actionBtn} disabled={!segDone}
              onClick={() => setMasksVisible(v => !v)}>
              {masksVisible ? <EyeOff size={14} /> : <Eye size={14} />}
              {masksVisible ? "Hide Masks" : "Show Masks"}
            </button>
            <button className={styles.actionBtn} disabled={!segDone}
              onClick={
                refineMode
                  ? () => {
                      saveRefinements();
                      setMasksVisible(b => !b);
                      setStatus("Edited masks saved!")

                    }
                  : () => {
                      setMasksVisible(b => !b);
                      enterRefineMode();
                      setStatus("You can refine the seg mask by moving the verticles on the polygon contour. \n Use space + mouse to pan around!")
                    }
                  
                }>
              <Sliders size={14} /> {refineMode ? "Save Refinements" : "Refine Masks"}
            </button>
            {refineMode && (
              <button className={styles.actionBtn} onClick={() => setRefineMode(false)}>
                Cancel
              </button>
            )}
            <button className={styles.actionBtn} disabled={!segDone}>
              <Download size={14} /> Export
            </button>
          </section>

          <section className={styles.sidebarSection}>
            <p className={styles.sidebarLabel}>Blackout Regions</p>
            
            {/* mode toggle */}
            <div style={{ display: "flex", gap: 4, marginBottom: 6 }}>
              <button className={`${styles.actionBtn} ${!inverseBlackoutMode ? styles.actionBtnPrimary : ""}`}
                onClick={() =>{
                  setInverseBlackoutMode(false);}}
                  disabled={!image}>
                Exclude
              </button>
              <button className={`${styles.actionBtn} ${inverseBlackoutMode ? styles.actionBtnPrimary : ""}`}
                onClick={() => {
                  setInverseBlackoutMode(true);}}
                  disabled={!image}>
                Isolate
              </button>
            </div>

            <button className={styles.actionBtn} disabled={!image}
              onClick={() => {
                setMasksVisible(false);
                setBlackoutMode(b => !b);}
              }>
              <Trash2 size={14} /> {blackoutMode ? "Exit" : "Draw Regions"}
            </button>

            {blackoutMode && (
              <button className={styles.actionBtn} onClick={() => applyBlackout()}>
                Apply
              </button>
            )}

            {/* Only clear for the current mode*/}
            {((inverseBlackoutMode ? inverseBlackoutRegions : blackoutRegions).length > 0) && !blackoutMode && (
              <button className={styles.actionBtn} onClick={() => {
                if (inverseBlackoutMode) {
                  setInverseBlackoutRegions([]);
                  setInverseCommittedRegions([]);
                  liveInverseRegionsRef.current = [];
                } else {
                    setBlackoutRegions([]);
                    setCommittedRegions([]);
                    liveRegionsRef.current = [];
                }
              setStatus(
                        `Cleared regions for ${
                          inverseBlackoutMode ? "inverse mode" : "blackout mode"
                        }!`
                      );
                }
              }
              >

                Clear Regions 
                (for current mode)
              </button>
            )}
            <p className={styles.sidebarHint}>
              {inverseBlackoutMode 
                ? "Isolate: model only sees selected regions."
                : "Exclude: model ignores selected regions."}
            </p>
          </section>
          
          
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
              {groundTruth ? groundTruthStatus : "Upload a ground truth mask to compute accuracy scores."}
            </p>
          </section>

          <section className={styles.sidebarSection}>

            {gtUrl && (
              <>
                <button className={styles.actionBtn} onClick={() => setGtVisible(v => !v)}>

                  {gtVisible ? <EyeOff size={14} /> : <Eye size={14} />}
                  {gtVisible ? "Hide GT" : "Show GT"}
                </button>
                <p className={styles.sidebarHint}>
                  {gtVisible? "Ground truth overlayed! " :"Overlay ground truth to verify it's accuracy"}
                </p>
              </>
            )}

          </section>

        </aside>

        {/* Canvas */}
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
              <p className={styles.dropHint}>TIF, JPEG, PNG supported</p>
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
                transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
                transformOrigin: "center center",
                transition: isPanning.current ? "none" : "transform 0.05s ease-out",
                cursor: blackoutMode || refineMode ? "default" : panning ? "grabbing" : "grab",
              }}
            >
              <img src={image} alt="TEM input" className={styles.temImage}
                style={{ display: "block", width: "100%", height: "100%" }}
                onLoad={e => setImgSize({
                  width: e.currentTarget.naturalWidth,
                  height: e.currentTarget.naturalHeight,
                })}
              />
            {blackoutMode && imgSize.width > 0 && (
              <div style={{ position: "absolute", top: 0, left: 0 }}>
                <BlackoutCanvas
                  imageSrc={image}
                  width={viewportSize.width}
                  height={viewportSize.height}
                  imgWidth={imgSize.width}
                  imgHeight={imgSize.height}
                  isInverse={inverseBlackoutMode}
                  initialRegions={inverseBlackoutMode ? inverseBlackoutRegions : blackoutRegions}
                  onChange={(regions) => {
                    if (inverseBlackoutMode) {
                      liveInverseRegionsRef.current = regions;
                      setInverseBlackoutRegions(regions);
                    } else {
                      liveRegionsRef.current = regions;
                      setBlackoutRegions(regions);
                    }
                  }}
                />
              </div>
            )}
            {refineMode && imgSize.width > 0 && (
              <div style={{ position: "absolute", top: 0, left: 0 }}>
                <RefineCanvas
                  imageSrc={image!}
                  width={viewportSize.width}
                  height={viewportSize.height}
                  imgWidth={imgSize.width}
                  imgHeight={imgSize.height}
                  instances={instances}
                  onChange={setInstances}
                />
              </div>
            )}
            {segDone && masksVisible && maskUrl && (
              <img src={maskUrl} style={{
                position: "absolute", top: 0, left: 0,
                width: viewportSize.width, height: viewportSize.height,
                opacity: 0.5, mixBlendMode: "screen",
              }} />
            )}
            
            {gtVisible && gtUrl && (
              <img src={gtUrl} style={{
                position: "absolute", top: 0, left: 0,
                width: viewportSize.width, height: viewportSize.height,
                opacity: 0.5, mixBlendMode: "screen",
                filter: "hue-rotate(200deg)",
              }} />
            )}

            </div>
          )}
          <input ref={fileRef} type="file" accept=".tif,.tiff,.jpg,.jpeg,.png, .npy"
            hidden onChange={onFileChange} />
        </main>

        {/* Stats Panel */}
        <aside className={styles.statsPanel}>
          <p className={styles.sidebarLabel}>Stats</p>
          {!segDone ? (
            <p className={styles.sidebarHint}>Run segmentation to see particle statistics.</p>
          ) : (
            <div className={styles.statsGrid}>
              {Object.entries(STAT_MAP).map(([label, key]) => (
                <div className={styles.statRow} key={label}>
                  <span className={styles.statLabel}>{label}</span>
                  <span className={styles.statVal}>
                    {stats && key in stats ? stats[key].toFixed(3) : "—"}
                  </span>
                </div>
              ))}
              <div className={styles.statDivider} />
              {(["IoU", "Dice", "Pixel Acc"] as const).map((label) => {
                const key = label === "Pixel Acc" ? "pixel_acc" : label.toLowerCase() as "iou" | "dice";
                return (
                  <div className={styles.statRow} key={label}>
                    <span className={styles.statLabel}>{label}</span>
                    <span className={styles.statVal}>
                      {groundTruthScore?.[key]?.toFixed(3) ?? "—"}
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
