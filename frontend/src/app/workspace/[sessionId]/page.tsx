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
import { BASE_URL, getModels, uploadImage, segmentImage, uploadGroundTruth, computeGTScore} from "@/lib/api";
import { useRouter } from "next/navigation";
import BlackoutCanvas from "./components/BlackOutCanvas";

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
  const regionsOutOfSync = segDone &&
    JSON.stringify(blackoutRegions) !== JSON.stringify(committedRegions);

  // ── ground truth ─────────────────────────────────────────────
  const [groundTruth,       setGroundTruth]       = useState(false);
  const [groundTruthStatus, setGroundTruthStatus] = useState("Upload ground truth for current image!");
  const [groundTruthScore,  setGroundTruthScore]  = useState<{
    iou?: number; dice?: number; pixel_acc?: number;
  } | null>(null);

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
    try {
      setStatus(`Running ${selectedModel}...`);
      const result = await segmentImage(sessionId, selectedModel, blackoutRegions);
      setMaskUrl(`${BASE_URL}${result.mask_url}?t=${Date.now()}`);
      setStats(result.stats);
      setCommittedRegions(blackoutRegions); // lock regions to this seg result
      setSegDone(true);
      setMasksVisible(true);
      setStatus("Segmentation complete. Refine masks or export.");


      if (groundTruth) {
        const scored = await computeGTScore(sessionId!, blackoutRegions); 
        setGroundTruthScore(scored.scores);
        setGroundTruthStatus("GT score computed.");
      }
    } catch (err) {
      console.error("Segmentation failed:", err);
      setStatus("Segmentation failed.");
    }
  }

  async function applyBlackout(regions: any[]) {
    setBlackoutRegions(regions);
    setBlackoutMode(false);
    setStatus(
      regionsOutOfSync
        ? "Blackout regions modified — re-run segmentation to update mask."
        : "Blackout applied — ready to segment."
    );
  }

  async function handleGroundTruth(file: File) {
    setGroundTruthStatus("Uploading ground truth...");
    const res = await uploadGroundTruth(sessionId!, file);
    setGroundTruth(true);

    if (res.warnings?.length > 0) setGroundTruthStatus(`Warning: ${res.warnings[0]}`);

    if (segDone) {
      // seg already done — compute immediately with committed regions
      const scored = await computeGTScore(sessionId!, committedRegions);
      setGroundTruthScore(scored.scores);
      setGroundTruthStatus("GT score computed.");
    } else {
      setGroundTruthStatus("GT uploaded — run segmentation to compute score.");
    }
  }

  function onGroundTruthFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleGroundTruth(file);
    e.target.value = "";
  }

  // ── render ───────────────────────────────────────────────────
  return (
    <div className={styles.workspaceRoot}>

      {/* Top Bar */}
      <header className={styles.topbar}>
        <div className={styles.topbarLeft}>
          <span className={styles.logo}>TEM<span className={styles.logoAccent}>seg</span></span>
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
            <button className={styles.actionBtn} disabled={!segDone}>
              <Sliders size={14} /> Refine Masks
            </button>
            <button className={styles.actionBtn} disabled={!segDone}>
              <Download size={14} /> Export
            </button>
          </section>

          <section className={styles.sidebarSection}>
            <p className={styles.sidebarLabel}>Blackout Regions</p>
            <button className={styles.actionBtn} disabled={!image}
              onClick={() => setBlackoutMode(b => !b)}>
              <Trash2 size={14} /> {blackoutMode ? "Exit Blackout" : "Mask Regions"}
            </button>
            {blackoutMode && (
              <button className={styles.actionBtn}
                onClick={() => applyBlackout(blackoutRegions)}>
                Apply Blackout
              </button>
            )}
            {blackoutRegions.length > 0 && !blackoutMode && (
              <button className={styles.actionBtn}
                onClick={() => { setBlackoutRegions([]); setCommittedRegions([]); }}>
                Clear All Regions
              </button>
            )}
            <p className={styles.sidebarHint}>
              Mask out regions to exclude before or after segmentation.
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
            <div className={styles.imageViewport} ref={viewportRef} style={{ position: "relative" }}>
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
                    onChange={setBlackoutRegions}
                  />
                </div>
              )}
              {segDone && masksVisible && maskUrl && (
                <img src={maskUrl} className={styles.maskOverlay} />
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
