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
} from "lucide-react";
import { BASE_URL, getModels, uploadImage, segmentImage, uploadGroundTruth} from "@/lib/api";
import { useRouter } from "next/navigation";
import BlackoutCanvas from "./components/BlackOutCanvas";

export default function Workspace({
  params,
}: {
  params: { session_id: string };
}) {
  const router = useRouter();
  const { data: models = [] } = useQuery({
    queryKey: ["models"],
    queryFn: getModels,
  });

  const STAT_MAP: Record<string, string> = {
    "Particles": "particle_count",
    "Avg. Size": "avg_size",
    "Avg. Circularity": "avg_circularity",
    "Coverage": "coverage",
  };

  const [image, setImage] = useState<string | null>(null);
  const [imgSize, setImgSize] = useState({ width: 0, height: 0 });
  const viewportRef = useRef<HTMLDivElement>(null);
  const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 });


  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false);
  const [masksVisible, setMasksVisible] = useState(true);
  const [status, setStatus] = useState<string>("Upload an image to begin.");
  const [stats, setStats] = useState<Record<string, number> | null>(null);
  const [segDone, setSegDone] = useState(false);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const gtFileRef = useRef<HTMLInputElement>(null);
  const [blackoutMode, setBlackoutMode] = useState(false);
  const [blackoutRegions, setBlackoutRegions] = useState<any[]>([]);
  const [groundTruth, setGroundTruth]= useState(false);
  const [groundTruthStatus, setGroundTruthStatus] = useState<string>("Upload ground truth for current image! ");


  useEffect(() => {
    if (models.length > 0) setSelectedModel(models[0]);
  }, [models]);

  useEffect(() => {
    if (!viewportRef.current) return;

    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      console.log("viewportSize updated:", { width, height });
      setViewportSize({ width, height });
    });

    observer.observe(viewportRef.current);
    return () => observer.disconnect();
  }, [image]); // run once on mount, observer handles updates

  useEffect(() => {
    if (imgSize.width && viewportSize.width) {
      console.log("ImgSize:", imgSize);
      console.log("ViewportSize:", viewportSize);
    }
  }, [imgSize, viewportSize]);

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  async function handleFile(file: File) {
    const url = URL.createObjectURL(file);
    setImage(url);
    setStatus("Uploading...");

    const result = await uploadImage(file);
    setSessionId(result.session_id);

    window.history.replaceState(
      null,
      "",
      `/workspace/${result.session_id}`
    );

    setStatus(`Loaded: ${file.name} — ready to segment.`);
  }
  
  async function handleGroundTruth(file: File) {
    setGroundTruthStatus("Uploading ground truth...");
    const res = await uploadGroundTruth(sessionId!, file);

    
    if (res.warnings?.length > 0) {
      setGroundTruthStatus(`Warning: ${res.warnings[0]}`);
    } 
    setGroundTruth(true);

    if (segDone) {
      setGroundTruthStatus("GT uploaded — computing score...");
      // TODO: trigger score computation
    } else {
      setGroundTruthStatus("GT uploaded — run segmentation to compute score.");
    }
  }

  function onGroundTruthFileChange(e: React.ChangeEvent<HTMLInputElement>){
    const file = e.target.files?.[0];
    if (file) handleGroundTruth(file);
    setStatus("Ground Truth File updated! ");
    e.target.value = "";
  }

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    setStatus("File updated! ");
    e.target.value = "";

  }

  async function handleRunSegmentation() {
    if (!sessionId || !selectedModel) {
      console.warn("No session id or model selected.");
      return;
    }

    try {
      setStatus(`Running ${selectedModel}...`);
      const resizedBlackoutRegions = blackoutRegions;
      const result = await segmentImage(sessionId, selectedModel, resizedBlackoutRegions);

      setMaskUrl(
        `${BASE_URL}${result.mask_url}?t=${Date.now()}`
      );
      setStats(result.stats);
      setSegDone(true);
      setStatus("Segmentation complete. Refine masks or export.");
    } catch (err) {
      console.error("Segmentation failed:", err);
      setStatus("Segmentation failed.");
    }
  }
  //
  //we at this point visually black out the selected region  
  //we want to send this to the backend as well, since that 
  //will make sure the segmentation happends on this new state 
  //and not the old one. 
  async function applyBlackout(regions: any[]) {
    console.log("Regions: ", regions );
    setBlackoutRegions(regions);
    setBlackoutMode(false);
    setStatus("Blackout applied — ready to segment.");
  }

  return (
    <div className={styles.workspaceRoot}>
      <header className={styles.topbar}>
        <div className={styles.topbarLeft}>
          <span className={styles.logo}>
            TEM<span className={styles.logoAccent}>seg</span>
          </span>
          <span className={styles.sessionTag}>
            session ·{" "}
            {sessionId
              ? sessionId.slice(0, 8)
              : "upload image to start"}
          </span>
        </div>
        <div className={styles.topbarRight}>
          <span className={styles.statusPill}>{status}</span>
        </div>
      </header>

      <div className={styles.workspaceBody}>
        <aside className={styles.sidebar}>
          <section className={styles.sidebarSection}>
            <p className={styles.sidebarLabel}>Model</p>
            <div className={styles.dropdownWrap}>
              <button
                className={styles.dropdownBtn}
                onClick={() =>
                  setModelDropdownOpen((o) => !o)
                }
              >
                {selectedModel} <ChevronDown size={14} />
             </button>

              {modelDropdownOpen && (
                <ul className={styles.dropdownList}>
                  {models.map((m) => (
                    <li
                      key={m}
                      className={`${styles.dropdownItem} ${
                        m === selectedModel
                          ? styles.dropdownItemActive
                          : ""
                      }`}
                      onClick={() => {
                        setSelectedModel(m);
                        setModelDropdownOpen(false);
                      }}
                    >
                      {m}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>

          <section className={styles.sidebarSection}>
            <p className={styles.sidebarLabel}>Actions</p>

            <button
              className={`${styles.actionBtn} ${styles.actionBtnPrimary}`}
              onClick={handleRunSegmentation}
              disabled={!image}
            >
              <Play size={14} /> Run Segmentation
            </button>

            <button
              className={styles.actionBtn}
              disabled={!segDone}
              onClick={() =>
                setMasksVisible((v) => !v)
              }
            >
              {masksVisible ? (
                <EyeOff size={14} />
              ) : (
                <Eye size={14} />
              )}
              {masksVisible
                ? "Hide Masks"
                : "Show Masks"}
            </button>

            <button
              className={styles.actionBtn}
              disabled={!segDone}
            >
              <Sliders size={14} /> Refine Masks
            </button>

            <button
              className={styles.actionBtn}
              disabled={!segDone}
            >
              <Download size={14} /> Export
            </button>
          </section>

          <section className={styles.sidebarSection}>
            <p className={styles.sidebarLabel}>
              Blackout Regions
            </p>
            <button
              className={styles.actionBtn}
              disabled={!image}
              onClick={() =>
                setBlackoutMode((b) => !b)
              }
            >
              <Trash2 size={14} />{" "}
              {blackoutMode
                ? "Exit Blackout"
                : "Mask Regions"}
            </button>

            {blackoutMode && (
              <button
                className={styles.actionBtn}
                onClick={() =>
                  applyBlackout(blackoutRegions)
                }
              >
                Apply Blackout
              </button>
            )}

            <p className={styles.sidebarHint}>
              Mask out regions to exclude before or
              after segmentation.
            </p>
          </section>

          <section className={styles.sidebarSection}>
            <p className={styles.sidebarLabel}>
              Ground Truth
            </p>
            <button 
            className={styles.actionBtn} onClick={() => gtFileRef.current?.click()}
            disabled={!segDone}
            >
              <Upload size={14} /> Upload GT
            </button>

            <input
              ref={gtFileRef}
              type="file"
              accept=".npy,.png,.tiff,.tif,.json"
              hidden
              onChange={onGroundTruthFileChange}
            />
            <p className={styles.sidebarHint}>
              {groundTruth ?
                "Ground truth is ready! ":
                "Upload a ground truth mask to compute accuracy scores."
              }

            </p>
          </section>
        </aside>

        <main className={styles.canvasArea}>
          {!image ? (
            <div
              className={`${styles.dropZone} ${
                isDragging
                  ? styles.dropZoneDragging
                  : ""
              }`}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() =>
                setIsDragging(false)
              }
              onDrop={onDrop}
              onClick={() =>
                fileRef.current?.click()
              }
            >
              <Upload size={32} strokeWidth={1.5} />
              <p className={styles.dropLabel}>
                Drop image here or click to upload
              </p>
              <p className={styles.dropHint}>
                TIF, JPEG, PNG supported
              </p>
            </div>
          ) : (
              <div className={styles.imageViewport} ref={viewportRef} style={{ position: "relative" }}>
                {image && (
                  <img
                    src={image}
                    alt="TEM input"
                    className={styles.temImage}
                    style={{ display: "block", width: "100%", height: "100%" }}
                    onLoad={(e) =>
                      setImgSize({
                        width: e.currentTarget.naturalWidth,
                        height: e.currentTarget.naturalHeight,
                      })
                    }
                  />
                )}

                {blackoutMode && imgSize.width && imgSize.height && (
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
                  <img
                    src={maskUrl}
                    className={styles.maskOverlay}
                  />
                )}
              </div>
          )}

          <input
            ref={fileRef}
            type="file"
            accept=".tif,.tiff,.jpg,.jpeg,.png"
            hidden
            onChange={onFileChange}
          />
        </main>

        <aside className={styles.statsPanel}>
          <p className={styles.sidebarLabel}>
            Stats
          </p>

          {!segDone ? (
            <p className={styles.sidebarHint}>
              Run segmentation to see particle
              statistics.
            </p>
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
            </div>

          )}
        </aside>

      </div>
    </div>
  );
} 
