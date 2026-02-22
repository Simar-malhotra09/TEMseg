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
import { BASE_URL, getModels, uploadImage, segmentImage } from "@/lib/api";
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

  const [image, setImage] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false);
  const [masksVisible, setMasksVisible] = useState(true);
  const [status, setStatus] = useState<string>("Upload an image to begin.");
  const [segDone, setSegDone] = useState(false);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [blackoutMode, setBlackoutMode] = useState(false);
  const [blackoutRegions, setBlackoutRegions] = useState<any[]>([]);

  useEffect(() => {
    if (models.length > 0) setSelectedModel(models[0]);
  }, [models]);

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

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }

  async function handleRunSegmentation() {
    if (!sessionId || !selectedModel) {
      console.warn("No session id or model selected.");
      return;
    }

    try {
      setStatus(`Running ${selectedModel}...`);

      const result = await segmentImage(sessionId, selectedModel);

      setMaskUrl(
        `${BASE_URL}${result.mask_url}?t=${Date.now()}`
      );

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
              className={styles.actionBtn}
              onClick={() =>
                fileRef.current?.click()
              }
              disabled={!segDone}
            >
              <Upload size={14} /> Upload GT
            </button>
            <p className={styles.sidebarHint}>
              Upload a ground truth mask to compute
              accuracy scores.
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
            <div className={styles.imageViewport}>
              {blackoutMode ? (
                // Konva element

                // A big problem right now is that 
                // this only works when the image is in 
                // it's org state that is masks aren't overlayed
                // since otherwise this condition translated to 
                // the sencond statement/
                // This means you cannot mask out regions 
                // with overlays as your visual guide which 
                // is obv required. 

                <BlackoutCanvas

                  imageSrc={image}
                  width={800}
                  height={600}
                  onChange={setBlackoutRegions}
                />
              ) : (
                <>
                  <img
                    src={image}
                    alt="TEM input"
                    className={styles.temImage}
                  />
                  {segDone &&
                    masksVisible &&
                    maskUrl && (
                      <img
                        src={maskUrl}
                        className={
                          styles.maskOverlay
                        }
                      />
                    )}
                </>
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
              {[
                ["Particles", "—"],
                ["Avg. Size", "—"],
                ["Avg. Circularity", "—"],
                ["Coverage", "—"],
                ["GT Score", "—"],
              ].map(([label, val]) => (
                <div
                  className={styles.statRow}
                  key={label}
                >
                  <span
                    className={styles.statLabel}
                  >
                    {label}
                  </span>
                  <span
                    className={styles.statVal}
                  >
                    {val}
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
