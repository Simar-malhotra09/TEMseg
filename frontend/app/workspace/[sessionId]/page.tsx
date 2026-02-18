"use client";
import styles from "./page.module.css"
import { useState, useRef } from "react";
import { Upload, Play, Download, Sliders, Eye, EyeOff, Trash2, ChevronDown } from "lucide-react";

const MODELS = ["YoloSAM", "Custom"];

export default function Workspace({ params }: { params: { session_id: string } }) {
  const [image, setImage] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedModel, setSelectedModel] = useState(MODELS[0]);
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false);
  const [masksVisible, setMasksVisible] = useState(true);
  const [status, setStatus] = useState<string>("Upload an image to begin.");
  const [segDone, setSegDone] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  function handleFile(file: File) {
    const url = URL.createObjectURL(file);
    setImage(url);
    setSegDone(false);
    setStatus(`Loaded: ${file.name} — ready to segment.`);
  }


  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }

  function handleRunSegmentation() {
    if (!image) return;
    setStatus(`Running ${selectedModel}...`);
    // TODO: POST /segmentation/{session_id}/run; this makes a http request 
    //       our backend thorugh the FastApi exposed endpoints. 
    //
    // Set timeout until I impl it.
    setTimeout(() => {
      setSegDone(true);
      setStatus("Segmentation complete. Refine masks or export.");
    }, 1800);
  }

  return (
    <div className={styles.workspaceRoot}>
      <header className={styles.topbar}>
        <div className={styles.topbarLeft}>
          <span className={styles.logo}>
            TEM<span className={styles.logoAccent}>seg</span>
          </span>
          <span className={styles.sessionTag}>
            session · {String("1")}
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
                onClick={() => setModelDropdownOpen(o => !o)}
              >
                {selectedModel} <ChevronDown size={14} />
              </button>

              {modelDropdownOpen && (
                <ul className={styles.dropdownList}>
                  {MODELS.map(m => (
                    <li
                      key={m}
                      className={`${styles.dropdownItem} ${
                        m === selectedModel ? styles.dropdownItemActive : ""
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
              onClick={() => setMasksVisible(v => !v)}
            >
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
            <button className={styles.actionBtn} disabled={!image}>
              <Trash2 size={14} /> Mask Regions
            </button>
            <p className={styles.sidebarHint}>
              Mask out regions to exclude before or after segmentation.
            </p>
          </section>

          <section className={styles.sidebarSection}>
            <p className={styles.sidebarLabel}>Ground Truth</p>
            <button
              className={styles.actionBtn}
              onClick={() => fileRef.current?.click()}
              disabled={!segDone}
            >
              <Upload size={14} /> Upload GT
            </button>
            <p className={styles.sidebarHint}>
              Upload a ground truth mask to compute accuracy scores.
            </p>
          </section>
        </aside>

        <main className={styles.canvasArea}>
          {!image ? (
            <div
              className={`${styles.dropZone} ${
                isDragging ? styles.dropZoneDragging : ""
              }`}
              onDragOver={e => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={onDrop}
              onClick={() => fileRef.current?.click()}
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
              <img
                src={image}
                alt="TEM input"
                className={styles.temImage}
              />

              {segDone && masksVisible && (
                <div className={styles.maskOverlayPlaceholder}>
                  <span className={styles.overlayLabel}>
                    masks visible · konva canvas
                  </span>
                </div>
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
          <p className={styles.sidebarLabel}>Stats</p>

          {!segDone ? (
            <p className={styles.sidebarHint}>
              Run segmentation to see particle statistics.
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
                <div className={styles.statRow} key={label}>
                  <span className={styles.statLabel}>{label}</span>
                  <span className={styles.statVal}>{val}</span>
                </div>
              ))}
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
