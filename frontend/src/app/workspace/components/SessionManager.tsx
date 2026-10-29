"use client";

import { useMemo, useState } from "react";
import { Copy, Trash2, X } from "lucide-react";
import styles from "./SessionManager.module.css";
import type { SessionStorageInfo } from "@/lib/api";

interface Props {
  storage: SessionStorageInfo;
  onClose: () => void;
  onDelete: (sessionId: string) => void;
  onCopySessionPath: (storagePath: string) => void;
}

type SortMode = "size" | "last-used";

function sessionFolderPath(storageDir: string, sessionId: string, reportedPath: string | undefined): string {
  if (reportedPath) return reportedPath;
  const separator = storageDir.includes("\\") ? "\\" : "/";
  return `${storageDir.replace(/[\\/]+$/, "")}${separator}${sessionId}`;
}

function formatStorageBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
}

function formatLastUsed(timestamp: number | null): string {
  if (timestamp === null) return "Not recorded";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(timestamp * 1000));
}

export default function SessionManager({ storage, onClose, onDelete, onCopySessionPath }: Props) {
  const [sortMode, setSortMode] = useState<SortMode>("size");
  const sessions = useMemo(() => [...storage.sessions].sort((left, right) => {
    if (sortMode === "size") return right.size_bytes - left.size_bytes;
    return (right.last_used_at ?? 0) - (left.last_used_at ?? 0);
  }), [sortMode, storage.sessions]);

  return (
    <div className={styles.backdrop} onClick={onClose}>
      <section
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby="session-manager-title"
        onClick={event => event.stopPropagation()}
      >
        <header className={styles.header}>
          <div>
            <h2 id="session-manager-title">Manage sessions</h2>
            <p>Delete saved session data to free space on this device.</p>
          </div>
          <button type="button" className={styles.closeButton} onClick={onClose} aria-label="Close session manager">
            <X size={16} />
          </button>
        </header>

        <div className={styles.summary}>
          <div><span>Sessions</span><strong>{storage.session_count}</strong></div>
          <div><span>Session data</span><strong>{formatStorageBytes(storage.total_size_bytes)}</strong></div>
        </div>

        <div className={styles.location}>
          <div>
            <span className={styles.locationLabel}>Storage location</span>
            <code>{storage.storage_dir}</code>
          </div>
        </div>
        <p className={styles.note}>Model weights are stored separately and are not included in this total. Deleting a session also removes it from its group.</p>

        <div className={styles.listHeader}>
          <span>Saved sessions</span>
          <div role="group" aria-label="Sort sessions">
            <button type="button" className={sortMode === "size" ? styles.sortActive : ""} onClick={() => setSortMode("size")}>Largest</button>
            <button type="button" className={sortMode === "last-used" ? styles.sortActive : ""} onClick={() => setSortMode("last-used")}>Last used</button>
          </div>
        </div>

        <div className={styles.sessionList}>
          {sessions.length === 0 ? (
            <p className={styles.empty}>No saved sessions.</p>
          ) : sessions.map(session => (
            <div className={styles.sessionRow} key={session.session_id}>
              <div className={styles.sessionName} title={session.session_id}>{session.session_id}</div>
              <span className={styles.lastUsed}>{formatLastUsed(session.last_used_at)}</span>
              <span className={styles.size}>{formatStorageBytes(session.size_bytes)}</span>
              <button
                type="button"
                className={styles.copyButton}
                title={`Copy folder path for ${session.session_id}`}
                aria-label={`Copy folder path for ${session.session_id}`}
                onClick={() => onCopySessionPath(sessionFolderPath(storage.storage_dir, session.session_id, session.storage_path))}
              >
                <Copy size={13} />
              </button>
              <button
                type="button"
                className={styles.deleteButton}
                title={`Delete session ${session.session_id}`}
                aria-label={`Delete session ${session.session_id}`}
                onClick={() => onDelete(session.session_id)}
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
