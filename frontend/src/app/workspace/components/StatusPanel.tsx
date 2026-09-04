"use client";
import { useEffect, useRef, useState } from "react";
import { CircleCheck, TriangleAlert, CircleX, Loader2, X } from "lucide-react";
import { getUiEvents, UiEvent } from "@/lib/api";
import styles from "./StatusPanel.module.css";

function timeOf(ts: string): string {
  const d = new Date(ts);
  if (isNaN(d.getTime())) return "";
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function EventIcon({ level }: { level: UiEvent["level"] }) {
  if (level === "error") return <CircleX size={13} className={styles.iconError} />;
  if (level === "warning") return <TriangleAlert size={13} className={styles.iconWarn} />;
  return <CircleCheck size={13} className={styles.iconInfo} />;
}

export default function StatusPanel({ onClose }: { onClose: () => void }) {
  const [events, setEvents] = useState<UiEvent[]>([]);
  const lastIdRef = useRef(0);
  const [fetching, setFetching] = useState(false);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      setFetching(true);
      try {
        const fresh = await getUiEvents(lastIdRef.current);
        if (!alive) return;
        if (fresh.length > 0) {
          lastIdRef.current = fresh[fresh.length - 1].id;
          setEvents(prev => [...prev, ...fresh].slice(-30));
        }
      } catch {
        /* backend unreachable — panel just shows what it has */
      } finally {
        if (alive) setFetching(false);
      }
    };
    poll();
    const id = setInterval(poll, 3000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const visible = [...events].reverse();

  return (
    <div className={styles.panel}>
      <div className={styles.head}>
        <span className={styles.title}>
          Activity {fetching ? <Loader2 size={11} className={styles.spin} /> : null}
        </span>
        <button type="button" className={styles.close} onClick={onClose} aria-label="Close activity">
          <X size={13} />
        </button>
      </div>
      <div className={styles.list}>
        {visible.length === 0 && <p className={styles.empty}>Nothing to report yet.</p>}
        {visible.map(e => (
          <div key={e.id} className={styles.row}>
            <span className={styles.icon}><EventIcon level={e.level} /></span>
            <div className={styles.body}>
              <p className={styles.msg}>{e.message}</p>
              {e.progress != null && (
                <div className={styles.progressTrack}>
                  <div
                    className={styles.progressFill}
                    style={{ width: `${Math.round(Math.min(1, Math.max(0, e.progress)) * 100)}%` }}
                  />
                </div>
              )}
            </div>
            <span className={styles.time}>{timeOf(e.ts)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
