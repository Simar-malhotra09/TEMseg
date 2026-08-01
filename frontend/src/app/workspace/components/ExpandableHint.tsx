"use client";

import { useState } from "react";
import { ChevronRight, ChevronDown } from "lucide-react";
import styles from "./ExpandableHint.module.css";

interface ExpandableHintProps {
  summary: string;
  children: React.ReactNode;
}

export default function ExpandableHint({ summary, children }: ExpandableHintProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className={styles.wrap}>
      <button
        type="button"
        className={styles.header}
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <span>{summary}…</span>
      </button>
      {open && <div className={styles.body}>{children}</div>}
    </div>
  );
}
