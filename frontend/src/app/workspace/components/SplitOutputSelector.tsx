import styles from "./SplitOutputSelector.module.css";
import type { SegmentationOutput } from "../hooks/useSegmentationState";

interface Props {
  outputs: SegmentationOutput[];
  selectedOutputIds: string[];
  formatModel: (model: string) => string;
  onToggle: (outputId: string, selected: boolean) => void;
  onCancel: () => void;
  onOpen: () => void;
}

export default function SplitOutputSelector({
  outputs,
  selectedOutputIds,
  formatModel,
  onToggle,
  onCancel,
  onOpen,
}: Props) {
  return (
    <div className={styles.backdrop} role="presentation" onMouseDown={event => {
      if (event.target === event.currentTarget) onCancel();
    }}>
      <section className={styles.dialog} role="dialog" aria-modal="true" aria-labelledby="split-select-title">
        <header className={styles.header}>
          <div>
            <h2 id="split-select-title">Choose segmentations</h2>
            <p>Select up to four outputs to view together.</p>
          </div>
          <span className={styles.selectionCount}>{selectedOutputIds.length}/4</span>
        </header>
        <div className={styles.outputList}>
          {outputs.map((output, index) => {
            const selected = selectedOutputIds.includes(output.id);
            const limitReached = selectedOutputIds.length >= 4;
            return (
              <label className={styles.outputRow} key={output.id}>
                <input
                  type="checkbox"
                  checked={selected}
                  disabled={!selected && limitReached}
                  onChange={event => onToggle(output.id, event.target.checked)}
                />
                <span className={styles.outputIndex}>Output {index + 1}</span>
                <span className={styles.modelName}>{formatModel(output.model)}</span>
                <span className={styles.outputMeta}>
                  {output.stats ? `${output.stats.particle_count} particles` : "No stats"}
                </span>
              </label>
            );
          })}
        </div>
        <footer className={styles.actions}>
          <button type="button" className={styles.secondaryButton} onClick={onCancel}>Cancel</button>
          <button
            type="button"
            className={styles.primaryButton}
            disabled={selectedOutputIds.length === 0}
            onClick={onOpen}
          >
            Open split view
          </button>
        </footer>
      </section>
    </div>
  );
}
