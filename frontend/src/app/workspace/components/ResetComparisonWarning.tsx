import styles from "./ResetComparisonWarning.module.css";

interface Props {
  modelName: string;
  dontShowAgain: boolean;
  onDontShowAgainChange: (checked: boolean) => void;
  onCancel: () => void;
  onAppend: () => void;
  onReset: () => void;
}

export default function ResetComparisonWarning({
  modelName,
  dontShowAgain,
  onDontShowAgainChange,
  onCancel,
  onAppend,
  onReset,
}: Props) {
  return (
    <div className={styles.backdrop} role="presentation" onMouseDown={event => {
      if (event.target === event.currentTarget) onCancel();
    }}>
      <section className={styles.dialog} role="dialog" aria-modal="true" aria-labelledby="reset-comparison-title">
        <header className={styles.header}>
          <h2 id="reset-comparison-title">Start a new comparison?</h2>
        </header>
        <div className={styles.body}>
          <p>
            A fresh run with <strong>{modelName}</strong> will replace the saved comparison
            and remove its other model outputs. To update this model and keep the others,
            append the run instead.
          </p>
          <label className={styles.preference}>
            <input
              type="checkbox"
              checked={dontShowAgain}
              onChange={event => onDontShowAgainChange(event.target.checked)}
            />
            <span>Don’t show this warning again</span>
          </label>
        </div>
        <footer className={styles.actions}>
          <button type="button" className={styles.secondaryButton} onClick={onCancel}>Cancel</button>
          <button type="button" className={styles.appendButton} onClick={onAppend}>Append {modelName}</button>
          <button type="button" className={styles.resetButton} onClick={onReset}>Run and reset</button>
        </footer>
      </section>
    </div>
  );
}
