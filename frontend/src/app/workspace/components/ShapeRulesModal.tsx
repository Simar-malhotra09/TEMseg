"use client";

import { useState, useEffect, useCallback } from "react";
import { X, Plus, Trash2, ArrowUp, ArrowDown, RotateCcw } from "lucide-react";
import styles from "./ShapeRulesModal.module.css";
import {
  ShapeRule,
  ShapeCondition,
  ShapeRulesConfig,
  StatsResult,
  getShapeRules,
  updateShapeRules,
  resetShapeRules,
  reclassify,
  listShapeRulePresets,
  saveShapeRulePreset,
  loadShapeRulePreset,
} from "@/lib/api";

interface Props {
  open: boolean;
  onClose: () => void;
  sessionId: string | null;
  onStatsUpdate?: (stats: StatsResult) => void;
}

function blankCondition(metric: string, op: string): ShapeCondition {
  return { metric, op, value: 0 };
}

export default function ShapeRulesModal({ open, onClose, sessionId, onStatsUpdate }: Props) {
  const [config, setConfig] = useState<ShapeRulesConfig | null>(null);
  const [rules, setRules] = useState<ShapeRule[]>([]);
  const [presets, setPresets] = useState<string[]>([]);
  const [selectedPreset, setSelectedPreset] = useState("");
  const [presetName, setPresetName] = useState("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [cfg, pres] = await Promise.all([
        getShapeRules(),
        listShapeRulePresets(),
      ]);
      setConfig(cfg);
      setRules(cfg.rules);
      setPresets(pres.presets);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load shape rules");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) load();
  }, [open, load]);

  if (!open) return null;

  const metrics = config?.available_metrics ?? [];
  const ops = config?.available_operators ?? [];

  function updateRule(i: number, patch: Partial<ShapeRule>) {
    setRules(rs => rs.map((r, idx) => (idx === i ? { ...r, ...patch } : r)));
  }

  function updateCondition(ri: number, ci: number, patch: Partial<ShapeCondition>) {
    setRules(rs => rs.map((r, idx) => idx !== ri ? r : {
      ...r,
      conditions: r.conditions.map((c, cidx) => cidx === ci ? { ...c, ...patch } : c),
    }));
  }

  function addRule() {
    setRules(rs => [...rs, { label: "", conditions: [blankCondition(metrics[0] ?? "circularity", ops[0] ?? ">")] }]);
  }

  function removeRule(i: number) {
    setRules(rs => rs.filter((_, idx) => idx !== i));
  }

  function moveRule(i: number, dir: -1 | 1) {
    setRules(rs => {
      const j = i + dir;
      if (j < 0 || j >= rs.length) return rs;
      const copy = [...rs];
      [copy[i], copy[j]] = [copy[j], copy[i]];
      return copy;
    });
  }

  function addCondition(ri: number) {
    setRules(rs => rs.map((r, idx) => idx !== ri ? r : {
      ...r,
      conditions: [...r.conditions, blankCondition(metrics[0] ?? "circularity", ops[0] ?? ">")],
    }));
  }

  function removeCondition(ri: number, ci: number) {
    setRules(rs => rs.map((r, idx) => idx !== ri ? r : {
      ...r,
      conditions: r.conditions.filter((_, cidx) => cidx !== ci),
    }));
  }

  async function handleSave() {
    setSaving(true);
    setError(null);
    try {
      const cfg = await updateShapeRules(rules);
      setConfig(cfg);
      setRules(cfg.rules);
      reclassifyCurrentSession();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save shape rules");
    } finally {
      setSaving(false);
    }
  }

  async function handleReset() {
    setSaving(true);
    setError(null);
    try {
      const cfg = await resetShapeRules();
      setConfig(cfg);
      setRules(cfg.rules);
      reclassifyCurrentSession();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to reset shape rules");
    } finally {
      setSaving(false);
    }
  }

  // No-op when nothing is segmented yet; next /segment picks the rules up anyway.
  function reclassifyCurrentSession() {
    if (!sessionId) return;
    reclassify(sessionId)
      .then(r => onStatsUpdate?.(r.stats))
      .catch(() => {});
  }

  async function handleSavePreset() {
    const name = presetName.trim();
    if (!name) return;
    setSaving(true);
    setError(null);
    try {
      await saveShapeRulePreset(name, rules);
      const pres = await listShapeRulePresets();
      setPresets(pres.presets);
      setSelectedPreset(name);
      setPresetName("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save preset");
    } finally {
      setSaving(false);
    }
  }

  async function handleLoadPreset() {
    if (!selectedPreset) return;
    setSaving(true);
    setError(null);
    try {
      const cfg = await loadShapeRulePreset(selectedPreset);
      setConfig(cfg);
      setRules(cfg.rules);
      reclassifyCurrentSession();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load preset");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className={styles.backdrop} onClick={onClose}>
      <div className={styles.panel} onClick={e => e.stopPropagation()}>
        <div className={styles.header}>
          <span className={styles.title}>
            Shape Rules
            {config && !config.is_default && <span className={styles.customBadge}>custom</span>}
          </span>
          <button type="button" className={styles.iconBtn} onClick={onClose} title="Close">
            <X size={14} />
          </button>
        </div>

        {loading && <p className={styles.hint}>Loading…</p>}
        {error && <p className={styles.error}>{error}</p>}

        {!loading && config && (
          <>
            <p className={styles.hint}>
              Checked top to bottom — first full match wins. Unmatched particles get{" "}
              <strong>{config.default_shape}</strong>. Saving reclassifies already
              segmented particles.
            </p>

            <div className={styles.rules}>
              {rules.map((rule, ri) => (
                <div className={styles.rule} key={ri}>
                  <div className={styles.ruleHeader}>
                    <input
                      className={styles.labelInput}
                      value={rule.label}
                      placeholder="label"
                      onChange={e => updateRule(ri, { label: e.target.value })}
                    />
                    <div className={styles.ruleActions}>
                      <button type="button" className={styles.iconBtn} disabled={ri === 0} onClick={() => moveRule(ri, -1)} title="Move up">
                        <ArrowUp size={12} />
                      </button>
                      <button type="button" className={styles.iconBtn} disabled={ri === rules.length - 1} onClick={() => moveRule(ri, 1)} title="Move down">
                        <ArrowDown size={12} />
                      </button>
                      <button type="button" className={styles.iconBtn} onClick={() => removeRule(ri)} title="Remove rule">
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </div>

                  {rule.conditions.map((cond, ci) => (
                    <div className={styles.condition} key={ci}>
                      <select
                        className={styles.select}
                        value={cond.metric}
                        onChange={e => updateCondition(ri, ci, { metric: e.target.value })}
                      >
                        {metrics.map(m => <option key={m} value={m}>{m}</option>)}
                      </select>
                      <select
                        className={styles.select}
                        value={cond.op}
                        onChange={e => updateCondition(ri, ci, { op: e.target.value })}
                      >
                        {ops.map(o => <option key={o} value={o}>{o}</option>)}
                      </select>
                      <input
                        type="number"
                        step="any"
                        className={styles.valueInput}
                        value={cond.value}
                        onChange={e => updateCondition(ri, ci, { value: Number(e.target.value) })}
                      />
                      <button
                        type="button"
                        className={styles.iconBtn}
                        onClick={() => removeCondition(ri, ci)}
                        disabled={rule.conditions.length <= 1}
                        title="Remove condition"
                      >
                        <Trash2 size={11} />
                      </button>
                    </div>
                  ))}

                  <button type="button" className={styles.addConditionBtn} onClick={() => addCondition(ri)}>
                    <Plus size={11} /> Condition
                  </button>
                </div>
              ))}
            </div>

            <button type="button" className={styles.addRuleBtn} onClick={addRule}>
              <Plus size={12} /> Add Rule
            </button>
          </>
        )}

        {!loading && config && (
          <div className={styles.presetRow}>
            <select
              className={styles.select}
              value={selectedPreset}
              onChange={e => setSelectedPreset(e.target.value)}
              disabled={saving}
            >
              <option value="">Saved presets…</option>
              {presets.map(p => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
            <button
              type="button"
              className={styles.resetBtn}
              disabled={!selectedPreset || saving}
              onClick={handleLoadPreset}
            >
              Load
            </button>
            <input
              className={styles.labelInput}
              placeholder="preset name"
              value={presetName}
              onChange={e => setPresetName(e.target.value)}
              disabled={saving}
            />
            <button
              type="button"
              className={styles.resetBtn}
              disabled={!presetName.trim() || saving}
              onClick={handleSavePreset}
            >
              Save As
            </button>
          </div>
        )}

        <div className={styles.footer}>
          <button type="button" className={styles.resetBtn} onClick={handleReset} disabled={saving || loading}>
            <RotateCcw size={12} /> Reset to Default
          </button>
          <div className={styles.footerRight}>
            <button type="button" className={styles.cancelBtn} onClick={onClose} disabled={saving}>
              Cancel
            </button>
            <button type="button" className={styles.saveBtn} onClick={handleSave} disabled={saving || loading}>
              {saving ? "Saving…" : "Save"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
