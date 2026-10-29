import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, Minus, Plus, Maximize, ChevronLeft, ChevronRight } from "lucide-react";
import styles from "./SplitWorkspaceView.module.css";
import type { SegmentationOutput } from "../hooks/useSegmentationState";
import { compareSegmentations, groupFindingsByNearestNeighbor, CandidateParticleComparison, ParticleDifference, ParticleDifferenceRegion } from "../utils/segmentationCompare";
import type { LabeledMask } from "../utils/numpyLabels";

interface Props {
  imageSrc: string;
  imageFilter: string;
  outputs: SegmentationOutput[];
  imageWidth: number;
  imageHeight: number;
  maskOpacity: number;
  onMaskOpacityChange: (opacity: number) => void;
  formatModel: (model: string) => string;
  onClose: () => void;
}

interface PaneViewState {
  zoom: number;
  panX: number;
  panY: number;
}

interface PanGesture {
  outputId: string;
  pointerId: number;
  zoom: number;
  startX: number;
  startY: number;
  startPanX: number;
  startPanY: number;
}

const INITIAL_PANE_VIEW: PaneViewState = { zoom: 1, panX: 0, panY: 0 };

type ComparisonMode = "masks" | "pixels";

function PixelDifferenceCanvas({ reference, candidate, opacity }: { reference: LabeledMask; candidate: LabeledMask; opacity: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (reference.width !== candidate.width || reference.height !== candidate.height) {
      throw new Error("Segmentation outputs have different image dimensions.");
    }
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = reference.width;
    canvas.height = reference.height;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Could not create a pixel difference overlay.");
    const image = context.createImageData(reference.width, reference.height);
    for (let index = 0; index < reference.labels.length; index += 1) {
      const referenceForeground = reference.labels[index] !== 0;
      const candidateForeground = candidate.labels[index] !== 0;
      const offset = index * 4;
      if (referenceForeground && !candidateForeground) {
        image.data[offset] = 20;
        image.data[offset + 1] = 210;
        image.data[offset + 2] = 255;
        image.data[offset + 3] = 210;
      } else if (!referenceForeground && candidateForeground) {
        image.data[offset] = 255;
        image.data[offset + 1] = 86;
        image.data[offset + 2] = 64;
        image.data[offset + 3] = 210;
      }
    }
    context.putImageData(image, 0, 0);
  }, [candidate, reference]);

  return <canvas ref={canvasRef} className={styles.differenceCanvas} style={{ opacity }} aria-label="Pixel disagreement overlay" />;
}

function InstanceBoundaryCanvas({ labels, instanceIds, color }: { labels: LabeledMask; instanceIds: number[]; color: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = labels.width;
    canvas.height = labels.height;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Could not create the selected particle outline.");
    context.clearRect(0, 0, labels.width, labels.height);
    if (instanceIds.length === 0) return;
    const image = context.createImageData(labels.width, labels.height);
    const selectedIds = new Set(instanceIds);
    const boundaryIndices: number[] = [];
    for (let y = 0; y < labels.height; y += 1) {
      for (let x = 0; x < labels.width; x += 1) {
        const index = y * labels.width + x;
        const instanceId = labels.labels[index];
        if (!selectedIds.has(instanceId)) continue;
        const isBoundary = x === 0 || y === 0 || x === labels.width - 1 || y === labels.height - 1
          || labels.labels[index - 1] !== instanceId
          || labels.labels[index + 1] !== instanceId
          || labels.labels[index - labels.width] !== instanceId
          || labels.labels[index + labels.width] !== instanceId;
        if (!isBoundary) continue;
        boundaryIndices.push(index);
      }
    }
    const outerColor = color === "reference" ? [12, 12, 12] : [255, 255, 255];
    const innerColor = color === "reference" ? [255, 255, 255] : [12, 12, 12];
    for (const [radius, pixelColor] of [[3, outerColor], [1, innerColor]] as const) {
      for (const index of boundaryIndices) {
        const boundaryX = index % labels.width;
        const boundaryY = Math.floor(index / labels.width);
        for (let offsetY = -radius; offsetY <= radius; offsetY += 1) {
          for (let offsetX = -radius; offsetX <= radius; offsetX += 1) {
            if (offsetX * offsetX + offsetY * offsetY > radius * radius) continue;
            const x = boundaryX + offsetX;
            const y = boundaryY + offsetY;
            if (x < 0 || y < 0 || x >= labels.width || y >= labels.height) continue;
            const pixelOffset = (y * labels.width + x) * 4;
            image.data[pixelOffset] = pixelColor[0];
            image.data[pixelOffset + 1] = pixelColor[1];
            image.data[pixelOffset + 2] = pixelColor[2];
            image.data[pixelOffset + 3] = 255;
          }
        }
      }
    }
    context.putImageData(image, 0, 0);
  }, [color, instanceIds, labels]);

  return <canvas ref={canvasRef} className={styles.differenceCanvas} aria-label="Selected particle outline" />;
}

function ParticleCentroidOverlay({ areas, width, height, zoom }: { areas: ParticleDifference["referenceAreas"]; width: number; height: number; zoom: number }) {
  const overlayRef = useRef<HTMLDivElement>(null);
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    const observer = new ResizeObserver(entries => {
      const entry = entries[0];
      if (!entry) return;
      setStageSize({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    observer.observe(overlay);
    return () => observer.disconnect();
  }, []);

  const displayedWidth = Math.min(stageSize.width, stageSize.height * width / height);
  const displayedHeight = displayedWidth * height / width;
  const offsetX = (stageSize.width - displayedWidth) / 2;
  const offsetY = (stageSize.height - displayedHeight) / 2;
  const smallestParticleDiameter = areas.length === 0
    ? 0
    : Math.min(...areas.map(({ area }) => 2 * Math.sqrt(area / Math.PI)));
  const displayedFontSize = Math.max(3, Math.min(9, smallestParticleDiameter * displayedWidth / width * zoom * 0.2));

  return (
    <div ref={overlayRef} className={styles.centroidOverlay} aria-label="Selected particle IDs">
      {areas.map(({ id, centerX, centerY }) => {
        return (
        <span
          key={`${id}:${centerX}:${centerY}`}
          className={styles.centroidLabel}
          style={{
            left: `${offsetX + centerX / width * displayedWidth}px`,
            top: `${offsetY + centerY / height * displayedHeight}px`,
            fontSize: `${displayedFontSize / zoom}px`,
          }}
        >
          #{id}
        </span>
        );
      })}
    </div>
  );
}

function formatInstanceAreas(areas: ParticleDifference["referenceAreas"]): string {
  if (areas.length === 0) return "none";
  return areas.map(({ id, area }) => `#${id} (${area.toLocaleString()} px²)`).join(", ");
}

function layoutClassName(outputCount: number): string {
  if (outputCount === 1) return styles.layoutOne;
  if (outputCount === 2) return styles.layoutTwo;
  if (outputCount === 3) return styles.layoutThree;
  return styles.layoutFour;
}

export default function SplitWorkspaceView({
  imageSrc,
  imageFilter,
  outputs,
  imageWidth,
  imageHeight,
  maskOpacity,
  onMaskOpacityChange,
  formatModel,
  onClose,
}: Props) {
  const [viewByOutput, setViewByOutput] = useState<Record<string, PaneViewState>>({});
  const [comparisonMode, setComparisonMode] = useState<ComparisonMode>("masks");
  const [referenceOutputId, setReferenceOutputId] = useState(outputs[0]?.id ?? "");
  const [regionIndex, setRegionIndex] = useState(0);
  const [regionGap, setRegionGap] = useState(1.5);
  const panGesture = useRef<PanGesture | null>(null);
  const stageByOutput = useRef<Map<string, HTMLDivElement>>(new Map());
  const referenceOutput = outputs.find(output => output.id === referenceOutputId) ?? outputs[0];
  const orderedOutputs = useMemo(() => referenceOutput
    ? [referenceOutput, ...outputs.filter(output => output.id !== referenceOutput.id)]
    : outputs, [outputs, referenceOutput]);
  const canCompareOutputs = outputs.length > 1 && outputs.every(output => Boolean(output.labelMask));
  const comparisonByOutput = useMemo(() => {
    if (!referenceOutput?.labelMask) return new Map<string, ReturnType<typeof compareSegmentations>>();
    const comparisons = new Map<string, ReturnType<typeof compareSegmentations>>();
    for (const output of outputs) {
      if (output.id === referenceOutput.id || !output.labelMask) continue;
      comparisons.set(output.id, compareSegmentations(referenceOutput.labelMask, output.labelMask));
    }
    return comparisons;
  }, [outputs, referenceOutput]);
  const regions: ParticleDifferenceRegion[] = useMemo(() => {
    const comparisons: CandidateParticleComparison[] = outputs
      .filter(output => output.id !== referenceOutput?.id)
      .map(candidate => ({
        candidateOutputId: candidate.id,
        findings: comparisonByOutput.get(candidate.id)?.particles ?? [],
        particles: candidate.stats?.particles ?? null,
      }));
    return groupFindingsByNearestNeighbor(comparisons, referenceOutput?.stats?.particles ?? null, regionGap);
  }, [comparisonByOutput, outputs, referenceOutput, regionGap]);
  const activeRegion = regions[regionIndex];
  const activeReferenceAreas = useMemo(() => {
    const areaById = new Map<number, ParticleDifference["referenceAreas"][number]>();
    for (const comparison of activeRegion?.comparisons ?? []) {
      for (const area of comparison.finding.referenceAreas) areaById.set(area.id, area);
    }
    return [...areaById.values()];
  }, [activeRegion]);
  const activeReferenceInstanceIds = useMemo(() => activeReferenceAreas.map(area => area.id), [activeReferenceAreas]);
  const activeCandidateAreasByOutput = useMemo(() => new Map(outputs.map(output => [
    output.id,
    (activeRegion?.comparisons ?? [])
      .filter(comparison => comparison.candidateOutputId === output.id)
      .flatMap(comparison => comparison.finding.candidateAreas),
  ])), [activeRegion, outputs]);
  const activeCandidateInstanceIdsByOutput = useMemo(() => new Map([...activeCandidateAreasByOutput].map(([outputId, areas]) => [
    outputId,
    [...new Set(areas.map(area => area.id))],
  ])), [activeCandidateAreasByOutput]);
  const pixelComparisons = outputs
    .filter(output => output.id !== referenceOutput?.id)
    .flatMap(output => {
      const comparison = comparisonByOutput.get(output.id);
      return comparison ? [{ output, comparison }] : [];
    });

  function changeZoom(outputId: string, delta: number) {
    setViewByOutput(current => {
      const currentView = current[outputId] ?? INITIAL_PANE_VIEW;
      const zoom = Math.max(1, Math.min(4, currentView.zoom + delta));
      return {
        ...current,
        [outputId]: {
          zoom,
          panX: zoom === 1 ? 0 : currentView.panX,
          panY: zoom === 1 ? 0 : currentView.panY,
        },
      };
    });
  }

  function resetZoom(outputId: string) {
    setViewByOutput(current => ({ ...current, [outputId]: INITIAL_PANE_VIEW }));
  }

  function focusRegion(index: number) {
    if (regions.length === 0) return;
    const normalizedIndex = (index + regions.length) % regions.length;
    setRegionIndex(normalizedIndex);
    const region = regions[normalizedIndex];
    for (const output of outputs) {
      const stage = stageByOutput.current.get(output.id);
      if (!stage) continue;
      const zoom = Math.max(viewByOutput[output.id]?.zoom ?? 1, 2);
      const displayedWidth = Math.min(stage.clientWidth, stage.clientHeight * imageWidth / imageHeight);
      const displayedHeight = displayedWidth * imageHeight / imageWidth;
      const desiredX = -((region.centerX / imageWidth) - 0.5) * displayedWidth * zoom;
      const desiredY = -((region.centerY / imageHeight) - 0.5) * displayedHeight * zoom;
      const maxPanX = (stage.clientWidth * (zoom - 1)) / 2;
      const maxPanY = (stage.clientHeight * (zoom - 1)) / 2;
      setViewByOutput(current => ({
        ...current,
        [output.id]: {
          zoom,
          panX: Math.min(maxPanX, Math.max(-maxPanX, desiredX)),
          panY: Math.min(maxPanY, Math.max(-maxPanY, desiredY)),
        },
      }));
    }
  }

  function handleWheel(outputId: string, event: React.WheelEvent<HTMLDivElement>) {
    event.preventDefault();
    changeZoom(outputId, event.deltaY > 0 ? -0.1 : 0.1);
  }

  function handlePanStart(outputId: string, view: PaneViewState, event: React.PointerEvent<HTMLDivElement>) {
    if (!(event.metaKey || event.ctrlKey) || event.button !== 0 || view.zoom <= 1) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    panGesture.current = {
      outputId,
      pointerId: event.pointerId,
      zoom: view.zoom,
      startX: event.clientX,
      startY: event.clientY,
      startPanX: view.panX,
      startPanY: view.panY,
    };
  }

  function handlePanMove(event: React.PointerEvent<HTMLDivElement>) {
    const gesture = panGesture.current;
    if (!gesture || gesture.pointerId !== event.pointerId) return;
    const maxPanX = (event.currentTarget.clientWidth * (gesture.zoom - 1)) / 2;
    const maxPanY = (event.currentTarget.clientHeight * (gesture.zoom - 1)) / 2;
    const panX = Math.min(maxPanX, Math.max(-maxPanX, gesture.startPanX + event.clientX - gesture.startX));
    const panY = Math.min(maxPanY, Math.max(-maxPanY, gesture.startPanY + event.clientY - gesture.startY));
    setViewByOutput(current => ({
      ...current,
      [gesture.outputId]: { zoom: gesture.zoom, panX, panY },
    }));
  }

  function handlePanEnd(event: React.PointerEvent<HTMLDivElement>) {
    if (panGesture.current?.pointerId === event.pointerId) panGesture.current = null;
  }

  return (
    <section className={styles.overlay} aria-label="Split segmentation view">
      <header className={styles.toolbar}>
        <div className={styles.toolbarTitle}>
          <span className={styles.kicker}>TEMseg</span>
          <strong>Split view</strong>
          <span className={styles.outputCount}>{outputs.length} output{outputs.length === 1 ? "" : "s"}</span>
        </div>
        <div className={styles.modeControls} role="group" aria-label="Comparison mode">
          {(["masks", "pixels"] as ComparisonMode[]).map(mode => (
            <button type="button" key={mode} className={comparisonMode === mode ? styles.activeMode : ""} disabled={mode !== "masks" && !canCompareOutputs} title={mode !== "masks" && !canCompareOutputs ? "Run fresh segmentations for each output to enable comparisons." : undefined} onClick={() => setComparisonMode(mode)}>
              {mode === "masks" ? "Masks" : "Pixel diff"}
            </button>
          ))}
        </div>
        {outputs.length > 1 && (
          <label className={styles.referenceControl}>
            <span>Compare to</span>
            <select value={referenceOutput?.id ?? ""} onChange={event => { setReferenceOutputId(event.target.value); setRegionIndex(0); }}>
              {outputs.map(output => {
                const sameModelOutputs = outputs.filter(item => item.model === output.model);
                const runSuffix = sameModelOutputs.length > 1 ? ` · run ${sameModelOutputs.indexOf(output) + 1}` : "";
                return <option key={output.id} value={output.id}>{formatModel(output.model)}{runSuffix}</option>;
              })}
            </select>
          </label>
        )}
        <label className={styles.opacityControl}>
          <span>Mask opacity</span>
          <input
            type="range"
            min={5}
            max={100}
            value={Math.round(maskOpacity * 100)}
            onChange={event => onMaskOpacityChange(Number(event.target.value) / 100)}
          />
          <span className={styles.opacityValue}>{Math.round(maskOpacity * 100)}%</span>
        </label>
        <button type="button" className={styles.closeButton} onClick={onClose}>
          <ArrowLeft size={14} /> Back to workspace
        </button>
      </header>
      {comparisonMode === "pixels" && (
        <div className={styles.comparisonSummary}>
          {pixelComparisons.map(({ output, comparison }) => (
            <div className={styles.comparisonStat} key={output.id}>
              <strong>{formatModel(output.model)} vs {referenceOutput ? formatModel(referenceOutput.model) : "reference"}</strong>
              <span>IoU {comparison.pixels.iou.toFixed(3)}</span>
              <span><i className={styles.referenceOnlySwatch} /> {comparison.pixels.referenceOnly.toLocaleString()} ref-only px</span>
              <span><i className={styles.candidateOnlySwatch} /> {comparison.pixels.candidateOnly.toLocaleString()} model-only px</span>
            </div>
          ))}
        </div>
      )}
      <div className={styles.findingExplorer}>
        <div className={styles.regionControls}>
          <div className={styles.regionNavigation}>
            <button type="button" onClick={() => focusRegion(regionIndex - 1)} disabled={regions.length === 0} aria-label="Previous region"><ChevronLeft size={14} /></button>
            <button type="button" onClick={() => focusRegion(regionIndex + 1)} disabled={regions.length === 0} aria-label="Next region"><ChevronRight size={14} /></button>
            <span>{activeRegion ? `Region ${regionIndex + 1} / ${regions.length}` : "Regions"}</span>
          </div>
          <label className={styles.regionGapControl}>
            <span>Group within</span>
            <input type="range" min={0.5} max={3} step={0.25} value={regionGap} onChange={event => { setRegionGap(Number(event.target.value)); setRegionIndex(0); }} />
            <span>{regionGap.toFixed(2)} diameters</span>
          </label>
        </div>
        {activeRegion ? (
          <div className={styles.findingDetails}>
            <div className={styles.regionTitle}>
              <strong>{new Set(activeRegion.comparisons.flatMap(comparison => comparison.finding.referenceIds)).size} base particles</strong>
              <span>{activeRegion.comparisons.length} particle comparisons across {new Set(activeRegion.comparisons.map(comparison => comparison.candidateOutputId)).size} model{new Set(activeRegion.comparisons.map(comparison => comparison.candidateOutputId)).size === 1 ? "" : "s"}</span>
            </div>
            <div className={styles.regionTableWrap}>
              <table className={styles.regionTable}>
                <thead>
                  <tr><th>Finding</th><th>Base · {referenceOutput ? formatModel(referenceOutput.model) : ""}</th><th>Compared model</th><th>Compared instance</th><th>Instance IoU</th></tr>
                </thead>
                <tbody>
                  {activeRegion.comparisons.map(({ candidateOutputId, finding }) => {
                    const candidate = outputs.find(output => output.id === candidateOutputId);
                    return (
                      <tr key={`${candidateOutputId}:${finding.id}`}>
                        <td>{finding.kind.toUpperCase()}</td>
                        <td>{formatInstanceAreas(finding.referenceAreas)}</td>
                        <td>{candidate ? formatModel(candidate.model) : "Unknown model"}</td>
                        <td>{formatInstanceAreas(finding.candidateAreas)}</td>
                        <td>{finding.iou === null ? "—" : finding.iou.toFixed(3)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className={styles.emptyRegionState}>{outputs.length < 2 ? "Select at least two outputs to navigate regions." : canCompareOutputs ? "No particle differences found for this reference." : "Region navigation needs labeled masks for every output."}</div>
        )}
      </div>
      <div className={`${styles.grid} ${layoutClassName(outputs.length)}`}>
        {orderedOutputs.map(output => {
          const view = viewByOutput[output.id] ?? INITIAL_PANE_VIEW;
          const isPanning = panGesture.current?.outputId === output.id;
          const candidateAreas = activeCandidateAreasByOutput.get(output.id) ?? [];
          const candidateInstanceIds = activeCandidateInstanceIdsByOutput.get(output.id) ?? [];
          return (
            <article className={styles.pane} key={output.id}>
              <header className={styles.paneHeader}>
                <div className={styles.paneLabel}>
                  <strong>{formatModel(output.model)}</strong>
                  {output.stats && <span>{output.stats.particle_count} particles</span>}
                  <span className={styles.paneRole}>
                    {output.id === referenceOutput?.id ? "BASE" : `VS ${referenceOutput ? formatModel(referenceOutput.model) : "BASE"}`}
                  </span>
                </div>
                <div className={styles.zoomControls}>
                  <button type="button" onClick={() => changeZoom(output.id, -0.25)} disabled={view.zoom <= 1} aria-label={`Zoom out ${formatModel(output.model)}`}>
                    <Minus size={13} />
                  </button>
                  <span>{Math.round(view.zoom * 100)}%</span>
                  <button type="button" onClick={() => changeZoom(output.id, 0.25)} disabled={view.zoom >= 4} aria-label={`Zoom in ${formatModel(output.model)}`}>
                    <Plus size={13} />
                  </button>
                  <button type="button" onClick={() => resetZoom(output.id)} aria-label={`Fit ${formatModel(output.model)}`} title="Fit to pane">
                    <Maximize size={12} />
                  </button>
                </div>
              </header>
              <div
                className={styles.imageStage}
                ref={node => {
                  if (node) stageByOutput.current.set(output.id, node);
                  else stageByOutput.current.delete(output.id);
                }}
                onWheel={event => handleWheel(output.id, event)}
                onPointerDown={event => handlePanStart(output.id, view, event)}
                onPointerMove={handlePanMove}
                onPointerUp={handlePanEnd}
                onPointerCancel={handlePanEnd}
                style={{ cursor: isPanning ? "grabbing" : view.zoom > 1 ? "grab" : "default" }}
              >
                <div
                  className={styles.imageStack}
                  style={{ transform: `translate3d(${view.panX}px, ${view.panY}px, 0) scale(${view.zoom})` }}
                >
                  <img src={imageSrc} alt="TEM input" className={styles.baseImage} style={{ filter: imageFilter }} />
                  {comparisonMode === "pixels" && output.id !== referenceOutput?.id && referenceOutput?.labelMask && output.labelMask ? (
                    <PixelDifferenceCanvas reference={referenceOutput.labelMask} candidate={output.labelMask} opacity={maskOpacity} />
                  ) : (
                    <img
                      src={output.maskUrl}
                      alt={`${formatModel(output.model)} mask overlay`}
                      className={styles.maskImage}
                      style={{ opacity: maskOpacity }}
                    />
                  )}
                  {activeRegion && output.labelMask && output.id === referenceOutput?.id && (
                    comparisonMode === "pixels"
                      ? <ParticleCentroidOverlay areas={activeReferenceAreas} width={output.labelMask.width} height={output.labelMask.height} zoom={view.zoom} />
                      : <InstanceBoundaryCanvas labels={output.labelMask} instanceIds={activeReferenceInstanceIds} color="reference" />
                  )}
                  {activeRegion && output.labelMask && candidateAreas.length > 0 && (
                    comparisonMode === "pixels"
                      ? <ParticleCentroidOverlay areas={candidateAreas} width={output.labelMask.width} height={output.labelMask.height} zoom={view.zoom} />
                      : <InstanceBoundaryCanvas labels={output.labelMask} instanceIds={candidateInstanceIds} color="candidate" />
                  )}
                </div>
              </div>
            </article>
          );
        })}
      </div>
      <footer className={styles.footer}>
        Scroll to zoom · ⌘/Ctrl-drag to pan · Zoom and pan are independent. {comparisonMode === "pixels" ? "Cyan is reference only; red is compared only. Crosshairs label selected particle IDs." : "Black and white outlines mark selected instances. Mask opacity applies to the masks."}
      </footer>
    </section>
  );
}
