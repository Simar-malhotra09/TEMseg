import { useState, useCallback, useEffect, useRef } from "react";
import { segmentImage, uploadGroundTruth, computeGTScore, getInstanceLabels, isRequestAborted, YoloParams } from "@/lib/api";
import { BASE_URL, StatsResult } from "@/lib/api";
import { BlackoutRect } from "../components/BlackOutCanvas" 
import { LabeledMask, parseNumpyLabels } from "../utils/numpyLabels";

interface Options {
  sessionId: string | null;
  encoderDepth: number;
}

export interface SegmentationOutput {
  id: string;
  model: string;
  maskUrl: string;
  labelMask: LabeledMask | null;
  stats: StatsResult | null;
  timeElapsed: number;
  groundTruthScore: { iou: number; dice: number; pixel_acc: number } | null;
}

export function useSegmentationState({ sessionId, encoderDepth }: Options) {

  // segmentation
  const [segDone, setSegDone] = useState(false);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [masksVisible, setMasksVisible] = useState(true);
  const [boxesUrl, setBoxesUrl] = useState<string | null>(null);
  const [boxesVisible, setBoxesVisible] = useState(false);
  const [stats, setStats] = useState<StatsResult | null>(null);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [comparisonOutputs, setComparisonOutputs] = useState<SegmentationOutput[]>([]);
  const [currentOutputIndex, setCurrentOutputIndex] = useState(0);
  const outputUrls = useRef<Set<string>>(new Set());

  const releaseComparisonOutputs = useCallback(() => {
    for (const url of outputUrls.current) URL.revokeObjectURL(url);
    outputUrls.current.clear();
    setComparisonOutputs([]);
    setCurrentOutputIndex(0);
  }, []);

  useEffect(() => () => {
    for (const url of outputUrls.current) URL.revokeObjectURL(url);
    outputUrls.current.clear();
  }, []);

  // exclude/include regions 
  // committed: the regions that were used for the last valid seg; used for out-of-sync warning
  const [blackoutRegions, setBlackoutRegions] = useState<BlackoutRect[]>([]);
  const [invBlackoutRegions, setInvBlackoutRegions] = useState<BlackoutRect[]>([]);
  const [committedRegions, setCommittedRegions] = useState<BlackoutRect[]>([]);
  const [invCommittedRegions, setInvCommittedRegions] = useState<BlackoutRect[]>([]);
  const [isInvBlackoutMode, setIsInvBlackoutMode] = useState(false);
  const [isBlackoutMode, setIsBlackoutMode] = useState(false);

  // ground truth
  const [groundTruth, setGroundTruth] = useState(false);
  const [groundTruthStatus, setGroundTruthStatus] = useState("Upload ground truth for current image!");
  const [groundTruthScore, setGroundTruthScore] = useState<{
    iou?: number; dice?: number; pixel_acc?: number;
  } | null>(null);
  const [gtUrl, setGtUrl] = useState<string | null>(null);
  const [gtVisible, setGtVisible] = useState(false);


  // regions changed after last seg run
  const regionsOutOfSync = segDone && (
    JSON.stringify(blackoutRegions) !== JSON.stringify(committedRegions) ||
    JSON.stringify(invBlackoutRegions) !== JSON.stringify(invCommittedRegions)
  );

  // run segmentation 
  // the selected regions passed in from page.tsx refs
  const runSegmentation = useCallback(async (
    model: string,
    activeRegions: BlackoutRect[],
    blackout: boolean,
    inverse: boolean,
    yolo: YoloParams,
    append: boolean,
    previousModel: string | null,
  ): Promise<string | null> => {
    if (!sessionId) return null;
    setIsSegmenting(true);
    try {
      let outputs = comparisonOutputs;
      if (append && outputs.length === 0 && segDone && maskUrl) {
        const previousResponse = await fetch(maskUrl);
        if (!previousResponse.ok) {
          throw new Error(`Could not preserve the current segmentation: HTTP ${previousResponse.status}`);
        }
        const previousUrl = URL.createObjectURL(await previousResponse.blob());
        outputUrls.current.add(previousUrl);
        const previousLabels = parseNumpyLabels(await getInstanceLabels(sessionId));
        outputs = [{
          id: crypto.randomUUID(),
          model: previousModel ?? "Previous result",
          maskUrl: previousUrl,
          labelMask: previousLabels,
          stats,
          timeElapsed: 0,
          groundTruthScore: null,
        }];
        setComparisonOutputs(outputs);
      }
      const result = await segmentImage(
        sessionId, model, activeRegions, blackout, inverse, true, encoderDepth, yolo
      );
      if ('error' in result) {
        return "Segmentation returned no results.";
      }
      const maskResponse = await fetch(`${BASE_URL}${result.mask_url}?t=${Date.now()}`);
      if (!maskResponse.ok) {
        throw new Error(`Could not load the ${model} mask: HTTP ${maskResponse.status}`);
      }
      const outputUrl = URL.createObjectURL(await maskResponse.blob());
      outputUrls.current.add(outputUrl);
      const labelMask = parseNumpyLabels(await getInstanceLabels(sessionId));
      const output: SegmentationOutput = {
        id: crypto.randomUUID(),
        model: result.model,
        maskUrl: outputUrl,
        labelMask,
        stats: result.stats,
        timeElapsed: result.time_elapsed,
        groundTruthScore: null,
      };
      const nextOutputs = append
        ? [...outputs, output]
        : [output];
      if (!append) {
        for (const item of outputs) {
          if (item.maskUrl !== outputUrl) {
            URL.revokeObjectURL(item.maskUrl);
            outputUrls.current.delete(item.maskUrl);
          }
        }
      }
      setComparisonOutputs(nextOutputs);
      setCurrentOutputIndex(nextOutputs.length - 1);
      setMaskUrl(`${BASE_URL}${result.mask_url}?t=${Date.now()}`);
      setStats(result.stats);
      setSegDone(true);
      setMasksVisible(true);
      if (result.warning) {
        // Zero-detection case: backend still caches the SAM embedding, so the
        // bootstrap/box modes should be usable. Flip segDone to unlock them
        // without claiming we have particles.
        setCommittedRegions(blackout ? activeRegions : []);
        setInvCommittedRegions(inverse ? activeRegions : []);
        return result.warning;
      }

      setCommittedRegions(blackout ? activeRegions : []);
      setInvCommittedRegions(inverse ? activeRegions : []);
      setBoxesUrl(
        result.debug_boxes_url
          ? `${BASE_URL}${result.debug_boxes_url}?t=${Date.now()}`
          : null
      );
      setBoxesVisible(false);
      return `Segmentation completed in ${result.time_elapsed.toFixed(2)}s.`;
    } catch (err) {
      if (isRequestAborted(err)) throw err;
      console.warn("Segmentation failed:", err);
      return null;
    } finally {
      setIsSegmenting(false);
    }
  }, [sessionId, encoderDepth, comparisonOutputs, segDone, maskUrl, stats]);

  // compute GT score 
  // called after seg or after GT upload
  const scoreGroundTruth = useCallback(async (
    activeRegions: BlackoutRect[],
    blackout: boolean,
    inverse: boolean,
  ): Promise<{ iou: number; dice: number; pixel_acc: number } | null> => {
    if (!sessionId) return null;
    try {
      const scored = await computeGTScore(sessionId, activeRegions, blackout, inverse);
      setGroundTruthScore(scored.scores);
      setGroundTruthStatus("GT score computed.");
      setGtUrl(`${BASE_URL}/gt/${sessionId}/preview?t=${Date.now()}`);
      setComparisonOutputs(current => {
        if (current.length === 0) return current;
        const lastIndex = current.length - 1;
        return current.map((output, index) => index === lastIndex
          ? { ...output, groundTruthScore: scored.scores }
          : output
        );
      });
      return scored.scores;
    } catch (err) {
      console.warn("GT scoring failed:", err);
      return null;
    }
  }, [sessionId]);

  // upload GT file
  const uploadGT = useCallback(async (
    file: File,
    activeRegions: BlackoutRect[],
    blackout: boolean,
    inverse: boolean,
  ) => {
    if (!sessionId) return;
    setGroundTruthStatus("Uploading ground truth...");
    const res = await uploadGroundTruth(sessionId, file).catch(() => null);
    if (!res) {
      setGroundTruthStatus("Ground truth upload failed.");
      return;
    }
    setGroundTruth(true);
    if (res.warnings?.length > 0) {
      setGroundTruthStatus(`Warning: ${res.warnings[0]}`);
    }
    if (segDone) {
      await scoreGroundTruth(activeRegions, blackout, inverse);
    } else {
      setGroundTruthStatus("GT uploaded — run segmentation to compute score.");
    }
  }, [sessionId, segDone, scoreGroundTruth]);

  // apply blackout. called when user clicks Apply in blackout mode
  const applyBlackout = useCallback((regions: BlackoutRect[]) => {
    if (isInvBlackoutMode) {
      setInvBlackoutRegions(regions);
    } else {
      setBlackoutRegions(regions);
    }
    setIsBlackoutMode(false);
  }, [isInvBlackoutMode]);

  // clear regions for current mode
  const clearRegions = useCallback(() => {
    if (isInvBlackoutMode) {
      setInvBlackoutRegions([]);
      setInvCommittedRegions([]);
    } else {
      setBlackoutRegions([]);
      setCommittedRegions([]);
    }
  }, [isInvBlackoutMode]);

  // reset everything. called when new image is loaded
  const reset = useCallback(() => {
    releaseComparisonOutputs();
    setSegDone(false);
    setIsSegmenting(false);
    setMaskUrl(null);
    setMasksVisible(true);
    setBoxesUrl(null);
    setBoxesVisible(false);
    setStats(null);
    setBlackoutRegions([]);
    setInvBlackoutRegions([]);
    setCommittedRegions([]);
    setInvCommittedRegions([]);
    setIsInvBlackoutMode(false);
    setIsBlackoutMode(false);
    setGroundTruth(false);
    setGroundTruthStatus("Upload ground truth for current image!");
    setGroundTruthScore(null);
    setGtUrl(null);
    setGtVisible(false);
  }, [releaseComparisonOutputs]);

  return {
    // segmentation state
    segDone,
    setSegDone,
    maskUrl,
    setMaskUrl,
    masksVisible,
    setMasksVisible,
    boxesUrl,
    setBoxesUrl,
    boxesVisible,
    setBoxesVisible,
    stats,
    setStats,
    isSegmenting,
    setIsSegmenting,
    comparisonOutputs,
    currentOutputIndex,
    setCurrentOutputIndex,

    // blackout state
    isBlackoutMode,
    setIsBlackoutMode,
    isInvBlackoutMode,
    setIsInvBlackoutMode,
    blackoutRegions,
    setBlackoutRegions,
    invBlackoutRegions,
    setInvBlackoutRegions,
    regionsOutOfSync,

    // ground truth state
    groundTruth,
    groundTruthStatus,
    groundTruthScore,
    gtUrl,
    gtVisible,
    setGtVisible,

    // actions
    runSegmentation,
    scoreGroundTruth,
    uploadGT,
    applyBlackout,
    clearRegions,
    reset,
  };
}
