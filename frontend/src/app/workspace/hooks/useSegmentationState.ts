import { useState, useCallback } from "react";
import { segmentImage, uploadGroundTruth, computeGTScore } from "@/lib/api";
import { BASE_URL, StatsResult } from "@/lib/api";
import { BlackoutRect } from "../components/BlackOutCanvas" 

interface Options {
  sessionId: string | null;
  selectedModel: string | null;
}

export function useSegmentationState({ sessionId, selectedModel }: Options) {

  // segmentation
  const [segDone, setSegDone] = useState(false);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [masksVisible, setMasksVisible] = useState(true);
  const [stats, setStats] = useState<StatsResult | null>(null);
  const [isSegmenting, setIsSegmenting] = useState(false);

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
    activeRegions: BlackoutRect[],
    blackout: boolean,
    inverse: boolean,
  ): Promise<string | null> => {
    if (!sessionId || !selectedModel) return null;
    setIsSegmenting(true);
    try {
      const result = await segmentImage(
        sessionId, selectedModel, activeRegions, blackout, inverse
      );
      if ('error' in result || 'warning' in result) {
        return result.warning ?? result.error ?? "Segmentation returned no results.";
      }

      setCommittedRegions(blackout ? activeRegions : []);
      setInvCommittedRegions(inverse ? activeRegions : []);
      setMaskUrl(`${BASE_URL}${result.mask_url}?t=${Date.now()}`);
      setStats(result.stats);
      setSegDone(true);
      setMasksVisible(true);
      return `Segmentation completed in ${result.time_elapsed.toFixed(2)}s.`;
    } catch (err) {
      console.error("Segmentation failed:", err);
      return null;
    } finally {
      setIsSegmenting(false);
    }
  }, [sessionId, selectedModel]);

  // compute GT score 
  // called after seg or after GT upload
  const scoreGroundTruth = useCallback(async (
    activeRegions: BlackoutRect[],
    blackout: boolean,
    inverse: boolean,
  ) => {
    if (!sessionId) return;
    try {
      const scored = await computeGTScore(sessionId, activeRegions, blackout, inverse);
      setGroundTruthScore(scored.scores);
      setGroundTruthStatus("GT score computed.");
      setGtUrl(`${BASE_URL}/gt/${sessionId}/preview?t=${Date.now()}`);
    } catch (err) {
      console.error("GT scoring failed:", err);
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
    const res = await uploadGroundTruth(sessionId, file);
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

  // apply blackout — called when user clicks Apply in blackout mode
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

  // reset everything — called when new image is loaded
  const reset = useCallback(() => {
    setSegDone(false);
    setMaskUrl(null);
    setMasksVisible(true);
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
  }, []);

  return {
    // segmentation state
    segDone,
    setSegDone,
    maskUrl,
    setMaskUrl,
    masksVisible,
    setMasksVisible,
    stats,
    setStats,
    isSegmenting,
    setIsSegmenting,

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
