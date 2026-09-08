import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Smallest positive integer not present in usedIds. Used to allocate particle
// ids so deleted ids get reused instead of growing unbounded across edits.
export function nextFreeId(usedIds: number[]): number {
  const used = new Set(usedIds);
  let candidate = 1;
  while (used.has(candidate)) candidate++;
  return candidate;
}

// Plain-English version of a /from-points or /from-boxes rejection reason so
// the status bar doesn't show raw backend strings like "area=312 < min".
export function describeRejection(reason: string): string {
  if (reason === "already segmented")
    return "That spot is already segmented — nothing new to add there.";
  if (reason === "box too small") return "Box too small — drag a bigger box.";
  if (reason === "point out of bounds") return "Click landed outside the image.";
  if (reason === "malformed box" || reason === "malformed point")
    return "Prompt came through malformed — try again.";
  if (reason.includes("< min"))
    return "Too little new area — that spot looks mostly segmented already.";
  if (reason.includes("> max"))
    return "Region too large — try one particle per click or box.";
  if (reason.includes("area window"))
    return "Nothing particle-sized found there.";
  if (reason === "no contour" || reason === "degenerate contour")
    return "Couldn't trace a clean outline there.";
  if (reason.startsWith("SAM error"))
    return "SAM failed on that prompt — try a slightly different spot.";
  return reason;
}
