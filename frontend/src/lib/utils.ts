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
