import type { LabeledMask } from "./numpyLabels";
import type { ParticleStats } from "@/lib/api";

export type ParticleDifferenceKind = "match" | "missing" | "extra" | "split" | "merge" | "complex" | "boundary" | "size";

export interface ParticleArea {
  id: number;
  area: number;
  centerX: number;
  centerY: number;
  bounds: { x: number; y: number; width: number; height: number };
}

export interface ParticleDifference {
  id: string;
  kind: ParticleDifferenceKind;
  referenceIds: number[];
  candidateIds: number[];
  referenceAreas: ParticleArea[];
  candidateAreas: ParticleArea[];
  iou: number | null;
  centerX: number;
  centerY: number;
  severity: number;
  detail: string;
}

export interface PixelDifferenceSummary {
  referenceOnly: number;
  candidateOnly: number;
  agreement: number;
  iou: number;
  dice: number;
}

interface InstanceData {
  area: number;
  xSum: number;
  ySum: number;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

interface OverlapEdge {
  referenceId: number;
  candidateId: number;
  intersection: number;
}

export interface SegmentationComparison {
  pixels: PixelDifferenceSummary;
  particles: ParticleDifference[];
}

export interface ParticleDifferenceRegion {
  id: string;
  comparisons: RegionParticleComparison[];
  centerX: number;
  centerY: number;
  bounds: { x: number; y: number; width: number; height: number };
}

export interface RegionParticleComparison {
  candidateOutputId: string;
  finding: ParticleDifference;
}

export interface CandidateParticleComparison {
  candidateOutputId: string;
  findings: ParticleDifference[];
  particles: ParticleStats[] | null;
}

const MIN_MATCH_OVERLAP = 0.12;
const BOUNDARY_IOU_LIMIT = 0.94;
const SIZE_DIFFERENCE_LIMIT = 0.2;

function instanceCenter(instance: InstanceData): [number, number] {
  return [instance.xSum / instance.area, instance.ySum / instance.area];
}

function findingCenter(
  referenceIds: number[],
  candidateIds: number[],
  referenceInstances: Map<number, InstanceData>,
  candidateInstances: Map<number, InstanceData>,
): [number, number] {
  const centers = [
    ...referenceIds.map(id => referenceInstances.get(id)).filter((item): item is InstanceData => item !== undefined),
    ...candidateIds.map(id => candidateInstances.get(id)).filter((item): item is InstanceData => item !== undefined),
  ];
  const totals = centers.reduce(([x, y], instance) => {
    const [centerX, centerY] = instanceCenter(instance);
    return [x + centerX, y + centerY];
  }, [0, 0]);
  return [totals[0] / centers.length, totals[1] / centers.length];
}

function makeFinding(
  kind: ParticleDifferenceKind,
  referenceIds: number[],
  candidateIds: number[],
  severity: number,
  detail: string,
  iou: number | null,
  referenceInstances: Map<number, InstanceData>,
  candidateInstances: Map<number, InstanceData>,
): ParticleDifference {
  const [centerX, centerY] = findingCenter(referenceIds, candidateIds, referenceInstances, candidateInstances);
  const orderedReferenceIds = [...referenceIds].sort((a, b) => a - b);
  const orderedCandidateIds = [...candidateIds].sort((a, b) => a - b);
  const referenceAreas = orderedReferenceIds.map(id => {
    const instance = referenceInstances.get(id);
    if (!instance) throw new Error(`Reference instance ${id} is missing from the comparison.`);
    const [instanceCenterX, instanceCenterY] = instanceCenter(instance);
    return {
      id,
      area: instance.area,
      centerX: instanceCenterX,
      centerY: instanceCenterY,
      bounds: {
        x: instance.minX,
        y: instance.minY,
        width: instance.maxX - instance.minX + 1,
        height: instance.maxY - instance.minY + 1,
      },
    };
  });
  const candidateAreas = orderedCandidateIds.map(id => {
    const instance = candidateInstances.get(id);
    if (!instance) throw new Error(`Compared instance ${id} is missing from the comparison.`);
    const [instanceCenterX, instanceCenterY] = instanceCenter(instance);
    return {
      id,
      area: instance.area,
      centerX: instanceCenterX,
      centerY: instanceCenterY,
      bounds: {
        x: instance.minX,
        y: instance.minY,
        width: instance.maxX - instance.minX + 1,
        height: instance.maxY - instance.minY + 1,
      },
    };
  });
  return {
    id: `${kind}:${orderedReferenceIds.join(",")}:${orderedCandidateIds.join(",")}`,
    kind,
    referenceIds: orderedReferenceIds,
    candidateIds: orderedCandidateIds,
    referenceAreas,
    candidateAreas,
    iou,
    centerX,
    centerY,
    severity,
    detail,
  };
}

export function compareSegmentations(reference: LabeledMask, candidate: LabeledMask): SegmentationComparison {
  if (reference.width !== candidate.width || reference.height !== candidate.height) {
    throw new Error("Segmentation outputs have different image dimensions.");
  }

  const referenceInstances = new Map<number, InstanceData>();
  const candidateInstances = new Map<number, InstanceData>();
  const overlaps = new Map<number, Map<number, number>>();
  let referenceOnly = 0;
  let candidateOnly = 0;
  let agreement = 0;
  let referenceForeground = 0;
  let candidateForeground = 0;

  for (let index = 0; index < reference.labels.length; index += 1) {
    const referenceId = reference.labels[index];
    const candidateId = candidate.labels[index];
    const x = index % reference.width;
    const y = Math.floor(index / reference.width);

    if (referenceId !== 0) {
      referenceForeground += 1;
      const data = referenceInstances.get(referenceId) ?? {
        area: 0,
        xSum: 0,
        ySum: 0,
        minX: x,
        minY: y,
        maxX: x,
        maxY: y,
      };
      data.area += 1;
      data.xSum += x;
      data.ySum += y;
      data.minX = Math.min(data.minX, x);
      data.minY = Math.min(data.minY, y);
      data.maxX = Math.max(data.maxX, x);
      data.maxY = Math.max(data.maxY, y);
      referenceInstances.set(referenceId, data);
    }
    if (candidateId !== 0) {
      candidateForeground += 1;
      const data = candidateInstances.get(candidateId) ?? {
        area: 0,
        xSum: 0,
        ySum: 0,
        minX: x,
        minY: y,
        maxX: x,
        maxY: y,
      };
      data.area += 1;
      data.xSum += x;
      data.ySum += y;
      data.minX = Math.min(data.minX, x);
      data.minY = Math.min(data.minY, y);
      data.maxX = Math.max(data.maxX, x);
      data.maxY = Math.max(data.maxY, y);
      candidateInstances.set(candidateId, data);
    }
    if (referenceId !== 0 && candidateId !== 0) {
      agreement += 1;
      const candidateOverlaps = overlaps.get(referenceId) ?? new Map<number, number>();
      candidateOverlaps.set(candidateId, (candidateOverlaps.get(candidateId) ?? 0) + 1);
      overlaps.set(referenceId, candidateOverlaps);
    } else if (referenceId !== 0) {
      referenceOnly += 1;
    } else if (candidateId !== 0) {
      candidateOnly += 1;
    }
  }

  const edges: OverlapEdge[] = [];
  for (const [referenceId, candidates] of overlaps) {
    const referenceArea = referenceInstances.get(referenceId)?.area ?? 0;
    for (const [candidateId, intersection] of candidates) {
      const candidateArea = candidateInstances.get(candidateId)?.area ?? 0;
      if (intersection / Math.min(referenceArea, candidateArea) >= MIN_MATCH_OVERLAP) {
        edges.push({ referenceId, candidateId, intersection });
      }
    }
  }

  const referenceEdges = new Map<number, OverlapEdge[]>();
  const candidateEdges = new Map<number, OverlapEdge[]>();
  for (const edge of edges) {
    referenceEdges.set(edge.referenceId, [...(referenceEdges.get(edge.referenceId) ?? []), edge]);
    candidateEdges.set(edge.candidateId, [...(candidateEdges.get(edge.candidateId) ?? []), edge]);
  }

  const visitedReferences = new Set<number>();
  const visitedCandidates = new Set<number>();
  const particles: ParticleDifference[] = [];

  for (const edge of edges) {
    if (visitedReferences.has(edge.referenceId) && visitedCandidates.has(edge.candidateId)) continue;
    const componentReferences = new Set<number>();
    const componentCandidates = new Set<number>();
    const pendingReferences = [edge.referenceId];
    const pendingCandidates: number[] = [];
    while (pendingReferences.length > 0 || pendingCandidates.length > 0) {
      while (pendingReferences.length > 0) {
        const referenceId = pendingReferences.pop();
        if (referenceId === undefined || componentReferences.has(referenceId)) continue;
        componentReferences.add(referenceId);
        visitedReferences.add(referenceId);
        for (const linkedEdge of referenceEdges.get(referenceId) ?? []) {
          pendingCandidates.push(linkedEdge.candidateId);
        }
      }
      while (pendingCandidates.length > 0) {
        const candidateId = pendingCandidates.pop();
        if (candidateId === undefined || componentCandidates.has(candidateId)) continue;
        componentCandidates.add(candidateId);
        visitedCandidates.add(candidateId);
        for (const linkedEdge of candidateEdges.get(candidateId) ?? []) {
          pendingReferences.push(linkedEdge.referenceId);
        }
      }
    }

    const referenceIds = [...componentReferences];
    const candidateIds = [...componentCandidates];
    if (referenceIds.length === 1 && candidateIds.length > 1) {
      particles.push(makeFinding("split", referenceIds, candidateIds, candidateIds.length, `${referenceIds.length} reference particle maps to ${candidateIds.length} candidate particles`, null, referenceInstances, candidateInstances));
      continue;
    }
    if (referenceIds.length > 1 && candidateIds.length === 1) {
      particles.push(makeFinding("merge", referenceIds, candidateIds, referenceIds.length, `${referenceIds.length} reference particles map to ${candidateIds.length} candidate particle`, null, referenceInstances, candidateInstances));
      continue;
    }
    if (referenceIds.length > 1 && candidateIds.length > 1) {
      particles.push(makeFinding("complex", referenceIds, candidateIds, Math.max(referenceIds.length, candidateIds.length), `${referenceIds.length} reference particles overlap ${candidateIds.length} candidate particles`, null, referenceInstances, candidateInstances));
      continue;
    }
    if (referenceIds.length !== 1 || candidateIds.length !== 1) continue;

    const referenceId = referenceIds[0];
    const candidateId = candidateIds[0];
    const referenceArea = referenceInstances.get(referenceId)?.area ?? 0;
    const candidateArea = candidateInstances.get(candidateId)?.area ?? 0;
    const intersection = overlaps.get(referenceId)?.get(candidateId) ?? 0;
    const union = referenceArea + candidateArea - intersection;
    const iou = union === 0 ? 1 : intersection / union;
    const areaChange = Math.abs(candidateArea - referenceArea) / Math.max(referenceArea, 1);
    if (iou < BOUNDARY_IOU_LIMIT) {
      particles.push(makeFinding("boundary", referenceIds, candidateIds, 1 - iou, "Boundary disagreement", iou, referenceInstances, candidateInstances));
    } else if (areaChange >= SIZE_DIFFERENCE_LIMIT) {
      particles.push(makeFinding("size", referenceIds, candidateIds, areaChange, `Area differs by ${(areaChange * 100).toFixed(0)}%`, iou, referenceInstances, candidateInstances));
    } else {
      particles.push(makeFinding("match", referenceIds, candidateIds, 0, "No major boundary or size difference", iou, referenceInstances, candidateInstances));
    }
  }

  for (const referenceId of referenceInstances.keys()) {
    if (!referenceEdges.has(referenceId)) {
      particles.push(makeFinding("missing", [referenceId], [], referenceInstances.get(referenceId)?.area ?? 0, "Present only in the reference output", null, referenceInstances, candidateInstances));
    }
  }
  for (const candidateId of candidateInstances.keys()) {
    if (!candidateEdges.has(candidateId)) {
      particles.push(makeFinding("extra", [], [candidateId], candidateInstances.get(candidateId)?.area ?? 0, "Present only in the compared output", null, referenceInstances, candidateInstances));
    }
  }

  particles.sort((left, right) => right.severity - left.severity);
  const union = referenceForeground + candidateForeground - agreement;
  return {
    pixels: {
      referenceOnly,
      candidateOnly,
      agreement,
      iou: union === 0 ? 1 : agreement / union,
      dice: referenceForeground + candidateForeground === 0
        ? 1
        : (2 * agreement) / (referenceForeground + candidateForeground),
    },
    particles,
  };
}

export function groupFindingsByNearestNeighbor(
  candidateComparisons: CandidateParticleComparison[],
  referenceParticles: ParticleStats[] | null,
  maxDiameterRatio: number,
): ParticleDifferenceRegion[] {
  const entries = candidateComparisons.flatMap(comparison => comparison.findings.map(finding => ({
    candidateOutputId: comparison.candidateOutputId,
    finding,
  })));
  const parent = entries.map((_, index) => index);
  const findRoot = (index: number): number => {
    const currentParent = parent[index];
    if (currentParent === undefined || currentParent === index) return index;
    const root = findRoot(currentParent);
    parent[index] = root;
    return root;
  };
  const join = (first: number, second: number): void => {
    const firstRoot = findRoot(first);
    const secondRoot = findRoot(second);
    if (firstRoot !== secondRoot) parent[secondRoot] = firstRoot;
  };
  const referenceIndicesById = new Map<number, number[]>();
  const candidateIndicesByOutputAndId = new Map<string, number[]>();
  entries.forEach(({ candidateOutputId, finding }, index) => {
    finding.referenceIds.forEach(id => referenceIndicesById.set(id, [...(referenceIndicesById.get(id) ?? []), index]));
    finding.candidateIds.forEach(id => {
      const key = `${candidateOutputId}:${id}`;
      candidateIndicesByOutputAndId.set(key, [...(candidateIndicesByOutputAndId.get(key) ?? []), index]);
    });
  });

  for (const indices of referenceIndicesById.values()) {
    const firstIndex = indices[0];
    if (firstIndex === undefined) continue;
    for (const index of indices.slice(1)) join(firstIndex, index);
  }

  function joinNearbyParticles(
    particles: ParticleStats[] | null,
    indexById: Map<number, number[]>,
  ): void {
    if (!particles) return;
    const particlesById = new Map(particles.map(particle => [particle.id, particle]));
    for (const particle of particles) {
      const ownFindingIndices = indexById.get(particle.id);
      const neighborId = particle.nearest_neighbor_id;
      if (!ownFindingIndices || neighborId == null) continue;
      const neighborFindingIndices = indexById.get(neighborId);
      const neighbor = particlesById.get(neighborId);
      if (!neighborFindingIndices || !neighbor) continue;
      const meanDiameter = (particle.diameter_px + neighbor.diameter_px) / 2;
      if (meanDiameter <= 0 || particle.nearest_neighbor_px == null) continue;
      if (particle.nearest_neighbor_px / meanDiameter <= maxDiameterRatio) {
        for (const ownIndex of ownFindingIndices) {
          for (const neighborIndex of neighborFindingIndices) join(ownIndex, neighborIndex);
        }
      }
    }
  }

  joinNearbyParticles(referenceParticles, referenceIndicesById);
  for (const comparison of candidateComparisons) {
    const candidateIndexById = new Map<number, number[]>();
    for (const [key, indices] of candidateIndicesByOutputAndId) {
      if (key.startsWith(`${comparison.candidateOutputId}:`)) {
        candidateIndexById.set(Number(key.slice(comparison.candidateOutputId.length + 1)), indices);
      }
    }
    joinNearbyParticles(comparison.particles, candidateIndexById);
  }

  const entriesByRoot = new Map<number, RegionParticleComparison[]>();
  entries.forEach((entry, index) => {
    const root = findRoot(index);
    entriesByRoot.set(root, [...(entriesByRoot.get(root) ?? []), entry]);
  });

  return [...entriesByRoot.values()]
    .map(comparisons => {
      const areas = comparisons.flatMap(comparison => [
        ...comparison.finding.referenceAreas,
        ...comparison.finding.candidateAreas,
      ]);
      const firstArea = areas[0];
      if (!firstArea) throw new Error("Cannot calculate bounds for an empty particle region.");
      const bounds = areas.slice(1).reduce((current, area) => {
        const x = Math.min(current.x, area.bounds.x);
        const y = Math.min(current.y, area.bounds.y);
        const right = Math.max(current.x + current.width, area.bounds.x + area.bounds.width);
        const bottom = Math.max(current.y + current.height, area.bounds.y + area.bounds.height);
        return { x, y, width: right - x, height: bottom - y };
      }, { ...firstArea.bounds });
      return {
        id: comparisons.map(comparison => `${comparison.candidateOutputId}:${comparison.finding.id}`).sort().join("|"),
        comparisons,
        centerX: bounds.x + bounds.width / 2,
        centerY: bounds.y + bounds.height / 2,
        bounds,
      };
    })
    .sort((left, right) => left.centerY - right.centerY || left.centerX - right.centerX);
}
