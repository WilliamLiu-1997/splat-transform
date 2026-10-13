import { assignRowsToNeighborGroupsRgbCov, buildRepGroupBounds } from './assign';
import { groupVoxels } from './group';
import type { Bounds } from './group';
import {
    candidateDistanceFromLog,
    candidateNoveltyPriority,
    candidatePriorityIsBetter,
    gaussianLogSimilarity,
    gaussianLogSimilarityRgbCovIfAbove,
    prepareBhattacharyyaTerms
} from './metrics';
import type { CovarianceTerms } from './metrics';

type Descriptors = {
    positions: Float32Array;
    rgb: Float32Array;
    covariance: Float32Array;
    weights: Float32Array;
    radii: Float32Array;
    opacity: Float32Array;
};

function representativeIndexForGroupRange(
    cloud: Descriptors,
    groupIndices: Int32Array,
    start: number,
    end: number,
    weightAt: (row: number) => number,
    radii: Float32Array,
    covarianceTerms: CovarianceTerms,
    rgb: Float32Array
) {
    if (end <= start) {
        return -1;
    }

    const pos = cloud.positions;
    let fallback = -1;
    let fallbackWeight = -Infinity;
    let fallbackRadius = -Infinity;
    let validCount = 0;
    let totalWeight = 0.0;

    for (let p = start; p < end; p++) {
        const idx = groupIndices[p];
        validCount += 1;
        const rawWeight = weightAt(idx);
        const w = Number.isFinite(rawWeight) ? Math.max(rawWeight, 1e-12) : 1e-12;
        const radius = Number.isFinite(radii[idx]) ? radii[idx] : -Infinity;
        totalWeight += w;
        if (
            fallback < 0 ||
            w > fallbackWeight + 1e-12 ||
            (Math.abs(w - fallbackWeight) <= 1e-12 &&
                (radius > fallbackRadius + 1e-12 || (Math.abs(radius - fallbackRadius) <= 1e-12 && idx > fallback)))
        ) {
            fallback = idx;
            fallbackWeight = w;
            fallbackRadius = radius;
        }
    }

    if (validCount <= 0 || fallback < 0) {
        return -1;
    }
    if (validCount === 1) {
        return fallback;
    }
    if (!Number.isFinite(totalWeight) || totalWeight <= 0.0) {
        return fallback;
    }

    let rep = fallback;
    let bestCost = Infinity;
    let bestMedoidCost = Infinity;
    let bestWeight = fallbackWeight;
    let bestRadius = fallbackRadius;
    const invTotalWeight = 1.0 / totalWeight;
    for (let p = start; p < end; p++) {
        const idx = groupIndices[p];
        const radius = Number.isFinite(radii[idx]) ? radii[idx] : -Infinity;
        const rawWeight = weightAt(idx);
        const w = Number.isFinite(rawWeight) ? Math.max(rawWeight, 1e-12) : 1e-12;
        let cost = 0.0;
        for (let q = start; q < end; q++) {
            const other = groupIndices[q];
            const otherRawWeight = weightAt(other);
            const otherWeight = Number.isFinite(otherRawWeight) ? Math.max(otherRawWeight, 1e-12) : 1e-12;
            const logSimilarity = gaussianLogSimilarity(covarianceTerms, pos, rgb, idx, other);
            cost += otherWeight * candidateDistanceFromLog(logSimilarity);
            if (cost * invTotalWeight > (bestCost + 1e-12) * w) {
                break;
            }
        }
        const medoidCost = cost * invTotalWeight;
        const weightedCost = medoidCost / w;
        if (
            weightedCost < bestCost - 1e-12 ||
            (Math.abs(weightedCost - bestCost) <= 1e-12 &&
                (medoidCost < bestMedoidCost - 1e-12 ||
                    (Math.abs(medoidCost - bestMedoidCost) <= 1e-12 &&
                        (w > bestWeight + 1e-12 ||
                            (Math.abs(w - bestWeight) <= 1e-12 &&
                                (radius > bestRadius + 1e-12 ||
                                    (Math.abs(radius - bestRadius) <= 1e-12 && idx > rep)))))))
        ) {
            rep = idx;
            bestCost = weightedCost;
            bestMedoidCost = medoidCost;
            bestWeight = w;
            bestRadius = radius;
        }
    }
    return rep;
}

function computeAdaptiveMaxRepsByGroup(
    groupIndices: Int32Array,
    groupOffsets: Uint32Array,
    target: number,
    representativesPerVoxelTarget: number,
    weights: Float32Array
) {
    const groupCount = groupOffsets.length - 1;
    const maxRepsByGroup = new Uint32Array(groupCount);
    if (groupCount <= 0) {
        return maxRepsByGroup;
    }

    const capacities = new Uint32Array(groupCount);
    const groupWeights = new Float64Array(groupCount);
    let baseBudget = 0;
    let capacityTotal = 0;

    for (let group = 0; group < groupCount; group++) {
        const start = groupOffsets[group];
        const end = groupOffsets[group + 1];
        const size = end - start;
        if (size <= 0) {
            continue;
        }

        maxRepsByGroup[group] = 1;
        baseBudget += 1;
        const capacity = size - 1;
        capacities[group] = capacity;
        capacityTotal += capacity;
        if (capacity <= 0) {
            continue;
        }

        let groupWeight = 0.0;
        for (let p = start; p < end; p++) {
            const row = groupIndices[p];
            const weight = Number.isFinite(weights[row]) ? Math.max(weights[row], 0.0) : 0.0;
            groupWeight += weight;
        }
        if (Number.isFinite(groupWeight) && groupWeight > 0.0) {
            groupWeights[group] = groupWeight;
        }
    }

    const averageBudget = Math.max(baseBudget, groupCount * representativesPerVoxelTarget);
    const localBudget = Math.min(target, baseBudget + capacityTotal, averageBudget);
    let remaining = Math.max(0, localBudget - baseBudget);
    if (remaining <= 0 || capacityTotal <= 0) {
        return maxRepsByGroup;
    }

    let totalScore = 0.0;
    for (let group = 0; group < groupCount; group++) {
        if (capacities[group] > 0) {
            totalScore += groupWeights[group];
        }
    }
    const useCapacityFallback = !Number.isFinite(totalScore) || totalScore <= 0.0;
    if (useCapacityFallback) {
        totalScore = capacityTotal;
    }

    const assignedExtras = new Uint32Array(groupCount);
    const exactExtras = new Float64Array(groupCount);
    const order = [];
    let allocated = 0;
    for (let group = 0; group < groupCount; group++) {
        const capacity = capacities[group];
        if (capacity <= 0) {
            continue;
        }
        const score = useCapacityFallback ? capacity : groupWeights[group];
        if (!Number.isFinite(score) || score <= 0.0) {
            continue;
        }

        const exact = (remaining * score) / totalScore;
        const extra = Math.min(capacity, Math.floor(exact));
        exactExtras[group] = exact;
        assignedExtras[group] = extra;
        maxRepsByGroup[group] += extra;
        allocated += extra;
        if (extra < capacity) {
            order.push(group);
        }
    }

    remaining -= allocated;
    order.sort((a, b) => {
        const aRemainder = exactExtras[a] - assignedExtras[a];
        const bRemainder = exactExtras[b] - assignedExtras[b];
        if (aRemainder !== bRemainder) {
            return bRemainder - aRemainder;
        }
        const aScore = useCapacityFallback ? capacities[a] : groupWeights[a];
        const bScore = useCapacityFallback ? capacities[b] : groupWeights[b];
        if (aScore !== bScore) {
            return bScore - aScore;
        }
        const aCapacity = capacities[a] - assignedExtras[a];
        const bCapacity = capacities[b] - assignedExtras[b];
        if (aCapacity !== bCapacity) {
            return bCapacity - aCapacity;
        }
        return a - b;
    });

    while (remaining > 0 && order.length > 0) {
        let wrote = 0;
        for (let i = 0; i < order.length && remaining > 0; i++) {
            const group = order[i];
            if (assignedExtras[group] >= capacities[group]) {
                continue;
            }
            assignedExtras[group] += 1;
            maxRepsByGroup[group] += 1;
            remaining -= 1;
            wrote += 1;
        }
        if (wrote === 0) {
            break;
        }
    }

    return maxRepsByGroup;
}

// One cached running similarity per input, bounded by the spatial working block.
// Local choices use epsilon ties; global completion compares priority, weight,
// distance, radius and row in order.
const selectRepresentatives = (cloud: Descriptors, bounds: Bounds, target: number) => {
    const { positions, rgb, weights, radii, opacity } = cloud;
    const n = opacity.length;
    const { groupOffsets, groupIndices, neighbors } = groupVoxels(positions, bounds, target);
    const groupCount = groupOffsets.length - 1;
    const terms = prepareBhattacharyyaTerms(cloud.covariance, n);
    const selected: number[] = [];
    const assignment = new Int32Array(n).fill(-1);
    const reps: number[][] = [];
    const similarities = new Float32Array(n).fill(-Infinity);
    const bestRows = new Int32Array(groupCount).fill(-1);
    const bestDistances = new Float64Array(groupCount);

    const update = (group: number, representative: number) => {
        let best = -1;
        let distance = -Infinity;
        for (let p = groupOffsets[group]; p < groupOffsets[group + 1]; p++) {
            const row = groupIndices[p];
            if (assignment[row] >= 0 || similarities[row] >= 0) continue;
            const similarity = gaussianLogSimilarityRgbCovIfAbove(
                terms,
                positions,
                rgb,
                row,
                representative,
                similarities[row]
            );
            if (similarity > similarities[row]) similarities[row] = similarity;
            if (similarities[row] >= 0) continue;
            const d = candidateDistanceFromLog(similarities[row]);
            if (candidatePriorityIsBetter(row, d, best, distance, weights, radii)) {
                best = row;
                distance = d;
            }
        }
        bestRows[group] = best;
        bestDistances[group] = distance;
    };
    const add = (group: number, row: number) => {
        assignment[row] = selected.length;
        reps[group].push(selected.length);
        selected.push(row);
        update(group, row);
    };
    for (let group = 0; group < groupCount; group++) {
        const row = representativeIndexForGroupRange(
            cloud,
            groupIndices,
            groupOffsets[group],
            groupOffsets[group + 1],
            (i) => weights[i],
            radii,
            terms,
            rgb
        );
        reps.push([]);
        add(group, row);
    }

    const quotas = computeAdaptiveMaxRepsByGroup(groupIndices, groupOffsets, target, 24, opacity);
    for (let group = 0; group < groupCount; group++) {
        while (reps[group].length < quotas[group] && bestRows[group] >= 0 && selected.length < target) {
            add(group, bestRows[group]);
        }
    }

    // Only a group's own candidate changes when it is popped. No stale heap
    // entries or per-splat heap objects are needed.
    const heap: number[] = [];
    const better = (a: number, b: number) => {
        const ar = bestRows[a];
        const br = bestRows[b];
        const ap = candidateNoveltyPriority(bestDistances[a], weights[ar]);
        const bp = candidateNoveltyPriority(bestDistances[b], weights[br]);
        if (ap !== bp) return ap > bp;
        if (weights[ar] !== weights[br]) return weights[ar] > weights[br];
        if (bestDistances[a] !== bestDistances[b]) return bestDistances[a] > bestDistances[b];
        if (radii[ar] !== radii[br]) return radii[ar] > radii[br];
        return ar !== br ? ar > br : a < b;
    };
    const push = (group: number) => {
        if (bestRows[group] < 0) return;
        let i = heap.length;
        heap.push(group);
        while (i > 0) {
            const parent = (i - 1) >> 1;
            if (!better(group, heap[parent])) break;
            heap[i] = heap[parent];
            i = parent;
        }
        heap[i] = group;
    };
    for (let group = 0; group < groupCount; group++) push(group);
    while (selected.length < target && heap.length > 0) {
        const group = heap[0];
        const last = heap.pop()!;
        if (heap.length > 0) {
            let i = 0;
            while (i * 2 + 1 < heap.length) {
                let child = i * 2 + 1;
                if (child + 1 < heap.length && better(heap[child + 1], heap[child])) child++;
                if (!better(heap[child], last)) break;
                heap[i] = heap[child];
                i = child;
            }
            heap[i] = last;
        }
        add(group, bestRows[group]);
        push(group);
    }

    const repBounds = buildRepGroupBounds(reps, selected, positions, rgb, terms);
    assignRowsToNeighborGroupsRgbCov(
        neighbors,
        reps,
        repBounds,
        groupCount,
        groupOffsets,
        groupIndices,
        assignment,
        positions,
        rgb,
        terms
    );

    // Covered duplicates still need an exact count. Promote singletons in
    // input order without emptying any existing representative's cluster.
    const mask = new Uint8Array(n);
    for (const row of selected) mask[row] = 1;
    for (let row = 0; row < n && selected.length < target; row++) {
        if (mask[row]) continue;
        assignment[row] = selected.length;
        selected.push(row);
    }
    return { assignment, count: selected.length };
};

export { selectRepresentatives, type Descriptors };
