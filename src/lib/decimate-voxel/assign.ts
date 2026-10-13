import type { CovarianceTerms } from './metrics';

type RepGroupBounds = ReturnType<typeof buildRepGroupBounds>;

function buildRepGroupBounds(
    groupReps: number[][],
    selected: number[],
    pos: Float32Array,
    rgb: Float32Array,
    covarianceTerms: CovarianceTerms
) {
    const groupCount = groupReps.length;
    const covariance = covarianceTerms.components;
    const logDetQuarter = covarianceTerms.logDetQuarter;
    const trace = covarianceTerms.trace;
    const posMin = new Float32Array(groupCount * 3);
    const posMax = new Float32Array(groupCount * 3);
    const rgbMin = new Float32Array(groupCount * 3);
    const rgbMax = new Float32Array(groupCount * 3);
    const traceMax = new Float32Array(groupCount);
    const packedReps = new Float32Array(selected.length * PACKED_REP_STRIDE);
    const packedRepValid = new Uint8Array(selected.length);

    for (let group = 0; group < groupCount; group++) {
        const base3 = group * 3;
        posMin[base3 + 0] = Infinity;
        posMin[base3 + 1] = Infinity;
        posMin[base3 + 2] = Infinity;
        rgbMin[base3 + 0] = Infinity;
        rgbMin[base3 + 1] = Infinity;
        rgbMin[base3 + 2] = Infinity;
        posMax[base3 + 0] = -Infinity;
        posMax[base3 + 1] = -Infinity;
        posMax[base3 + 2] = -Infinity;
        rgbMax[base3 + 0] = -Infinity;
        rgbMax[base3 + 1] = -Infinity;
        rgbMax[base3 + 2] = -Infinity;
        traceMax[group] = -Infinity;

        const reps = groupReps[group];
        if (!reps || reps.length === 0) {
            continue;
        }
        for (let i = 0; i < reps.length; i++) {
            const slot = reps[i];
            const row = selected[slot];
            const row3 = row * 3;
            const row6 = row * 6;
            const repBase = slot * PACKED_REP_STRIDE;
            const px = pos[row3 + 0];
            const py = pos[row3 + 1];
            const pz = pos[row3 + 2];
            packedReps[repBase + 0] = px;
            packedReps[repBase + 1] = py;
            packedReps[repBase + 2] = pz;
            if (px < posMin[base3 + 0]) posMin[base3 + 0] = px;
            if (py < posMin[base3 + 1]) posMin[base3 + 1] = py;
            if (pz < posMin[base3 + 2]) posMin[base3 + 2] = pz;
            if (px > posMax[base3 + 0]) posMax[base3 + 0] = px;
            if (py > posMax[base3 + 1]) posMax[base3 + 1] = py;
            if (pz > posMax[base3 + 2]) posMax[base3 + 2] = pz;

            const rr = rgb[row3 + 0];
            const rg = rgb[row3 + 1];
            const rb = rgb[row3 + 2];
            packedReps[repBase + 3] = rr;
            packedReps[repBase + 4] = rg;
            packedReps[repBase + 5] = rb;
            if (rr < rgbMin[base3 + 0]) rgbMin[base3 + 0] = rr;
            if (rg < rgbMin[base3 + 1]) rgbMin[base3 + 1] = rg;
            if (rb < rgbMin[base3 + 2]) rgbMin[base3 + 2] = rb;
            if (rr > rgbMax[base3 + 0]) rgbMax[base3 + 0] = rr;
            if (rg > rgbMax[base3 + 1]) rgbMax[base3 + 1] = rg;
            if (rb > rgbMax[base3 + 2]) rgbMax[base3 + 2] = rb;

            packedReps[repBase + 6] = covariance[row6 + 0];
            packedReps[repBase + 7] = covariance[row6 + 1];
            packedReps[repBase + 8] = covariance[row6 + 2];
            packedReps[repBase + 9] = covariance[row6 + 3];
            packedReps[repBase + 10] = covariance[row6 + 4];
            packedReps[repBase + 11] = covariance[row6 + 5];
            const repLogDet = logDetQuarter[row];
            const repTrace = trace[row];
            packedReps[repBase + 12] = repLogDet;
            packedReps[repBase + 13] = repTrace;
            packedRepValid[slot] = Number.isFinite(repLogDet) ? 1 : 0;
            if (repTrace > traceMax[group]) {
                traceMax[group] = repTrace;
            }
        }
    }

    return {
        posMin,
        posMax,
        rgbMin,
        rgbMax,
        traceMax,
        packedReps,
        packedRepValid
    };
}

// Inlined hot path for final assignment when RGB and covariance terms are both
// present. Hoists per-row invariants and skips whole neighbor groups when their
// color/position bounds cannot beat the current best representative.
const PACKED_REP_STRIDE = 14;
const PACKED_REP_POS = 0;
const PACKED_REP_RGB = 3;
const PACKED_REP_COV = 6;
const PACKED_REP_LOG_DET = 12;
const PACKED_REP_TRACE = 13;

function assignRowsToNeighborGroupsRgbCov(
    neighborGroupsByGroup: number[][],
    groupReps: number[][],
    repGroupBounds: RepGroupBounds,
    groupCount: number,
    groupOffsets: Uint32Array,
    groupIndices: Int32Array,
    assignment: Int32Array,
    pos: Float32Array,
    rgb: Float32Array,
    covarianceTerms: CovarianceTerms
) {
    const covComponents = covarianceTerms.components;
    const logDetQuarter = covarianceTerms.logDetQuarter;
    const traceArr = covarianceTerms.trace;
    const posMin = repGroupBounds.posMin;
    const posMax = repGroupBounds.posMax;
    const rgbMin = repGroupBounds.rgbMin;
    const rgbMax = repGroupBounds.rgbMax;
    const traceMax = repGroupBounds.traceMax;
    const packedReps = repGroupBounds.packedReps;
    const packedRepValid = repGroupBounds.packedRepValid;

    for (let group = 0; group < groupCount; group++) {
        const start = groupOffsets[group];
        const end = groupOffsets[group + 1];
        if (end <= start) {
            continue;
        }
        const neighborGroups = neighborGroupsByGroup[group];
        let repCount = 0;
        let onlySlot = -1;
        let ownGroupRank = -1;
        for (let i = 0; i < neighborGroups.length; i++) {
            const repGroup = neighborGroups[i];
            const reps = groupReps[repGroup];
            if (!reps || reps.length === 0) {
                continue;
            }
            if (onlySlot < 0) {
                onlySlot = reps[0];
            }
            if (repGroup === group) {
                ownGroupRank = repCount;
            }
            repCount += reps.length;
        }
        if (repCount === 0) {
            continue;
        }
        if (repCount === 1) {
            for (let p = start; p < end; p++) {
                assignment[groupIndices[p]] = onlySlot;
            }
            continue;
        }

        for (let p = start; p < end; p++) {
            const orig = groupIndices[p];
            if (assignment[orig] >= 0) {
                continue;
            }
            const orig3 = orig * 3;
            const orig6 = orig * 6;
            const origR = rgb[orig3 + 0];
            const origG = rgb[orig3 + 1];
            const origB = rgb[orig3 + 2];
            const origX = pos[orig3 + 0];
            const origY = pos[orig3 + 1];
            const origZ = pos[orig3 + 2];
            const origTrace = traceArr[orig];
            const origLogDet = logDetQuarter[orig];
            const a00 = covComponents[orig6 + 0];
            const a01 = covComponents[orig6 + 1];
            const a02 = covComponents[orig6 + 2];
            const a11 = covComponents[orig6 + 3];
            const a12 = covComponents[orig6 + 4];
            const a22 = covComponents[orig6 + 5];
            const origLogDetFinite = origLogDet === origLogDet && origLogDet !== Infinity && origLogDet !== -Infinity;

            let bestSlot = onlySlot;
            if (!origLogDetFinite) {
                assignment[orig] = bestSlot;
                continue;
            }
            let bestLogSimilarity = -Infinity;
            let bestOriginalRank = Infinity;

            // Evaluate the row's own voxel first to establish a useful bound, then
            // visit the remaining voxels in their previous sorted order. Keep each
            // representative's old rank so equal similarities retain the same
            // deterministic winner despite the changed evaluation order.
            let originalRankCursor = 0;
            for (let visit = -1; visit < neighborGroups.length; visit++) {
                let repGroup;
                let reps;
                let originalRankStart;
                if (visit < 0) {
                    if (ownGroupRank < 0) {
                        continue;
                    }
                    repGroup = group;
                    reps = groupReps[repGroup];
                    originalRankStart = ownGroupRank;
                } else {
                    repGroup = neighborGroups[visit];
                    reps = groupReps[repGroup];
                    if (!reps || reps.length === 0) {
                        continue;
                    }
                    originalRankStart = originalRankCursor;
                    originalRankCursor += reps.length;
                    if (repGroup === group) {
                        continue;
                    }
                }

                if (bestLogSimilarity > -Infinity) {
                    const bound3 = repGroup * 3;
                    let dr = 0.0;
                    if (origR < rgbMin[bound3 + 0]) {
                        dr = rgbMin[bound3 + 0] - origR;
                    } else if (origR > rgbMax[bound3 + 0]) {
                        dr = origR - rgbMax[bound3 + 0];
                    }
                    let dg = 0.0;
                    if (origG < rgbMin[bound3 + 1]) {
                        dg = rgbMin[bound3 + 1] - origG;
                    } else if (origG > rgbMax[bound3 + 1]) {
                        dg = origG - rgbMax[bound3 + 1];
                    }
                    let db = 0.0;
                    if (origB < rgbMin[bound3 + 2]) {
                        db = rgbMin[bound3 + 2] - origB;
                    } else if (origB > rgbMax[bound3 + 2]) {
                        db = origB - rgbMax[bound3 + 2];
                    }
                    const colorUpperBound = -(dr * dr + dg * dg + db * db);
                    if (
                        colorUpperBound < bestLogSimilarity ||
                        (colorUpperBound === bestLogSimilarity && originalRankStart >= bestOriginalRank)
                    ) {
                        continue;
                    }

                    const maxTrace = traceMax[repGroup];
                    const traceSum = origTrace + maxTrace;
                    if (traceSum > 0.0) {
                        let dx = 0.0;
                        if (origX < posMin[bound3 + 0]) {
                            dx = posMin[bound3 + 0] - origX;
                        } else if (origX > posMax[bound3 + 0]) {
                            dx = origX - posMax[bound3 + 0];
                        }
                        let dy = 0.0;
                        if (origY < posMin[bound3 + 1]) {
                            dy = posMin[bound3 + 1] - origY;
                        } else if (origY > posMax[bound3 + 1]) {
                            dy = origY - posMax[bound3 + 1];
                        }
                        let dz = 0.0;
                        if (origZ < posMin[bound3 + 2]) {
                            dz = posMin[bound3 + 2] - origZ;
                        } else if (origZ > posMax[bound3 + 2]) {
                            dz = origZ - posMax[bound3 + 2];
                        }
                        const upperBound = colorUpperBound - (0.25 * (dx * dx + dy * dy + dz * dz)) / traceSum;
                        if (
                            upperBound < bestLogSimilarity ||
                            (upperBound === bestLogSimilarity && originalRankStart >= bestOriginalRank)
                        ) {
                            continue;
                        }
                    }
                }

                for (let r = 0; r < reps.length; r++) {
                    const slot = reps[r];
                    const originalRank = originalRankStart + r;
                    const repBase = slot * PACKED_REP_STRIDE;
                    const dr = origR - packedReps[repBase + PACKED_REP_RGB + 0];
                    const dg = origG - packedReps[repBase + PACKED_REP_RGB + 1];
                    const db = origB - packedReps[repBase + PACKED_REP_RGB + 2];
                    const colorLog = -(dr * dr + dg * dg + db * db);
                    if (
                        colorLog < bestLogSimilarity ||
                        (colorLog === bestLogSimilarity && originalRank >= bestOriginalRank)
                    ) {
                        continue;
                    }

                    const dx = origX - packedReps[repBase + PACKED_REP_POS + 0];
                    const dy = origY - packedReps[repBase + PACKED_REP_POS + 1];
                    const dz = origZ - packedReps[repBase + PACKED_REP_POS + 2];
                    const traceSum = origTrace + packedReps[repBase + PACKED_REP_TRACE];
                    if (traceSum > 0.0) {
                        const upperBound = colorLog - (0.25 * (dx * dx + dy * dy + dz * dz)) / traceSum;
                        if (
                            upperBound < bestLogSimilarity ||
                            (upperBound === bestLogSimilarity && originalRank >= bestOriginalRank)
                        ) {
                            continue;
                        }
                    }

                    if (!packedRepValid[slot]) {
                        continue;
                    }
                    const logDetB = packedReps[repBase + PACKED_REP_LOG_DET];
                    const b00 = packedReps[repBase + PACKED_REP_COV + 0];
                    const b01 = packedReps[repBase + PACKED_REP_COV + 1];
                    const b02 = packedReps[repBase + PACKED_REP_COV + 2];
                    const b11 = packedReps[repBase + PACKED_REP_COV + 3];
                    const b12 = packedReps[repBase + PACKED_REP_COV + 4];
                    const b22 = packedReps[repBase + PACKED_REP_COV + 5];
                    const m00 = 0.5 * (a00 + b00);
                    const m01 = 0.5 * (a01 + b01);
                    const m02 = 0.5 * (a02 + b02);
                    const m11 = 0.5 * (a11 + b11);
                    const m12 = 0.5 * (a12 + b12);
                    const m22 = 0.5 * (a22 + b22);
                    const cof00 = m11 * m22 - m12 * m12;
                    const cof01 = m12 * m02 - m01 * m22;
                    const cof02 = m01 * m12 - m11 * m02;
                    const detM = m00 * cof00 + m01 * cof01 + m02 * cof02;
                    if (detM <= 0.0) {
                        continue;
                    }
                    const invDetM = 1.0 / detM;
                    const inv00 = cof00 * invDetM;
                    const inv01 = cof01 * invDetM;
                    const inv02 = cof02 * invDetM;
                    const inv11 = (m00 * m22 - m02 * m02) * invDetM;
                    const inv12 = (m01 * m02 - m00 * m12) * invDetM;
                    const inv22 = (m00 * m11 - m01 * m01) * invDetM;
                    const quad =
                        inv00 * dx * dx +
                        inv11 * dy * dy +
                        inv22 * dz * dz +
                        2.0 * (inv01 * dx * dy + inv02 * dx * dz + inv12 * dy * dz);
                    const safeQuad = quad > 0.0 ? quad : 0.0;
                    const logCoefficient = origLogDet + logDetB - 0.5 * Math.log(detM) - 0.125 * safeQuad;
                    if (
                        logCoefficient !== logCoefficient ||
                        logCoefficient === Infinity ||
                        logCoefficient === -Infinity
                    ) {
                        continue;
                    }
                    const bhatLog = logCoefficient > 0.0 ? 0.0 : logCoefficient;
                    const logSimilarity = bhatLog + colorLog;

                    if (
                        logSimilarity > bestLogSimilarity ||
                        (bestLogSimilarity > -Infinity &&
                            logSimilarity === bestLogSimilarity &&
                            originalRank < bestOriginalRank)
                    ) {
                        bestLogSimilarity = logSimilarity;
                        bestSlot = slot;
                        bestOriginalRank = originalRank;
                        if (bestLogSimilarity >= 0.0) {
                            break;
                        }
                    }
                }
                if (bestLogSimilarity >= 0.0 && bestOriginalRank === 0) {
                    break;
                }
            }
            assignment[orig] = bestSlot;
        }
    }
}

export { buildRepGroupBounds, PACKED_REP_STRIDE, assignRowsToNeighborGroupsRgbCov };
