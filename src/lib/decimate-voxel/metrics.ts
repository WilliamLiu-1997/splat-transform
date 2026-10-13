type CovarianceTerms = { components: Float32Array; logDetQuarter: Float32Array; trace: Float32Array };

function determinantSymmetric3(c00: number, c01: number, c02: number, c11: number, c12: number, c22: number) {
    return c00 * (c11 * c22 - c12 * c12) - c01 * (c01 * c22 - c12 * c02) + c02 * (c01 * c12 - c11 * c02);
}

function prepareBhattacharyyaTerms(covariance: Float32Array, rowCount: number) {
    const components = covariance;
    const logDetQuarter = new Float32Array(rowCount);
    const trace = new Float32Array(rowCount);

    for (let i = 0; i < rowCount; i++) {
        const base = i * 6;
        const c00 = covariance[base + 0];
        const c01 = covariance[base + 1];
        const c02 = covariance[base + 2];
        const c11 = covariance[base + 3];
        const c12 = covariance[base + 4];
        const c22 = covariance[base + 5];
        const determinant = determinantSymmetric3(c00, c01, c02, c11, c12, c22);
        logDetQuarter[i] = determinant > 0.0 && Number.isFinite(determinant) ? 0.25 * Math.log(determinant) : -Infinity;
        trace[i] = c00 + c11 + c22;
    }

    return { components, logDetQuarter, trace };
}

function colorLogSimilarityRgb(rgb: Float32Array, lhs: number, rhs: number) {
    const a = lhs * 3;
    const b = rhs * 3;
    const dr = rgb[a + 0] - rgb[b + 0];
    const dg = rgb[a + 1] - rgb[b + 1];
    const db = rgb[a + 2] - rgb[b + 2];
    return -(dr * dr + dg * dg + db * db);
}

function bhattacharyyaLogCoeffPrepared(covarianceTerms: CovarianceTerms, pos: Float32Array, lhs: number, rhs: number) {
    const covariance = covarianceTerms.components;
    const a = lhs * 6;
    const b = rhs * 6;
    const a00 = covariance[a + 0];
    const a01 = covariance[a + 1];
    const a02 = covariance[a + 2];
    const a11 = covariance[a + 3];
    const a12 = covariance[a + 4];
    const a22 = covariance[a + 5];
    const b00 = covariance[b + 0];
    const b01 = covariance[b + 1];
    const b02 = covariance[b + 2];
    const b11 = covariance[b + 3];
    const b12 = covariance[b + 4];
    const b22 = covariance[b + 5];

    const m00 = 0.5 * (a00 + b00);
    const m01 = 0.5 * (a01 + b01);
    const m02 = 0.5 * (a02 + b02);
    const m11 = 0.5 * (a11 + b11);
    const m12 = 0.5 * (a12 + b12);
    const m22 = 0.5 * (a22 + b22);
    const logDetA = covarianceTerms.logDetQuarter[lhs];
    const logDetB = covarianceTerms.logDetQuarter[rhs];
    if (!Number.isFinite(logDetA) || !Number.isFinite(logDetB)) {
        return -Infinity;
    }
    const detM = determinantSymmetric3(m00, m01, m02, m11, m12, m22);
    if (detM <= 0.0) {
        return -Infinity;
    }

    const invDetM = 1.0 / detM;
    const inv00 = (m11 * m22 - m12 * m12) * invDetM;
    const inv01 = (m02 * m12 - m01 * m22) * invDetM;
    const inv02 = (m01 * m12 - m02 * m11) * invDetM;
    const inv11 = (m00 * m22 - m02 * m02) * invDetM;
    const inv12 = (m01 * m02 - m00 * m12) * invDetM;
    const inv22 = (m00 * m11 - m01 * m01) * invDetM;
    const lhs3 = lhs * 3;
    const rhs3 = rhs * 3;
    const dx = pos[lhs3 + 0] - pos[rhs3 + 0];
    const dy = pos[lhs3 + 1] - pos[rhs3 + 1];
    const dz = pos[lhs3 + 2] - pos[rhs3 + 2];
    const quad =
        dx * (inv00 * dx + inv01 * dy + inv02 * dz) +
        dy * (inv01 * dx + inv11 * dy + inv12 * dz) +
        dz * (inv02 * dx + inv12 * dy + inv22 * dz);
    const logCoefficient = logDetA + logDetB - 0.5 * Math.log(detM) - 0.125 * Math.max(quad, 0.0);
    return Number.isFinite(logCoefficient) ? Math.min(0.0, logCoefficient) : -Infinity;
}

function gaussianLogSimilarity(
    covarianceTerms: CovarianceTerms,
    pos: Float32Array,
    rgb: Float32Array,
    lhs: number,
    rhs: number
) {
    return bhattacharyyaLogCoeffPrepared(covarianceTerms, pos, lhs, rhs) + colorLogSimilarityRgb(rgb, lhs, rhs);
}

function gaussianLogSimilarityRgbCovIfAbove(
    covarianceTerms: CovarianceTerms,
    pos: Float32Array,
    rgb: Float32Array,
    lhs: number,
    rhs: number,
    minLogSimilarity: number
) {
    const colorLog = colorLogSimilarityRgb(rgb, lhs, rhs);
    if (colorLog < minLogSimilarity) {
        return -Infinity;
    }
    if (minLogSimilarity > -Infinity) {
        const lhs3 = lhs * 3;
        const rhs3 = rhs * 3;
        const dx = pos[lhs3 + 0] - pos[rhs3 + 0];
        const dy = pos[lhs3 + 1] - pos[rhs3 + 1];
        const dz = pos[lhs3 + 2] - pos[rhs3 + 2];
        const traceM = 0.5 * (covarianceTerms.trace[lhs] + covarianceTerms.trace[rhs]);
        if (traceM > 0.0) {
            const upperBound = colorLog - (0.125 * (dx * dx + dy * dy + dz * dz)) / traceM;
            if (upperBound < minLogSimilarity) {
                return -Infinity;
            }
        }
    }
    return bhattacharyyaLogCoeffPrepared(covarianceTerms, pos, lhs, rhs) + colorLog;
}

const CANDIDATE_NOVELTY_DISTANCE_SCALE = 3.0;

function candidateNoveltyFactor(distance: number) {
    if (!Number.isFinite(distance)) {
        return distance > 0.0 ? 1.0 : 0.0;
    }
    const safeDistance = Math.max(distance, 0.0);
    return 1.0 - Math.exp(-safeDistance / CANDIDATE_NOVELTY_DISTANCE_SCALE);
}

function candidateNoveltyPriority(distance: number, weight: number) {
    return candidateNoveltyFactor(distance) * Math.max(weight, 1e-12);
}

function candidateDistanceFromLog(logSimilarity: number) {
    return logSimilarity === -Infinity ? Infinity : Math.max(0.0, -logSimilarity);
}

function candidatePriorityIsBetter(
    row: number,
    distance: number,
    bestRow: number,
    bestDistance: number,
    weights: Float32Array,
    radii: Float32Array,
    preparedBestNovelty: number = null
) {
    if (bestRow < 0) {
        return true;
    }
    const weight = weights[row];
    const bestWeight = weights[bestRow];
    const novelty = candidateNoveltyFactor(distance);
    const bestNovelty = preparedBestNovelty === null ? candidateNoveltyFactor(bestDistance) : preparedBestNovelty;
    const priority = novelty * Math.max(weight, 1e-12);
    const bestPriority = bestNovelty * Math.max(bestWeight, 1e-12);
    if (priority > bestPriority + 1e-12) {
        return true;
    }
    if (Math.abs(priority - bestPriority) > 1e-12) {
        return false;
    }

    if (novelty > bestNovelty + 1e-12) {
        return true;
    }
    if (Math.abs(novelty - bestNovelty) > 1e-12) {
        return false;
    }

    const weightDiff = weight - bestWeight;
    if (Math.abs(weightDiff) > 1e-12) {
        return weightDiff > 0.0;
    }

    const radiusDiff = radii[row] - radii[bestRow];
    if (Math.abs(radiusDiff) > 1e-12) {
        return radiusDiff > 0.0;
    }
    return row > bestRow;
}

export {
    type CovarianceTerms,
    candidateDistanceFromLog,
    candidateNoveltyFactor,
    candidateNoveltyPriority,
    candidatePriorityIsBetter,
    gaussianLogSimilarity,
    gaussianLogSimilarityRgbCovIfAbove,
    prepareBhattacharyyaTerms
};
