import {
    covarianceComponentsFromValuesWithScales,
    covarianceToScaleQuat,
    ellipsoidArea,
    makeCovarianceToScaleQuatScratch
} from './covariance';
import { selectRepresentatives } from './select';
import type { Descriptors } from './select';

// Private records: xyz, quaternion WXYZ, log scales, LINEAR opacity, SH,
// extra-column words. They never cross the public ChunkSource boundary.
const planBlock = (data: Float32Array, stride: number, target: number) => {
    const n = data.length / stride;
    const cloud: Descriptors = {
        positions: new Float32Array(n * 3),
        rgb: new Float32Array(n * 3),
        covariance: new Float32Array(n * 6),
        weights: new Float32Array(n),
        radii: new Float32Array(n),
        opacity: new Float32Array(n)
    };
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    const scales = new Float64Array(3);
    for (let i = 0; i < n; i++) {
        const base = i * stride;
        for (let a = 0; a < 3; a++) {
            const p = data[base + a];
            cloud.positions[i * 3 + a] = p;
            min[a] = Math.min(min[a], p);
            max[a] = Math.max(max[a], p);
            cloud.rgb[i * 3 + a] = Math.max(0, Math.min(1, 0.5 + 0.15 * data[base + 11 + a]));
        }
        covarianceComponentsFromValuesWithScales(
            data[base + 7],
            data[base + 8],
            data[base + 9],
            data[base + 4],
            data[base + 5],
            data[base + 6],
            data[base + 3],
            cloud.covariance,
            i * 6,
            scales
        );
        cloud.opacity[i] = data[base + 10];
        cloud.weights[i] = Math.max(1e-12, Math.fround(data[base + 10] * Math.max(...scales)));
        const ex = Math.fround(3 * Math.sqrt(Math.max(cloud.covariance[i * 6], 1e-20)));
        const ey = Math.fround(3 * Math.sqrt(Math.max(cloud.covariance[i * 6 + 3], 1e-20)));
        const ez = Math.fround(3 * Math.sqrt(Math.max(cloud.covariance[i * 6 + 5], 1e-20)));
        const volume = Math.cbrt(ex * ey * ez);
        const rms = Math.sqrt((ex * ex + ey * ey + ez * ez) / 3);
        cloud.radii[i] = volume * Math.pow(rms / volume, 0.5);
    }
    return selectRepresentatives(cloud, { minimum: min, extents: () => max.map((v, a) => v - min[a]) }, target);
};

// Weighted online covariance keeps small local detail at large world offsets.
// All SH coefficients use the same opacity-area weights. Unknown extra columns
// retain the first assigned input's bits, including uint32 attributes.
const mergeBlock = (data: Float32Array, stride: number, colorDim: number, target: number): Float32Array => {
    const { assignment, count } = planBlock(data, stride, target);
    const output = new Float32Array(count * stride);
    const sourceWords = new Uint32Array(data.buffer, data.byteOffset, data.length);
    const outputWords = new Uint32Array(output.buffer);
    const counts = new Uint32Array(count);
    const weights = new Float64Array(count);
    const means = new Float64Array(count * 3);
    const covariances = new Float64Array(count * 6);
    const colors = new Float64Array(count * colorDim);
    const cov = new Float64Array(6);
    const scales = new Float64Array(3);

    for (let i = 0; i < assignment.length; i++) {
        const base = i * stride;
        const slot = assignment[i];
        const dst = slot * stride;
        if (counts[slot]++ === 0) outputWords.set(sourceWords.subarray(base, base + stride), dst);
        covarianceComponentsFromValuesWithScales(
            data[base + 7],
            data[base + 8],
            data[base + 9],
            data[base + 4],
            data[base + 5],
            data[base + 6],
            data[base + 3],
            cov,
            0,
            scales
        );
        const weight = data[base + 10] * ellipsoidArea(scales[0], scales[1], scales[2]);
        const sum = weights[slot] + weight;
        weights[slot] = sum;
        const m = slot * 3;
        const c = slot * 6;
        if (Number.isFinite(weight) && weight > 0 && Number.isFinite(sum)) {
            const alpha = weight * (1 / sum);
            const dx = data[base] - means[m];
            const dy = data[base + 1] - means[m + 1];
            const dz = data[base + 2] - means[m + 2];
            means[m] += alpha * dx;
            means[m + 1] += alpha * dy;
            means[m + 2] += alpha * dz;
            const dx2 = data[base] - means[m];
            const dy2 = data[base + 1] - means[m + 1];
            const dz2 = data[base + 2] - means[m + 2];
            covariances[c] += weight * (cov[0] + dx * dx2);
            covariances[c + 1] += weight * (cov[1] + dx * dy2);
            covariances[c + 2] += weight * (cov[2] + dx * dz2);
            covariances[c + 3] += weight * (cov[3] + dy * dy2);
            covariances[c + 4] += weight * (cov[4] + dy * dz2);
            covariances[c + 5] += weight * (cov[5] + dz * dz2);
        }
        for (let k = 0; k < colorDim; k++) colors[slot * colorDim + k] += data[base + 11 + k] * weight;
    }

    const scratch = makeCovarianceToScaleQuatScratch();
    const quat = new Float32Array(4);
    for (let slot = 0; slot < count; slot++) {
        const sum = weights[slot];
        if (counts[slot] <= 1 || !Number.isFinite(sum) || sum <= 1e-12) continue;
        const base = slot * stride;
        const c = slot * 6;
        const inv = 1 / sum;
        for (let a = 0; a < 3; a++) output[base + a] = means[slot * 3 + a];
        covarianceToScaleQuat(
            Math.max(covariances[c] * inv, 1e-20),
            covariances[c + 1] * inv,
            covariances[c + 2] * inv,
            Math.max(covariances[c + 3] * inv, 1e-20),
            covariances[c + 4] * inv,
            Math.max(covariances[c + 5] * inv, 1e-20),
            output,
            base + 7,
            quat,
            0,
            scratch
        );
        output[base + 3] = quat[3];
        output[base + 4] = quat[0];
        output[base + 5] = quat[1];
        output[base + 6] = quat[2];
        const area = ellipsoidArea(Math.exp(output[base + 7]), Math.exp(output[base + 8]), Math.exp(output[base + 9]));
        const opacity = sum / Math.max(area, 1e-30);
        output[base + 10] = Number.isFinite(opacity) && opacity > 0 ? opacity : 1e-6;
        for (let k = 0; k < colorDim; k++) output[base + 11 + k] = colors[slot * colorDim + k] * inv;
    }
    return output;
};

const simplifyBlock = (input: Float32Array, stride: number, colorDim: number, target: number): Float32Array => {
    let data = input;
    while (data.length / stride > target) {
        data = mergeBlock(data, stride, colorDim, Math.max(target, Math.ceil(data.length / stride / 2)));
    }
    return data;
};

export { simplifyBlock };
