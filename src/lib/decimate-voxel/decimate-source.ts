// Voxel-based decimation algorithm by WilliamLiu-1997: https://github.com/WilliamLiu-1997

import { createChunkDataPool, orderedLayers } from '../chunk';
import type { ChunkData, ChunkDataPool, ChunkLayer, ChunkSource, ChunkSourceMetadata, ReadRequest } from '../chunk';
import { bakeTransform } from '../ops/bake-transform';
import { logger, sigmoid, Transform } from '../utils';

import { simplifyBlock } from './merge';
import { RecordStore } from './store';
import type { RecordSource, VoxelSpill } from './store';

type VoxelOptions = {
    /** Display-only scale adjustment. Default 0.5; 0 disables it. */
    coverageBoostScale?: number;
    /** Working and intermediate-storage budget. Default 512 MiB; excludes the input and output writer. */
    memoryBudgetBytes?: number;
    /** Disk/browser storage for scenes that do not fit the in-memory budget. */
    spill?: VoxelSpill;
};

/** Options for adaptive-voxel representative selection and moment matching. */
type DecimateVoxelOptions = VoxelOptions & {
    /** Exact number of Gaussians to keep, at least 1. */
    targetCount: number;
};

/** Options for constructing an unboosted reduction chain for Streamed SOG. */
type VoxelLodOptions = VoxelOptions & {
    /** Strictly decreasing absolute counts for levels 1 onward. Level 0 is the input. */
    targetCounts: number[];
};

const DEFAULT_MEMORY_BUDGET = 512 * 2 ** 20;
const MAX_BLOCK_ROWS = 1 << 18;

// Allocate one slot per non-empty child, then
// proportional extras, capacity-clamped and rounded by largest remainder.
const allocateTargets = (counts: number[], target: number): number[] => {
    const result = counts.map(() => 1);
    let remaining = Math.max(0, target - counts.length);
    while (remaining > 0) {
        const active = counts.map((_, i) => i).filter((i) => result[i] < counts[i]);
        const weight = active.reduce((sum, i) => sum + counts[i], 0);
        const remainders = new Float64Array(counts.length);
        const available = remaining;
        for (const i of active) {
            const exact = (available * counts[i]) / weight;
            const extra = Math.min(counts[i] - result[i], Math.floor(exact));
            result[i] += extra;
            remaining -= extra;
            remainders[i] = exact - Math.floor(exact);
        }
        active.sort((a, b) => remainders[b] - remainders[a] || counts[b] - counts[a] || a - b);
        for (const i of active) {
            if (remaining === 0) break;
            if (result[i] < counts[i]) {
                result[i]++;
                remaining--;
            }
        }
    }
    return result;
};

// Position-only analysis is streamed as well: no scene-sized XYZ or permutation
// arrays. A binned median on the widest axis gives spatially coherent children.
// Coincident positions split by row order so the working-set limit still holds.
const partition = async (source: RecordSource, store: RecordStore): Promise<RecordSource[]> => {
    const { batchRows, stride } = store;
    let axis = 0;
    for (let a = 1; a < 3; a++) {
        if (source.max[a] - source.min[a] > source.max[axis] - source.min[axis]) axis = a;
    }
    const extent = source.max[axis] - source.min[axis];
    const bins = new Uint32Array(64);
    const binOf = (v: number) => Math.min(63, Math.max(0, Math.floor(((v - source.min[axis]) / extent) * 64)));
    if (extent > 0) {
        for (let start = 0; start < source.count; start += batchRows) {
            const rows = await source.read(start, Math.min(batchRows, source.count - start));
            for (let i = 0; i < rows.length; i += stride) bins[binOf(rows[i + axis])]++;
        }
    }
    let boundary = -1;
    let distance = Infinity;
    let cumulative = 0;
    for (let i = 0; i < 63; i++) {
        cumulative += bins[i];
        if (cumulative > 0 && cumulative < source.count && Math.abs(cumulative - source.count / 2) < distance) {
            boundary = i;
            distance = Math.abs(cumulative - source.count / 2);
        }
    }
    const writers = [await store.createWriter(), await store.createWriter()];
    const buffers = [new Float32Array(batchRows * stride), new Float32Array(batchRows * stride)];
    const words = buffers.map((b) => new Uint32Array(b.buffer));
    for (let start = 0; start < source.count; start += batchRows) {
        const data = await source.read(start, Math.min(batchRows, source.count - start));
        const srcWords = new Uint32Array(data.buffer, data.byteOffset, data.length);
        const counts = [0, 0];
        for (let i = 0; i < data.length / stride; i++) {
            const left =
                boundary < 0 ? start + i < Math.floor(source.count / 2) : binOf(data[i * stride + axis]) <= boundary;
            const child = left ? 0 : 1;
            words[child].set(srcWords.subarray(i * stride, (i + 1) * stride), counts[child]++ * stride);
        }
        for (let c = 0; c < 2; c++) await writers[c].append(buffers[c].subarray(0, counts[c] * stride));
    }
    return [await writers[0].finish(), await writers[1].finish()];
};

const reduceRecords = async (
    input: RecordSource,
    target: number,
    store: RecordStore,
    colorDim: number,
    blockRows: number,
    retained: Set<RecordSource>
): Promise<RecordSource> => {
    if (input.count <= target) return input;
    const dispose = async (source: RecordSource) => {
        if (!retained.has(source)) await source.close();
    };
    if (input.count <= blockRows) {
        const data = await input.read(0, input.count);
        const reduced = simplifyBlock(data, store.stride, colorDim, target);
        const writer = await store.createWriter();
        await writer.append(reduced);
        const result = await writer.finish();
        await dispose(input);
        return result;
    }

    const children = await partition(input, store);
    await dispose(input);
    const budgets = allocateTargets(
        children.map((c) => c.count),
        target
    );
    // Process children serially; only one heavy simplification block is resident.
    for (let c = 0; c < children.length; c++) {
        children[c] = await reduceRecords(children[c], budgets[c], store, colorDim, blockRows, retained);
    }
    const writer = await store.createWriter();
    for (const child of children) {
        for (let start = 0; start < child.count; start += store.batchRows) {
            await writer.append(await child.read(start, Math.min(store.batchRows, child.count - start)));
        }
        await child.close();
    }
    const combined = await writer.finish();
    // A parent budget below its child count needs one final reduction upward.
    return combined.count > target ? reduceRecords(combined, target, store, colorDim, blockRows, retained) : combined;
};

const stageSource = async (source: ChunkSource, store: RecordStore): Promise<RecordSource> => {
    const pool = createChunkDataPool({ chunkSize: store.batchRows, maxPooledBytes: 0 });
    const layers = orderedLayers(source.meta.availableLayers);
    const indices = new Uint32Array(store.batchRows);
    const writer = await store.createWriter();
    try {
        for (let start = 0; start < source.meta.numGaussians; start += store.batchRows) {
            const count = Math.min(store.batchRows, source.meta.numGaussians - start);
            for (let i = 0; i < count; i++) indices[i] = start + i;
            const buffers: Partial<Record<ChunkLayer, ChunkData>> = {};
            try {
                for (const layer of layers) buffers[layer] = pool.acquire(layer, source.meta.layouts[layer]!, count);
                await source.read({ indices, indexOffset: 0, count, ...buffers });
                const data = new Float32Array(count * store.stride);
                const words = new Uint32Array(data.buffer);
                let offset = 0;
                for (const layer of layers) {
                    const buffer = buffers[layer]!;
                    const width = buffer.stride / 4;
                    const src = new Uint32Array(buffer.data, 0, count * width);
                    for (let i = 0; i < count; i++)
                        words.set(src.subarray(i * width, (i + 1) * width), i * store.stride + offset);
                    offset += width;
                }
                const colorEnd = 11 + source.meta.layouts.color!.stride / 4;
                for (let i = 0; i < count; i++) {
                    const base = i * store.stride;
                    for (let k = 0; k < colorEnd; k++) {
                        // Infinite logit opacities are valid endpoints; other
                        // non-finite geometry/SH must be explicitly filtered.
                        if (k === 10 ? Number.isNaN(data[base + k]) : !Number.isFinite(data[base + k])) {
                            throw new Error(
                                'Voxel decimation requires finite geometry and SH; use --filter-nan first.'
                            );
                        }
                    }
                    for (let a = 7; a < 10; a++) {
                        if (!Number.isFinite(Math.fround(Math.exp(2 * data[base + a])))) {
                            throw new Error('Voxel decimation scale overflows its covariance.');
                        }
                    }
                    data[base + 10] = sigmoid(data[base + 10]);
                }
                await writer.append(data);
            } finally {
                for (const buffer of Object.values(buffers)) buffer.release();
            }
        }
        return await writer.finish();
    } finally {
        pool.destroy();
    }
};

// The boost exists only on the public read view. Internal reductions always
// read original moments and unclamped linear opacity from RecordStore.
const displayGeometry = (geometry: Float32Array, count: number, strength: number) => {
    for (let i = 0; i < count; i++) {
        const base = i * 8;
        const opacity = geometry[base + 7];
        if (strength > 0 && opacity > 1) {
            const sx = Math.exp(geometry[base + 4]);
            const sy = Math.exp(geometry[base + 5]);
            const sz = Math.exp(geometry[base + 6]);
            const max = Math.max(sx, sy, sz);
            const mid = sx + sy + sz - max - Math.min(sx, sy, sz);
            const ratio = mid / max;
            const rest = 1 + strength * (Math.sqrt(Math.min(opacity, 1000)) - 1);
            const top = 1 + ((rest - 1) * ratio) / (1 - ratio + ratio * ratio);
            geometry[base + 4] += Math.log(sx >= max ? top : rest);
            geometry[base + 5] += Math.log(sy >= max ? top : rest);
            geometry[base + 6] += Math.log(sz >= max ? top : rest);
        }
        const p = Math.max(1e-6, Math.min(1 - 1e-6, opacity));
        geometry[base + 7] = Math.log(p / (1 - p));
    }
};

const outputSource = (
    levels: RecordSource[],
    inputMeta: ChunkSourceMetadata,
    store: RecordStore,
    strength: number,
    close: () => Promise<void>,
    original?: ChunkSource
): ChunkSource => {
    const counts = levels.map((s) => s.count);
    const meta: ChunkSourceMetadata = {
        ...inputMeta,
        numGaussians: counts[0],
        numLods: counts.length,
        lodCounts: counts,
        numChunks: counts.map((n) => Math.ceil(n / inputMeta.chunkSize))
    };
    const layers = orderedLayers(meta.availableLayers);
    const read = async (request: ReadRequest) => {
        if (original && (request.lod ?? 0) === 0) return original.read(request);
        const source = levels[request.lod ?? 0];
        if (!source) throw new Error('Voxel output LOD is out of range');
        const gather = 'indices' in request;
        const start = gather ? 0 : request.chunkIndex * meta.chunkSize;
        const count = gather ? request.count : Math.min(meta.chunkSize, source.count - start);
        const order = gather ? Uint32Array.from({ length: count }, (_, i) => i) : null;
        if (gather)
            order!.sort((a, b) => request.indices[request.indexOffset + a] - request.indices[request.indexOffset + b]);
        const rowAt = (i: number) => (gather ? request.indices[request.indexOffset + order![i]] : start + i);
        let done = 0;
        while (done < count) {
            const first = rowAt(done);
            let end = done + 1;
            // Coalesce nearby gathers but bound every scratch read by batchRows.
            while (end < count && rowAt(end) - first < store.batchRows && rowAt(end) - rowAt(end - 1) <= 64) end++;
            const data = await source.read(first, rowAt(end - 1) - first + 1);
            const words = new Uint32Array(data.buffer, data.byteOffset, data.length);
            let offset = 0;
            for (const layer of layers) {
                const width = meta.layouts[layer]!.stride / 4;
                const buffer = request[layer];
                if (buffer) {
                    const dst = new Uint32Array(buffer.data);
                    for (let i = done; i < end; i++) {
                        const src = (rowAt(i) - first) * store.stride + offset;
                        dst.set(words.subarray(src, src + width), (order ? order[i] : i) * width);
                    }
                }
                offset += width;
            }
            done = end;
        }
        if (request.geometric) displayGeometry(new Float32Array(request.geometric.data), count, strength);
    };
    return { meta, read, close };
};

const buildLevels = async (
    source: ChunkSource,
    pool: ChunkDataPool,
    targets: number[],
    includeInput: boolean,
    options: VoxelOptions
): Promise<ChunkSource> => {
    const budget = options.memoryBudgetBytes ?? DEFAULT_MEMORY_BUDGET;
    const strength = options.coverageBoostScale ?? 0.5;
    if (!Number.isFinite(budget) || budget < 2 ** 20) throw new Error('Voxel memory budget must be at least 1 MiB');
    if (!Number.isFinite(strength) || strength < 0) throw new Error('coverageBoostScale must be finite and >= 0');
    if (source.meta.numLods !== 1) throw new Error('Voxel decimation requires a single LOD; select a level first');
    for (const layer of ['position', 'geometric', 'color'] as const) {
        if (!source.meta.availableLayers.has(layer)) throw new Error(`Voxel decimation requires the ${layer} layer`);
    }
    let previous = source.meta.numGaussians;
    for (const target of targets) {
        if (!Number.isSafeInteger(target) || target < 1 || target > previous || (includeInput && target === previous)) {
            throw new Error('Voxel targets must be positive, decreasing integer counts within the input count');
        }
        previous = target;
    }
    if (targets.length === 0 || (!includeInput && targets[0] === source.meta.numGaussians)) return source;
    const baked = bakeTransform(source, Transform.PLY);
    const colorDim = baked.meta.layouts.color!.stride / 4;
    const stride = orderedLayers(baked.meta.availableLayers).reduce((s, l) => s + baked.meta.layouts[l]!.stride / 4, 0);
    // Allow for records, descriptor arrays, adaptive groups, candidate cache,
    // double-precision merge accumulators and simultaneously live input/output.
    const blockRows = Math.max(2, Math.min(MAX_BLOCK_ROWS, Math.floor(budget / 2 / (stride * 12 + 1024))));
    const batchRows = Math.min(pool.chunkSize, 8192, blockRows);
    const store = new RecordStore(stride, batchRows, Math.floor(budget / 3), options.spill);
    let closed = false;
    const close = async () => {
        if (closed) return;
        closed = true;
        try {
            await store.close();
        } finally {
            await source.close();
        }
    };
    try {
        if (options.spill) await options.spill.writeFs.mkdir(options.spill.scratchDir);
        let current = await stageSource(baked, store);
        const levels = includeInput ? [current] : [];
        // LOD 0 reads the original source directly, so its staged copy can be
        // deleted after the first reduction. Only coarse levels need retention.
        const retained = new Set<RecordSource>();
        for (const target of targets) {
            const group = logger.group(`Voxel decimate ${current.count} → ${target}`);
            try {
                current = await reduceRecords(current, target, store, colorDim, blockRows, retained);
                levels.push(current);
                retained.add(current);
            } finally {
                group.end();
            }
        }
        return outputSource(levels, baked.meta, store, strength, close, includeInput ? baked : undefined);
    } catch (error) {
        await close();
        throw error;
    }
};

/**
 * Simplify with adaptive voxel representatives, Gaussian/color similarity and
 * opacity-area moment matching. Reductions retain at least half their input per
 * pass and reach the exact target. The result supports repeated and gather reads.
 * Large inputs are partitioned with streamed disk passes when spill is supplied.
 *
 * @param source - Single-LOD input, consumed; close the result to release it and scratch files.
 * @param pool - Caller pool; caps the size of temporary gather batches.
 * @param options - Target, memory budget, optional spill storage and display coverage strength.
 * @returns A standard logit-opacity ChunkSource with display coverage applied on read.
 * @throws If the input is invalid or in-memory intermediates exceed the budget without spill storage.
 * @example
 * const reduced = await decimateSourceVoxel(source, pool, { targetCount: 100000 });
 */
const decimateSourceVoxel = (
    source: ChunkSource,
    pool: ChunkDataPool,
    options: DecimateVoxelOptions
): Promise<ChunkSource> => {
    return buildLevels(source, pool, [options.targetCount], false, options);
};

/**
 * Build a structural LOD pyramid using the voxel simplifier. Level 0 is the input;
 * later levels consume unboosted moments and unclamped opacity from their parent.
 * Coverage boost is applied only when the returned source is read by a writer.
 *
 * @param source - Single-LOD input, consumed; close the result after writing.
 * @param pool - Caller pool, as for decimateSourceVoxel.
 * @param options - Decreasing absolute target counts, memory budget and optional spill storage.
 * @returns A gatherable multi-LOD source suitable for writeLodSource.
 * @throws If counts are not decreasing or data/storage requirements are not met.
 * @example
 * const lods = await createVoxelLodSource(source, pool, { targetCounts: [500000, 250000] });
 */
const createVoxelLodSource = (
    source: ChunkSource,
    pool: ChunkDataPool,
    options: VoxelLodOptions
): Promise<ChunkSource> => {
    return buildLevels(source, pool, options.targetCounts, true, options);
};

export { decimateSourceVoxel, createVoxelLodSource, type DecimateVoxelOptions, type VoxelLodOptions, type VoxelSpill };
