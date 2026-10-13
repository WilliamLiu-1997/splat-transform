import { join } from 'pathe';

import type { ReadFileSystem } from '../io/read';
import type { FileSystem, Writer } from '../io/write';

/** File-system pair used for private, unboosted voxel-decimation records. */
type VoxelSpill = {
    writeFs: FileSystem;
    readFs: ReadFileSystem;
    scratchDir: string;
    /** Remove a completed temporary file after its last consumer closes it. */
    remove: (path: string) => Promise<void>;
};

type RecordSource = {
    count: number;
    min: number[];
    max: number[];
    read(start: number, count: number): Promise<Float32Array>;
    close(): Promise<void>;
};

// Raw records avoid round-tripping through PLY's logit opacity, which would
// destroy alpha > 1 between reductions. File access stays behind the existing
// browser/Node interfaces; the CLI owns its isolated scratch directory.
class RecordStore {
    private sources = new Set<RecordSource>();
    private writers = new Set<Writer>();
    private paths = new Set<string>();
    private residentBytes = 0;
    private nextId = 0;
    private prefix = `voxel-${crypto.randomUUID()}`;

    constructor(
        readonly stride: number,
        readonly batchRows: number,
        private memoryLimit: number,
        private spill?: VoxelSpill
    ) {}

    async createWriter() {
        const { stride, batchRows, spill } = this;
        const path = spill ? join(spill.scratchDir, `${this.prefix}-${this.nextId++}.bin`) : null;
        const writer = path ? await spill!.writeFs.createWriter(path) : null;
        if (writer) this.writers.add(writer);
        let chunks: Float32Array[] = [];
        const buffer = new Float32Array(batchRows * stride);
        let buffered = 0;
        let count = 0;
        let storedBytes = 0;
        const min = [Infinity, Infinity, Infinity];
        const max = [-Infinity, -Infinity, -Infinity];
        const flush = async () => {
            if (buffered === 0) return;
            const bytes = buffered * stride * 4;
            if (writer) {
                await writer.write(new Uint8Array(buffer.buffer, 0, bytes));
            } else {
                if (this.residentBytes + bytes > this.memoryLimit) {
                    throw new Error(
                        'Voxel decimation exceeds its in-memory storage budget; supply spill storage (--scratch-dir in the CLI).'
                    );
                }
                chunks.push(buffer.slice(0, buffered * stride));
                this.residentBytes += bytes;
                storedBytes += bytes;
            }
            buffered = 0;
        };
        const append = async (data: Float32Array) => {
            const rows = data.length / stride;
            const words = new Uint32Array(data.buffer, data.byteOffset, data.length);
            const dst = new Uint32Array(buffer.buffer);
            for (let i = 0; i < rows; i++) {
                for (let a = 0; a < 3; a++) {
                    const v = data[i * stride + a];
                    min[a] = Math.min(min[a], v);
                    max[a] = Math.max(max[a], v);
                }
            }
            let offset = 0;
            while (offset < rows) {
                const take = Math.min(rows - offset, batchRows - buffered);
                dst.set(words.subarray(offset * stride, (offset + take) * stride), buffered * stride);
                buffered += take;
                offset += take;
                if (buffered === batchRows) await flush();
            }
            count += rows;
        };
        const finish = async (): Promise<RecordSource> => {
            await flush();
            if (writer) {
                await writer.close();
                this.writers.delete(writer);
                this.paths.add(path!);
            }
            const file = path ? await spill!.readFs.createSource(path) : null;
            let closed = false;
            const source: RecordSource = {
                count,
                min,
                max,
                read: async (start, length) => {
                    if (closed) throw new Error('Voxel record source has been closed');
                    if (start < 0 || length < 0 || start + length > count) {
                        throw new Error('Voxel record read is outside the source');
                    }
                    const data = new Float32Array(length * stride);
                    if (file) {
                        const stream = file.read(start * stride * 4, (start + length) * stride * 4);
                        const bytes = new Uint8Array(data.buffer);
                        try {
                            let offset = 0;
                            while (offset < bytes.length) {
                                const read = await stream.pull(bytes.subarray(offset));
                                if (read === 0) throw new Error('Truncated voxel scratch file');
                                offset += read;
                            }
                        } finally {
                            stream.close();
                        }
                    } else {
                        const dst = new Uint32Array(data.buffer);
                        let copied = 0;
                        while (copied < length) {
                            const row = start + copied;
                            const chunk = chunks[Math.floor(row / batchRows)];
                            const local = row % batchRows;
                            const take = Math.min(length - copied, chunk.length / stride - local);
                            dst.set(new Uint32Array(chunk.buffer, local * stride * 4, take * stride), copied * stride);
                            copied += take;
                        }
                    }
                    return data;
                },
                close: async () => {
                    if (closed) return;
                    closed = true;
                    file?.close();
                    chunks = [];
                    this.residentBytes -= storedBytes;
                    this.sources.delete(source);
                    if (path) {
                        await spill!.remove(path);
                        this.paths.delete(path);
                    }
                }
            };
            this.sources.add(source);
            return source;
        };
        return { append, finish };
    }

    async close(): Promise<void> {
        const results = await Promise.allSettled([
            ...[...this.writers].map((writer) => writer.abort()),
            ...[...this.sources].map((source) => source.close())
        ]);
        this.writers.clear();
        // Also covers failures between committing a file and opening its reader.
        for (const path of this.paths) await this.spill!.remove(path);
        this.paths.clear();
        const failed = results.find((r) => r.status === 'rejected');
        if (failed?.status === 'rejected') throw failed.reason;
    }
}

export { RecordStore, type RecordSource, type VoxelSpill };
