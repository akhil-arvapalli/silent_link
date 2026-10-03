/**
 * Concatenative landmark motion stitching (mirrors model/src/synthesis/stitch.py).
 *
 * Takes a gloss sequence and produces a continuous flat (T_total, 42, 3)
 * trajectory by concatenating each gloss's motion template with a raised-cosine
 * coarticulation crossfade at the boundaries.
 */

import { C, V, type MotionLibrary } from './motion';

function raisedCosineBlend(window: number): Float32Array {
  const ramp = new Float32Array(window);
  const denom = window > 1 ? window - 1 : 1;
  for (let i = 0; i < window; i++) {
    ramp[i] = 0.5 - 0.5 * Math.cos((Math.PI * i) / denom);
  }
  return ramp;
}

/** Flat index for frame t, joint j, channel c. */
function idx(t: number, j: number, c: number): number {
  return (t * V + j) * C + c;
}

/** Slice a flat (T*V*C) array by frame range [t0, t1) into a new flat array. */
function frameSlice(flat: Float32Array, t0: number, t1: number): Float32Array {
  const n = Math.max(0, t1 - t0);
  const out = new Float32Array(n * V * C);
  for (let t = 0; t < n; t++) {
    for (let j = 0; j < V; j++) {
      for (let c = 0; c < C; c++) {
        out[idx(t, j, c)] = flat[idx(t0 + t, j, c)];
      }
    }
  }
  return out;
}

export function stitchGlosses(
  glosses: string[],
  library: MotionLibrary,
  blendFrames = 8,
): Float32Array {
  if (glosses.length === 0) return new Float32Array(0);
  const frames = library.frames;

  const blocks = glosses.map((g) => library.get(g));
  if (blocks.length === 1) return blocks[0];

  const parts: Float32Array[] = [blocks[0]];
  for (let i = 1; i < blocks.length; i++) {
    const next = blocks[i];
    const w = Math.min(blendFrames, parts[parts.length - 1].length / (V * C), frames);
    if (w <= 0) {
      parts.push(next);
      continue;
    }
    const tail = frameSlice(parts[parts.length - 1], parts[parts.length - 1].length / (V * C) - w, parts[parts.length - 1].length / (V * C));
    const head = frameSlice(next, 0, w);
    const ramp = raisedCosineBlend(w);
    const blended = new Float32Array(w * V * C);
    for (let t = 0; t < w; t++) {
      for (let j = 0; j < V; j++) {
        for (let c = 0; c < C; c++) {
          blended[idx(t, j, c)] = tail[idx(t, j, c)] * (1 - ramp[t]) + head[idx(t, j, c)] * ramp[t];
        }
      }
    }
    parts[parts.length - 1] = frameSlice(parts[parts.length - 1], 0, parts[parts.length - 1].length / (V * C) - w);
    parts.push(blended);
    parts.push(frameSlice(next, w, frames));
  }

  const total = parts.reduce((sum, b) => sum + b.length, 0);
  const result = new Float32Array(total);
  let offset = 0;
  for (const b of parts) {
    result.set(b, offset);
    offset += b.length;
  }
  return result;
}
