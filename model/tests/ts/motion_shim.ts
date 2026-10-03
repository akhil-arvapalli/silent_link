/**
 * Node-side stand-in for app/src/synthesis/motion.ts.
 *
 * The real module imports motionLibrary.json, which Node's ESM loader rejects
 * without an import attribute (that syntax is a bundler-only convention). This
 * shim exposes the identical public surface — V, C, MotionLibrary — reading the
 * same JSON asset from disk, so the stitch mirror under test is the real source.
 */

import { readFileSync } from 'node:fs';

export const V = 42;
export const C = 3;

export interface MotionLibraryJson {
  frames: number;
  version: string;
  motions: Record<string, number[]>;
}

export class MotionLibrary {
  private motions: Record<string, number[]>;
  frames: number;

  constructor(data: MotionLibraryJson) {
    this.motions = data.motions;
    this.frames = data.frames;
  }

  has(gloss: string): boolean {
    return gloss in this.motions;
  }

  get(gloss: string): Float32Array {
    const raw = this.motions[gloss];
    if (!raw) return new Float32Array(this.frames * V * C);
    return Float32Array.from(raw);
  }
}

export function loadFrom(path: string): MotionLibrary {
  return new MotionLibrary(JSON.parse(readFileSync(path, 'utf8')) as MotionLibraryJson);
}
