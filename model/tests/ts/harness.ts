/**
 * Cross-language parity harness.
 *
 * Driven by model/tests/test_ts_parity.py, which stages this file next to
 * verbatim copies of the app's TypeScript mirrors and runs it under Node's
 * native type stripping. Node executes the *real* mirror sources, so any
 * numeric divergence from the Python originals shows up as a test failure
 * instead of silently degrading on-device inference.
 *
 * Reads `inputs.json` (written by the test) and writes `outputs.json`.
 */

import { readFileSync, writeFileSync } from 'node:fs';

import { normalizeFrame, packHands } from './mirror_model_normalize.ts';
import { GlossNormalizer } from './mirror_synth_normalize.ts';
import { stitchGlosses } from './mirror_stitch.ts';
import { MotionLibrary } from './motion_shim.ts';

const inputs = JSON.parse(readFileSync('inputs.json', 'utf8'));

const out: Record<string, unknown> = {};

// Each block is optional: the driver runs one concern per invocation.

// inputs.sequences: number[][] of length T*V*C, already packed frames.
if (inputs.sequences) {
  out.normalized = inputs.sequences.map((seq: number[]) => {
    const frame = Float32Array.from(seq);
    normalizeFrame(frame);
    return Array.from(frame);
  });
}

// inputs.pairs: [left, right] landmark triples -> packed frame round-trip.
if (inputs.pairs) {
  out.packed = inputs.pairs.map((pair: [number[][] | null, number[][] | null]) => {
    const toHand = (pts: number[][] | null) =>
      pts ? pts.map((p) => ({ x: p[0], y: p[1], z: p[2] })) : undefined;
    return Array.from(packHands(toHand(pair[0]), toHand(pair[1])));
  });
}

// inputs.sentences + inputs.glossSpecs
if (inputs.sentences) {
  const normalizer = new GlossNormalizer(inputs.glossSpecs);
  out.glosses = inputs.sentences.map((s: string) => normalizer.normalize(s));
}

// inputs.stitchCases
if (inputs.stitchCases) {
  const library = new MotionLibrary(
    JSON.parse(readFileSync(inputs.motionLibraryPath, 'utf8')),
  );
  out.stitched = {};
  for (const glosses of inputs.stitchCases) {
    out.stitched[glosses.join(' ')] = Array.from(stitchGlosses(glosses, library));
  }
}

writeFileSync('outputs.json', JSON.stringify(out));
