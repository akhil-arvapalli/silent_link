/**
 * Per-frame fingerspelling hand-shape classifier.
 *
 * Fingerspelling is a per-frame hand-shape task (no temporal window), matching
 * glosses.json -> finger_spelling. This module:
 *   1. Extracts a compact hand-shape feature vector from a (21,3) hand
 *      (per-finger curl + spread), and
 *   2. Classifies it against a small reference letter library.
 *
 * The classifier is a deterministic nearest-neighbour stub until a real
 * per-frame letter model (ONNX/TFLite) is trained — swap `classifyLetter`
 * for model inference without changing the screen.
 */

import type { Hand } from './normalize';

const NUM_LANDMARKS = 21;

// MediaPipe finger tip landmarks per finger (index, middle, ring, pinky) and thumb.
const FINGER_TIPS = [8, 12, 16, 20];

// Finger joint chains for curl: [base, mid, tip] indices (thumb uses its own).
const FINGER_CHAINS: Array<[number, number, number]> = [
  [5, 6, 8], // index
  [9, 10, 12], // middle
  [13, 14, 16], // ring
  [17, 18, 20], // pinky
];

interface Vec3 {
  x: number;
  y: number;
  z: number;
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function dot(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function norm(a: Vec3): number {
  return Math.sqrt(dot(a, a));
}

/** Cosine of the angle at vertex `b` between segments a-b and b-c. */
function jointCos(a: Vec3, b: Vec3, c: Vec3): number {
  const ab = sub(a, b);
  const bc = sub(c, b);
  const d = norm(ab) * norm(bc);
  if (d === 0) return 1;
  return dot(ab, bc) / d;
}

/**
 * Extract a hand-shape feature vector: per-finger curl (0 = straight,
 * 1 = curled) + spread (thumb vs each finger tip). Returns a flat number[].
 */
export function handFeatures(hand: Hand | undefined): number[] {
  if (!hand || hand.length < NUM_LANDMARKS) return Array(8).fill(0);
  const lm = (i: number): Vec3 => hand[i] as Vec3;

  const curl = FINGER_CHAINS.map(([base, mid, tip]) => {
    // 1 - cos is 0 when straight (cos~1), ~1 when fully curled (cos~-1).
    const c = jointCos(lm(base), lm(mid), lm(tip));
    return Math.max(0, Math.min(1, (1 - c) / 2));
  });

  // Thumb curl: angle between thumb base and tip relative to index base.
  const thumbCurl = Math.max(0, Math.min(1, (1 - jointCos(lm(0), lm(1), lm(4))) / 2));

  // Spread: normalized distance from index base to each finger tip.
  const wrist = lm(0);
  const indexBase = lm(5);
  const spread = FINGER_TIPS.map((tip) => {
    const d = norm(sub(lm(tip), indexBase));
    const w = norm(sub(lm(tip), wrist));
    return w === 0 ? 0 : d / w;
  });

  return [thumbCurl, ...curl, ...spread];
}

/**
 * Reference letter shapes (curl + spread) used for the demo nearest-neighbour.
 * Keyed by letter; values are [thumbCurl, index..pinky curl, index..pinky spread].
 * Replace with a trained model for real accuracy.
 */
const LETTER_SHAPES: Record<string, number[]> = {
  a: [1, 1, 1, 1, 1, 0.7, 0.7, 0.7, 0.7],
  b: [0.2, 0.1, 0.1, 0.1, 0.6, 1.1, 1.1, 1.1, 0.9],
  c: [0.5, 0.4, 0.4, 0.4, 0.4, 0.9, 0.9, 0.9, 0.8],
  l: [0.3, 0.1, 0.8, 0.8, 0.8, 1.1, 1.1, 1.1, 0.9],
  o: [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
  v: [0.2, 0.1, 0.1, 0.9, 0.9, 1.1, 1.1, 0.6, 0.6],
  y: [0.2, 0.1, 0.9, 0.9, 0.1, 1.1, 0.5, 0.5, 1.1],
};

function distance(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

/** Classify a single hand frame to the nearest letter (or null if too far). */
export function classifyLetter(hand: Hand | undefined): { letter: string; confidence: number } | null {
  const f = handFeatures(hand);
  let best: string | null = null;
  let bestD = Infinity;
  for (const [letter, shape] of Object.entries(LETTER_SHAPES)) {
    const d = distance(f, shape);
    if (d < bestD) {
      bestD = d;
      best = letter;
    }
  }
  if (best === null || bestD > 1.6) return null;
  const confidence = Math.max(0, 1 - bestD / 1.6);
  return { letter: best, confidence };
}
