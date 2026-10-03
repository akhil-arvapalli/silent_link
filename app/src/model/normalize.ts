/**
 * Mirror of the Python training-pipeline normalization
 * (model/src/data/normalize.py) so on-device inputs match training.
 *
 * A hand is 21 landmarks {x, y, z}. Two hands are packed into a (42, 3)
 * frame with the LEFT hand in rows [0:21] and RIGHT hand in rows [21:42]
 * (matching pack_hands). Each hand is then made wrist-relative and
 * scale-invariant (divided by mean bone length).
 */

export interface Landmark {
  x: number;
  y: number;
  z: number;
}

/** Landmarks per hand, MediaPipe topology. Must match data/normalize.py. */
export const NUM_LANDMARKS = 21;

/** Joints per frame: two hands x 21 landmarks. Must match data/normalize.py. */
export const V = 42;
/** x, y, z per joint. Must match data/normalize.py. */
export const C = 3;

export type Hand = Landmark[];

function handToArray(hand: Hand | undefined): Float32Array {
  const out = new Float32Array(NUM_LANDMARKS * C);
  if (!hand) return out;
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    const lm = hand[i];
    if (!lm) continue;
    out[i * C] = lm.x;
    out[i * C + 1] = lm.y;
    out[i * C + 2] = lm.z;
  }
  return out;
}

/** Pack left/right hands into a single (42, 3) frame, left rows first. */
export function packHands(left?: Hand, right?: Hand): Float32Array {
  const frame = new Float32Array(V * C);
  frame.set(handToArray(left), 0);
  frame.set(handToArray(right), NUM_LANDMARKS * C);
  return frame;
}

function normalizeHand(frame: Float32Array, offset: number): void {
  // wrist = landmark 0
  const wx = frame[offset];
  const wy = frame[offset + 1];
  const wz = frame[offset + 2];

  const rel = new Float32Array(NUM_LANDMARKS * C);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    rel[i * C] = frame[offset + i * C] - wx;
    rel[i * C + 1] = frame[offset + i * C + 1] - wy;
    rel[i * C + 2] = frame[offset + i * C + 2] - wz;
  }

  let boneSum = 0;
  for (let i = 1; i < NUM_LANDMARKS; i++) {
    const dx = rel[i * C] - rel[(i - 1) * C];
    const dy = rel[i * C + 1] - rel[(i - 1) * C + 1];
    const dz = rel[i * C + 2] - rel[(i - 1) * C + 2];
    boneSum += Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  const boneLen = boneSum / (NUM_LANDMARKS - 1);
  const scale = boneLen > 1e-6 ? boneLen : 1;
  for (let i = 0; i < NUM_LANDMARKS * C; i++) {
    frame[offset + i] = rel[i] / scale;
  }
}

/** Wrist-relative + scale-invariant normalize of a packed (42,3) frame. */
export function normalizeFrame(frame: Float32Array): Float32Array {
  normalizeHand(frame, 0);
  normalizeHand(frame, NUM_LANDMARKS * C);
  return frame;
}
