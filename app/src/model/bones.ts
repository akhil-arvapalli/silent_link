/**
 * Hand skeleton topology + helpers for the Skia avatar.
 *
 * Edges mirror model/src/models/stgcn.py `_HAND_EDGES` (MediaPipe hand graph).
 * A frame is flat (V*C = 126) with [0:63] = left hand, [63:126] = right hand.
 */

export const HAND_EDGES: ReadonlyArray<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4], // thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // index
  [0, 9], [9, 10], [10, 11], [11, 12], // middle
  [0, 13], [13, 14], [14, 15], [15, 16], // ring
  [0, 17], [17, 18], [18, 19], [19, 20], // pinky
];

export const V = 42;
export const C = 3;

/**
 * Read the (x, y) screen position of joint `j` at frame `t` from a flat
 * (T*V*C) trajectory, projected to a canvas of the given width/height.
 *
 * Coordinates are wrist-relative + scale-invariant (mean bone ~1), so we
 * scale by `scale` px per unit and center on the canvas. Z is dropped.
 */
export function jointPoint(
  data: Float32Array,
  t: number,
  j: number,
  width: number,
  height: number,
  scale: number,
): { x: number; y: number } {
  const base = t * V * C + j * C;
  const x = data[base];
  const y = data[base + 1];
  const z = data[base + 2];
  // Push z into x for a slight pseudo-depth; keeps values small.
  const depth = z * 0.25 * scale;
  return {
    x: width / 2 + x * scale + depth,
    y: height / 2 + y * scale,
  };
}
