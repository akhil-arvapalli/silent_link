/**
 * SkeletonAvatar — Skia renderer for a (T, 42, 3) landmark trajectory.
 *
 * Plays back a stitched motion sequence as an animated two-hand skeleton.
 * Each hand is drawn as a single Skia path of bones; the frame index advances
 * on every reanimated frame callback.
 */

import { Canvas, Group, Path, Skia, type SkPath } from '@shopify/react-native-skia';
import { useEffect } from 'react';
import { runOnJS, useDerivedValue, useFrameCallback, useSharedValue } from 'react-native-reanimated';

import { HAND_EDGES, jointPoint } from '../model/bones';
import { NUM_LANDMARKS } from '../model/normalize';
import { colors } from '../theme';

interface Props {
  data: Float32Array;
  totalFrames: number;
  width: number;
  height: number;
  scale?: number;
  color?: string;
  onLoop?: () => void;
}

function buildPath(
  data: Float32Array,
  frame: number,
  width: number,
  height: number,
  scale: number,
): SkPath {
  const path = Skia.Path.Make();
  for (let hand = 0; hand < 2; hand++) {
    const offset = hand * NUM_LANDMARKS;
    for (const [a, b] of HAND_EDGES) {
      const p1 = jointPoint(data, frame, offset + a, width, height, scale);
      const p2 = jointPoint(data, frame, offset + b, width, height, scale);
      path.moveTo(p1.x, p1.y);
      path.lineTo(p2.x, p2.y);
    }
  }
  return path;
}

export function SkeletonAvatar({
  data,
  totalFrames,
  width,
  height,
  scale = 60,
  color = colors.teal,
  onLoop,
}: Props) {
  const frame = useSharedValue(0);

  const path = useDerivedValue(() => {
    const f = Math.min(Math.floor(frame.value), totalFrames - 1);
    return buildPath(data, f, width, height, scale);
  }, [data, totalFrames, width, height, scale]);

  useFrameCallback((info) => {
    'worklet';
    if (!info.timeSincePreviousFrame) return;
    const next = frame.value + 1;
    if (next >= totalFrames) {
      frame.value = 0;
      if (onLoop) runOnJS(onLoop)();
      return;
    }
    frame.value = next;
  });

  useEffect(() => {
    frame.value = 0;
  }, [data, totalFrames, frame]);

  return (
    <Canvas style={{ width, height }}>
      <Group>
        <Path path={path} style="stroke" strokeWidth={3.5} strokeCap="round" strokeJoin="round" color={color} />
        <Path path={path} style="stroke" strokeWidth={8} strokeCap="round" strokeJoin="round" color={color} opacity={0.15} />
      </Group>
    </Canvas>
  );
}
