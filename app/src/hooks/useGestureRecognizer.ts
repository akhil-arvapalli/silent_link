import { useCallback, useRef, useState } from 'react';
import type { Frame } from 'react-native-vision-camera';

import { topGloss } from '../config/glosses';
import * as classifier from '../model/classifier';
import { SEQUENCE_LENGTH } from '../model/classifier';
import { C, V, normalizeFrame, packHands, type Hand } from '../model/normalize';

export interface HandednessCategory {
  categoryName: string;
  score: number;
  displayName: string;
}

export interface HandDetectionResult {
  hands: Hand[];
  handedness?: HandednessCategory[][];
  imageWidth?: number;
  imageHeight?: number;
  delegates?: Record<string, string>;
  error?: string;
}

/**
 * The frame processor installed by the native MediaPipe plugin.
 *
 * Declared once, here, next to the result type it returns. GestureScreen and
 * FingerspellingScreen each used to declare it too, with different result
 * types; `declare global` merges duplicate functions as overloads rather than
 * erroring, so the effective type was decided by file ordering in the program
 * instead of by the code.
 */
declare global {
  function detectHandLandmarks(frame: Frame): HandDetectionResult | null;
}

export interface RecognitionResult {
  gloss: string;
  label: string;
  confidence: number;
}

function toLeftRight(result: HandDetectionResult): { left?: Hand; right?: Hand } {
  let left: Hand | undefined;
  let right: Hand | undefined;
  const handedness = result.handedness;
  result.hands.forEach((hand, i) => {
    const name = handedness?.[i]?.[0]?.categoryName ?? 'Left';
    if (name.toLowerCase() === 'left') left = hand;
    else right = hand;
  });
  return { left, right };
}

/**
 * Buffers normalized frames and runs the on-device classifier when a full
 * window is available. Feed `ingest` from the frame processor's runOnJS.
 */
export function useGestureRecognizer() {
  const bufferRef = useRef<Float32Array[]>([]);
  const busyRef = useRef(false);
  // Bumped by reset(); an inference that started before the bump must not
  // repopulate the result card the user just cleared.
  const generationRef = useRef(0);
  const [result, setResult] = useState<RecognitionResult | null>(null);
  const [progress, setProgress] = useState(0);

  const ingest = useCallback((detection: HandDetectionResult) => {
    if (busyRef.current) return;
    if (detection.error) return;
    const { left, right } = toLeftRight(detection);
    if (!left && !right) return; // require at least one hand
    const frame = packHands(left, right);
    normalizeFrame(frame);

    const buffer = bufferRef.current;
    buffer.push(frame);
    if (buffer.length > SEQUENCE_LENGTH) buffer.shift();
    setProgress(buffer.length);

    if (buffer.length >= SEQUENCE_LENGTH) {
      busyRef.current = true;
      const generation = generationRef.current;
      const input = new Float32Array(SEQUENCE_LENGTH * V * C);
      buffer.forEach((f, i) => input.set(f, i * V * C));
      classifier
        .predict(input)
        .then((logits) => {
          if (generationRef.current !== generation) return;
          setResult(topGloss(logits));
        })
        .catch((err) => {
          console.warn('classifier error', err);
        })
        .finally(() => {
          if (generationRef.current === generation) {
            bufferRef.current = [];
            setProgress(0);
          }
          busyRef.current = false;
        });
    }
  }, []);

  const reset = useCallback(() => {
    generationRef.current += 1;
    bufferRef.current = [];
    setProgress(0);
    setResult(null);
  }, []);

  return { ingest, result, progress, reset };
}
