import { useCallback, useRef, useState } from 'react';

import { topGloss } from '../config/glosses';
import { normalizeFrame, packHands, type Hand, type Landmark } from '../model/normalize';
import * as classifier from '../model/classifier';

const SEQUENCE_LENGTH = 40;
const V = 42;
const C = 3;

export interface HandednessCategory {
  categoryName: string;
  score: number;
  displayName: string;
}

export interface HandDetectionResult {
  hands: Hand[];
  handedness?: HandednessCategory[][];
  pose?: Landmark[];
  face?: Landmark[];
  error?: string;
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

    if (buffer.length >= SEQUENCE_LENGTH && !busyRef.current) {
      busyRef.current = true;
      const input = new Float32Array(SEQUENCE_LENGTH * V * C);
      buffer.forEach((f, i) => input.set(f, i * V * C));
      classifier
        .predict(input)
        .then((logits) => {
          setResult(topGloss(logits));
        })
        .catch((err) => {
          console.warn('classifier error', err);
        })
        .finally(() => {
          busyRef.current = false;
          bufferRef.current = [];
          setProgress(0);
        });
    }
  }, []);

  const reset = useCallback(() => {
    bufferRef.current = [];
    setProgress(0);
    setResult(null);
  }, []);

  return { ingest, result, progress, reset };
}
