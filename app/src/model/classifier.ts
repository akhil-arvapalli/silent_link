import { Asset } from 'expo-asset';
import * as ort from 'onnxruntime-react-native';

import { C, V } from './normalize';

/** Frames per inference window. Fixed by the exported graph's input shape. */
export const SEQUENCE_LENGTH = 40;

/** Tensor name the exported graph declares for its input. */
const INPUT_NAME = 'landmarks';
/** Tensor name the exported graph declares for its output. */
const OUTPUT_NAME = 'logits';

let sessionPromise: Promise<ort.InferenceSession> | null = null;

async function getSession(): Promise<ort.InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const asset = Asset.fromModule(require('../../assets/model.onnx'));
      await asset.downloadAsync();
      const uri = asset.localUri ?? asset.uri;
      return ort.InferenceSession.create(uri, { executionProviders: ['cpu'] });
    })().catch((err: unknown) => {
      // A failed load must not be cached. The promise is assigned before it
      // settles, so without this reset one transient failure (asset copy, ORT
      // native lib load) leaves every later predict() rejecting instantly and
      // gesture recognition dead for the rest of the session.
      sessionPromise = null;
      throw err;
    });
  }
  return sessionPromise;
}

/**
 * Run the ST-GCN classifier on a normalized (T, 42, 3) sequence.
 * `sequence` is a Float32Array of length T*V*C in row-major (T, V, C) order.
 * Returns the raw logits (length = number of glosses).
 */
export async function predict(sequence: Float32Array): Promise<number[]> {
  if (sequence.length !== SEQUENCE_LENGTH * V * C) {
    throw new Error(`Expected ${SEQUENCE_LENGTH * V * C} floats, got ${sequence.length}`);
  }
  const session = await getSession();
  const input = new ort.Tensor('float32', sequence, [1, SEQUENCE_LENGTH, V, C]);
  try {
    const results = await session.run({ [INPUT_NAME]: input });
    const logits = results[OUTPUT_NAME];
    // `results[OUTPUT_NAME]` is a single Tensor: onnxruntime-common types
    // OnnxValueMapType values as OnnxValue = Tensor. No array case exists, and
    // an Array.isArray branch here widened the result to `any[]`.
    const values = Array.from(logits.data as Float32Array);
    logits.dispose();
    return values;
  } finally {
    // Releases the native tensor and its 5040-float buffer. Inference happens
    // once per 40-frame window, so without this it accumulates for the life of
    // the (never-released) module-level session.
    input.dispose();
  }
}

export async function disposeSession(): Promise<void> {
  if (sessionPromise) {
    const session = await sessionPromise;
    await session.release();
    sessionPromise = null;
  }
}
