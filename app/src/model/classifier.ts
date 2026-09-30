import { Asset } from 'expo-asset';
import * as ort from 'onnxruntime-react-native';

const SEQUENCE_LENGTH = 40;
const V = 42;
const C = 3;

let sessionPromise: Promise<ort.InferenceSession> | null = null;

async function getSession(): Promise<ort.InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const asset = Asset.fromModule(require('../../assets/model.onnx'));
      await asset.downloadAsync();
      const uri = asset.localUri ?? asset.uri;
      return ort.InferenceSession.create(uri, { executionProviders: ['cpu'] });
    })();
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
  const feeds: Record<string, ort.Tensor> = { landmarks: input };
  const results = await session.run(feeds);
  const logits = results.logits;
  if (Array.isArray(logits)) {
    return Array.from(logits);
  }
  return Array.from(logits.data as Float32Array);
}

export async function disposeSession(): Promise<void> {
  if (sessionPromise) {
    const session = await sessionPromise;
    await session.release();
    sessionPromise = null;
  }
}
