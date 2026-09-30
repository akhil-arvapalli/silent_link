import { useMemo, useState } from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
  type Frame,
} from 'react-native-vision-camera';
import { Worklets } from 'react-native-worklets-core';

import { GLOSSES } from '../config/glosses';
import { useGestureRecognizer, type HandDetectionResult } from '../hooks/useGestureRecognizer';
import { colors, radius, spacing, type } from '../theme';

declare global {
  function detectHandLandmarks(frame: Frame): HandDetectionResult | null;
}

export default function GestureScreen() {
  const device = useCameraDevice('front');
  const { hasPermission, requestPermission } = useCameraPermission();
  const [facing, setFacing] = useState<'front' | 'back'>('front');
  const activeDevice = useCameraDevice(facing);

  const { ingest, result, progress, reset } = useGestureRecognizer();

  const onResult = useMemo(
    () =>
      Worklets.createRunOnJS((detection: HandDetectionResult) => {
        ingest(detection);
      }),
    [ingest],
  );

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      const detection = detectHandLandmarks(frame);
      if (detection?.hands?.length) {
        onResult(detection);
      }
    },
    [onResult],
  );

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text style={styles.centerTitle}>Silent Link</Text>
        <Text style={styles.hint}>Camera access is needed to recognize ISL gestures.</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant camera access</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!activeDevice) {
    return (
      <View style={styles.center}>
        <Text style={styles.centerTitle}>No camera available</Text>
        <Text style={styles.hint}>No camera available on this device.</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={activeDevice}
        isActive
        pixelFormat="rgb"
        frameProcessor={frameProcessor}
        onError={(e) => console.warn('camera error', e)}
      />

      <View style={styles.overlay}>
        <Text style={styles.title}>Silent Link</Text>
        <Text style={styles.counter}>
          {progress}/{40}
        </Text>

        <View style={styles.progressTrack}>
          <View style={[styles.progressFill, { width: `${(progress / 40) * 100}%` }]} />
        </View>

        {result ? (
          <View style={styles.resultCard}>
            <Text style={styles.resultLabel}>{result.label}</Text>
            <Text style={styles.resultSub}>
              {result.gloss} · {(result.confidence * 100).toFixed(0)}%
            </Text>
          </View>
        ) : (
          <View style={styles.resultCard}>
            <Text style={styles.resultHint}>Hold a sign in view</Text>
          </View>
        )}

        <View style={styles.controls}>
          <TouchableOpacity
            style={styles.button}
            onPress={() => setFacing((f) => (f === 'front' ? 'back' : 'front'))}
          >
            <Text style={styles.buttonText}>Flip camera</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={reset}>
            <Text style={styles.buttonText}>Clear</Text>
          </TouchableOpacity>
        </View>

        <Text style={styles.vocab}>{GLOSSES.length} glosses loaded</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.bgTop, padding: 24 },
  overlay: { flex: 1, padding: 24, justifyContent: 'space-between' },
  title: { color: '#fff', fontSize: 28, fontWeight: '800', textAlign: 'center' },
  centerTitle: { ...type.title, textAlign: 'center' },
  counter: { color: '#fff', fontSize: 18, textAlign: 'center', marginTop: 8 },
  progressTrack: { height: 8, borderRadius: 4, backgroundColor: '#333', overflow: 'hidden', marginTop: 8 },
  progressFill: { height: '100%', backgroundColor: colors.coral },
  resultCard: { backgroundColor: colors.mint, borderRadius: radius.md, padding: 20, alignItems: 'center' },
  resultLabel: { color: colors.inkTeal, fontSize: 40, fontWeight: '800' },
  resultSub: { color: colors.teal, fontSize: 16, marginTop: 4 },
  resultHint: { color: colors.teal, fontSize: 18 },
  controls: { flexDirection: 'row', justifyContent: 'space-around', marginTop: 16 },
  button: { backgroundColor: colors.coral, paddingVertical: 12, paddingHorizontal: 20, borderRadius: radius.sm },
  buttonText: { color: colors.inkTeal, fontSize: 16, fontWeight: '700' },
  hint: { color: colors.teal, fontSize: 16, textAlign: 'center', marginBottom: 16 },
  vocab: { color: colors.grey, fontSize: 12, textAlign: 'center', marginTop: 12 },
});
