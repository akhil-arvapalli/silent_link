/**
 * FingerspellingScreen — per-frame hand-shape letter recognition.
 *
 * Uses the camera + MediaPipe hand landmarker (like GestureScreen) but
 * classifies each frame's hand-shape to an ISL finger-spelling letter. Letters
 * are appended to a word with a short "hold" to commit, and a clear button
 * separates words.
 *
 * Note: the letter classifier is a deterministic stub (model/src/fingerspell.ts)
 * until a real per-frame letter model is trained.
 */

import { useMemo, useRef, useState } from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { Worklets } from 'react-native-worklets-core';

import { classifyLetter } from '../model/fingerspell';
import { colors, radius, type, TOP_BAR_HEIGHT } from '../theme';

const HOLD_FRAMES = 6;

export default function FingerspellingScreen() {
  const device = useCameraDevice('front');
  const { hasPermission, requestPermission } = useCameraPermission();

  const [letter, setLetter] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0);
  const [word, setWord] = useState('');
  const holdRef = useRef(0);
  const lastRef = useRef<string | null>(null);

  const onDetection = useMemo(
    () =>
      Worklets.createRunOnJS((hands: import('../model/normalize').Hand[]) => {
        if (!hands.length) {
          holdRef.current = 0;
          return;
        }
        const best = classifyLetter(hands[0]);
        if (!best) {
          setLetter(null);
          holdRef.current = 0;
          return;
        }
        setLetter(best.letter);
        setConfidence(best.confidence);
        if (best.letter === lastRef.current) {
          holdRef.current += 1;
          if (holdRef.current >= HOLD_FRAMES) {
            holdRef.current = 0;
            setWord((w) => (w + best.letter));
          }
        } else {
          lastRef.current = best.letter;
          holdRef.current = 0;
        }
      }),
    [],
  );

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      const detection = detectHandLandmarks(frame);
      if (detection?.hands?.length) {
        onDetection(detection.hands);
      }
    },
    [onDetection],
  );

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text style={styles.centerTitle}>Silent Link</Text>
        <Text style={styles.hint}>Camera access is needed for fingerspelling.</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant camera access</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.center}>
        <Text style={styles.centerTitle}>No camera available</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive
        pixelFormat="rgb"
        frameProcessor={frameProcessor}
        onError={(e) => console.warn('camera error', e)}
      />

      <View style={styles.overlay}>
        <View style={styles.letterCard}>
          <Text style={styles.letter}>{letter ?? '—'}</Text>
          <Text style={styles.confidence}>{letter ? `${(confidence * 100).toFixed(0)}%` : 'hold a letter'}</Text>
        </View>

        <View style={styles.wordCard}>
          <Text style={styles.word}>{word || 'Your word appears here'}</Text>
        </View>

        <View style={styles.controls}>
          <TouchableOpacity style={styles.button} onPress={() => setWord((w) => w + ' ')}>
            <Text style={styles.buttonText}>Space</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={() => setWord('')}>
            <Text style={styles.buttonText}>Clear</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.bgTop, padding: 24 },
  centerTitle: { ...type.title, textAlign: 'center' },
  hint: { color: colors.teal, fontSize: 16, textAlign: 'center', marginBottom: 16 },
  overlay: { flex: 1, padding: 24, paddingTop: TOP_BAR_HEIGHT, justifyContent: 'space-between' },
  letterCard: {
    backgroundColor: colors.mint,
    borderRadius: radius.md,
    padding: 24,
    alignItems: 'center',
    alignSelf: 'center',
    minWidth: 160,
  },
  letter: { color: colors.inkTeal, fontSize: 88, fontWeight: '800', lineHeight: 96 },
  confidence: { color: colors.teal, fontSize: 15, marginTop: 4 },
  wordCard: {
    backgroundColor: 'rgba(0,0,0,0.55)',
    borderRadius: radius.md,
    padding: 20,
    alignItems: 'center',
    minHeight: 64,
    justifyContent: 'center',
  },
  word: { color: '#fff', fontSize: 26, fontWeight: '700', textAlign: 'center' },
  controls: { flexDirection: 'row', justifyContent: 'space-around', marginTop: 16 },
  button: { backgroundColor: colors.coral, paddingVertical: 12, paddingHorizontal: 20, borderRadius: radius.sm },
  buttonText: { color: colors.inkTeal, fontSize: 16, fontWeight: '700' },
});
