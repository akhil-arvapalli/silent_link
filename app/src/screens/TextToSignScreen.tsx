/**
 * TextToSignScreen — Phase 2.
 *
 * Type a phrase -> on-device synthesis animates the sign (offline-first). When
 * connected to the backend, the same phrase is also sent to /synthesis to
 * confirm the cloud mapping matches on-device.
 */

import { useState } from 'react';
import { ScrollView, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';

import { api } from '../api/client';
import { SkeletonAvatar } from '../components/SkeletonAvatar';
import { GLOSSES, glossLabel, glossWords } from '../config/glosses';
import { createMotionLibrary } from '../synthesis/motion';
import { GlossNormalizer } from '../synthesis/normalize';
import { stitchGlosses } from '../synthesis/stitch';
import { colors, radius, spacing, type, TOP_BAR_HEIGHT } from '../theme';

const library = createMotionLibrary();
const normalizer = new GlossNormalizer(
  Object.fromEntries(GLOSSES.map((g) => [g, { words: glossWords(g), type: 'phrase' }])),
);

interface Props {
  connected: boolean;
  onOpenLogin: () => void;
}

export default function TextToSignScreen({ connected, onOpenLogin }: Props) {
  const [text, setText] = useState('');
  const [glosses, setGlosses] = useState<string[]>([]);
  const [trajectory, setTrajectory] = useState<Float32Array | null>(null);
  const [totalFrames, setTotalFrames] = useState(0);
  const [cloudStatus, setCloudStatus] = useState<string | null>(null);

  const translate = async () => {
    const seq = normalizer.normalize(text);
    const result = stitchGlosses(seq, library);
    setGlosses(seq);
    setTrajectory(result);
    setTotalFrames(seq.length > 0 ? result.length / (42 * 3) : 0);

    if (connected && text.trim()) {
      setCloudStatus('syncing…');
      try {
        const res = await api.synthesize(text);
        const same = res.glosses.join() === seq.join();
        setCloudStatus(
          same
            ? `Cloud agrees: ${res.glosses.join(', ') || 'no glosses'}`
            : `Cloud: ${res.glosses.join(', ') || 'none'}`,
        );
      } catch {
        setCloudStatus('cloud unavailable');
      }
    } else {
      setCloudStatus(null);
    }
  };

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <Text style={type.title}>Type to sign</Text>
      <Text style={type.body}>
        Enter a phrase and Silent Link turns it into a signed animation, one hand-motion at a time.
      </Text>

      <View style={styles.card}>
        <TextInput
          style={styles.input}
          placeholder="e.g. good morning, please, i love you"
          placeholderTextColor={colors.grey}
          value={text}
          onChangeText={setText}
          returnKeyType="done"
          onSubmitEditing={translate}
        />
        <TouchableOpacity style={styles.cta} onPress={translate} activeOpacity={0.85}>
          <Text style={styles.ctaText}>Translate</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.statusRow}>
        {connected ? (
          <Text style={styles.connected}>● Synced to cloud</Text>
        ) : (
          <TouchableOpacity onPress={onOpenLogin}>
            <Text style={styles.offline}>○ Offline — sign in to sync</Text>
          </TouchableOpacity>
        )}
        {cloudStatus ? <Text style={styles.cloudStatus}>{cloudStatus}</Text> : null}
      </View>

      {glosses.length > 0 && (
        <View style={styles.chips}>
          {glosses.map((g) => (
            <View key={g} style={styles.chip}>
              <Text style={styles.chipText}>{glossLabel(g)}</Text>
            </View>
          ))}
        </View>
      )}

      {trajectory && totalFrames > 0 ? (
        <View style={styles.stage}>
          <View style={styles.stageCard}>
            <SkeletonAvatar
              data={trajectory}
              totalFrames={totalFrames}
              width={300}
              height={280}
              scale={100}
            />
            <Text style={styles.hint}>Signed by the on-device skeleton</Text>
          </View>
        </View>
      ) : (
        <View style={styles.stage}>
          <View style={[styles.stageCard, styles.stageEmpty]}>
            <Text style={styles.emptyTitle}>The signed phrase appears here</Text>
            <Text style={styles.emptyHint}>
              {glosses.length === 0 && text ? 'Nothing in the vocabulary matched yet.' : ''}
            </Text>
          </View>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bgTop },
  content: { padding: spacing.lg, paddingTop: TOP_BAR_HEIGHT + spacing.md, paddingBottom: 48 },
  card: { marginTop: spacing.xl, backgroundColor: colors.card, borderRadius: radius.md, padding: spacing.md },
  input: {
    borderWidth: 0,
    backgroundColor: colors.fieldBlue,
    borderRadius: radius.sm,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    fontSize: 15,
    color: colors.ink,
  },
  cta: {
    marginTop: spacing.md,
    backgroundColor: colors.coral,
    borderRadius: radius.sm,
    paddingVertical: spacing.md,
    alignItems: 'center',
  },
  ctaText: { color: colors.inkTeal, fontSize: 16, fontWeight: '700' },
  statusRow: { flexDirection: 'row', alignItems: 'center', flexWrap: 'wrap', marginTop: spacing.sm },
  connected: { ...type.label, color: colors.green },
  offline: { ...type.label, color: colors.grey },
  cloudStatus: { ...type.caption, marginLeft: spacing.sm },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.xs, marginTop: spacing.md },
  chip: { backgroundColor: colors.mint, borderRadius: 999, paddingHorizontal: spacing.md, paddingVertical: spacing.xs },
  chipText: { color: colors.teal, fontWeight: '600', fontSize: 14 },
  stage: { marginTop: spacing.xl, alignItems: 'center' },
  stageCard: {
    width: '100%',
    backgroundColor: colors.mint,
    borderRadius: radius.lg,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.lg,
    minHeight: 320,
  },
  stageEmpty: { backgroundColor: colors.lightBlue, justifyContent: 'center' },
  emptyTitle: { ...type.heading, textAlign: 'center' },
  emptyHint: { ...type.caption, textAlign: 'center', marginTop: spacing.xs },
  hint: { ...type.caption, textAlign: 'center', marginTop: spacing.sm },
});
