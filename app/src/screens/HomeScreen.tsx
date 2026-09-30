/**
 * HomeScreen — entry point with the two core flows: Gesture->Text and
 * Text->Sign. Styled to the "Group 2133" mockup (mint/teal chips on soft blue).
 */

import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';

import { Icon, type IconName } from '../components/Icon';
import { colors, radius, spacing, type } from '../theme';

interface Props {
  onOpenGesture: () => void;
  onOpenTextToSign: () => void;
  onOpenFingerspell: () => void;
  onOpenLogin: () => void;
  authed: boolean;
  onLogout: () => void;
}

function FeatureCard({
  icon,
  label,
  title,
  subtitle,
  tint,
  onPress,
}: {
  icon: IconName;
  label: string;
  title: string;
  subtitle: string;
  tint: string;
  onPress: () => void;
}) {
  return (
    <TouchableOpacity style={styles.feature} onPress={onPress} activeOpacity={0.85}>
      <View style={[styles.iconChip, { backgroundColor: tint }]}>
        <Icon name={icon} size={30} color={colors.teal} />
      </View>
      <View style={styles.featureBody}>
        <Text style={styles.featureLabel}>{label}</Text>
        <Text style={styles.featureTitle}>{title}</Text>
        <Text style={styles.featureSub}>{subtitle}</Text>
      </View>
      <Text style={styles.chevron}>›</Text>
    </TouchableOpacity>
  );
}

export default function HomeScreen({
  onOpenGesture,
  onOpenTextToSign,
  onOpenFingerspell,
  onOpenLogin,
  authed,
  onLogout,
}: Props) {
  return (
    <View style={styles.screen}>
      <View style={styles.header}>
        <Text style={type.title}>Silent Link</Text>
        <Text style={styles.subtitle}>Speak with your hands. ISL, on-device.</Text>
      </View>

      <View style={styles.statusRow}>
        <View style={styles.statusDot} />
        <Text style={styles.statusText}>Recognition model ready</Text>
        <TouchableOpacity
          style={styles.accountChip}
          onPress={authed ? onLogout : onOpenLogin}
          activeOpacity={0.85}
        >
          <Text style={styles.accountText}>{authed ? 'Log out' : 'Sign in'}</Text>
        </TouchableOpacity>
      </View>

      <FeatureCard
        icon="hand"
        label="Gesture to text"
        title="Sign in, read it out"
        subtitle="Point your camera at a sign and the app names it."
        tint={colors.mint}
        onPress={onOpenGesture}
      />
      <FeatureCard
        icon="text"
        label="Text to sign"
        title="Type a phrase, watch it sign"
        subtitle="The app animates the signed motion from text."
        tint={colors.coral}
        onPress={onOpenTextToSign}
      />
      <FeatureCard
        icon="keys"
        label="Fingerspelling"
        title="Spell letter by letter"
        subtitle="Hold each hand shape to build a word."
        tint={colors.fieldBlue}
        onPress={onOpenFingerspell}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bgTop, padding: spacing.lg },
  header: { marginTop: spacing.xl },
  subtitle: { ...type.body, marginTop: spacing.xs },
  statusRow: { flexDirection: 'row', alignItems: 'center', marginTop: spacing.xl, marginBottom: spacing.sm },
  statusDot: { width: 10, height: 10, borderRadius: 5, backgroundColor: colors.green, marginRight: spacing.xs },
  statusText: { ...type.label },
  accountChip: {
    marginLeft: 'auto',
    backgroundColor: colors.mint,
    borderRadius: 999,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
  },
  accountText: { color: colors.teal, fontWeight: '700', fontSize: 13 },
  feature: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.card,
    borderRadius: radius.md,
    padding: spacing.md,
    marginTop: spacing.md,
  },
  iconChip: { width: 56, height: 56, borderRadius: 16, alignItems: 'center', justifyContent: 'center' },
  featureBody: { flex: 1, marginLeft: spacing.md },
  featureLabel: { ...type.label, fontSize: 12 },
  featureTitle: { ...type.heading, fontSize: 17, marginTop: 2 },
  featureSub: { ...type.caption, marginTop: 4, lineHeight: 18 },
  chevron: { fontSize: 30, color: colors.greyLight, marginLeft: spacing.sm },
});
