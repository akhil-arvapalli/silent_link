/**
 * LoginScreen — authenticate against the Silent Link backend (signup/login).
 * Styled to the Group 2133 mockup.
 */

import { useState } from 'react';
import { KeyboardAvoidingView, Platform, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';

import { ApiError, api, setToken } from '../api/client';
import { colors, radius, spacing, type, TOP_BAR_HEIGHT } from '../theme';

interface Props {
  onAuthenticated: () => void;
}

export default function LoginScreen({ onAuthenticated }: Props) {
  const [mode, setMode] = useState<'login' | 'signup'>('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = mode === 'login' ? await api.login(username, password) : await api.signup(username, password);
      setToken(res.token);
      onAuthenticated();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Something went wrong');
    } finally {
      setBusy(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.screen}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <View style={styles.card}>
        <Text style={type.title}>Silent Link</Text>
        <Text style={type.body}>Sign in to sync your signs to the cloud.</Text>

        <TextInput
          style={styles.input}
          placeholder="Username"
          placeholderTextColor={colors.grey}
          autoCapitalize="none"
          value={username}
          onChangeText={setUsername}
        />
        <TextInput
          style={styles.input}
          placeholder="Password"
          placeholderTextColor={colors.grey}
          secureTextEntry
          value={password}
          onChangeText={setPassword}
          onSubmitEditing={submit}
        />

        {error && <Text style={styles.error}>{error}</Text>}

        <TouchableOpacity style={styles.cta} onPress={submit} activeOpacity={0.85} disabled={busy}>
          <Text style={styles.ctaText}>{busy ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Create account'}</Text>
        </TouchableOpacity>

        <TouchableOpacity onPress={() => setMode((m) => (m === 'login' ? 'signup' : 'login'))}>
          <Text style={styles.switch}>
            {mode === 'login' ? 'New here? Create an account' : 'Have an account? Sign in'}
          </Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: colors.bgTop,
    justifyContent: 'center',
    padding: spacing.lg,
    paddingTop: TOP_BAR_HEIGHT + spacing.md,
  },
  card: { backgroundColor: colors.card, borderRadius: radius.lg, padding: spacing.lg },
  input: {
    marginTop: spacing.md,
    backgroundColor: colors.fieldBlue,
    borderRadius: radius.sm,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    fontSize: 15,
    color: colors.ink,
  },
  cta: {
    marginTop: spacing.lg,
    backgroundColor: colors.coral,
    borderRadius: radius.sm,
    paddingVertical: spacing.md,
    alignItems: 'center',
  },
  ctaText: { color: colors.inkTeal, fontSize: 16, fontWeight: '700' },
  switch: { ...type.label, textAlign: 'center', marginTop: spacing.md },
  error: { ...type.caption, color: colors.coralDark, marginTop: spacing.sm },
});
