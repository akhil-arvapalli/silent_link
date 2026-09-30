import { StatusBar } from 'expo-status-bar';
import { useState } from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';

import { getToken, setToken } from './src/api/client';
import FingerspellingScreen from './src/screens/FingerspellingScreen';
import GestureScreen from './src/screens/GestureScreen';
import HomeScreen from './src/screens/HomeScreen';
import LoginScreen from './src/screens/LoginScreen';
import TextToSignScreen from './src/screens/TextToSignScreen';
import { colors, spacing } from './src/theme';

type Route = 'home' | 'gesture' | 'text' | 'spell' | 'login';

const TITLES: Record<Route, string> = {
  home: 'Silent Link',
  gesture: 'Gesture to text',
  text: 'Text to sign',
  spell: 'Fingerspelling',
  login: 'Account',
};

export default function App() {
  const [route, setRoute] = useState<Route>('home');
  const [authed, setAuthed] = useState(!!getToken());

  const navigate = (r: Route) => setRoute(r);

  const handleAuthenticated = () => {
    setAuthed(true);
    navigate('home');
  };

  const handleLogout = () => {
    setToken(null);
    setAuthed(false);
    navigate('home');
  };

  const screen = (() => {
    switch (route) {
      case 'gesture':
        return <GestureScreen />;
      case 'text':
        return (
          <TextToSignScreen connected={authed} onOpenLogin={() => navigate('login')} />
        );
      case 'spell':
        return <FingerspellingScreen />;
      case 'login':
        return <LoginScreen onAuthenticated={handleAuthenticated} />;
      default:
        return (
          <HomeScreen
            onOpenGesture={() => navigate('gesture')}
            onOpenTextToSign={() => navigate('text')}
            onOpenFingerspell={() => navigate('spell')}
            onOpenLogin={() => navigate('login')}
            authed={authed}
            onLogout={handleLogout}
          />
        );
    }
  })();

  return (
    <View style={styles.root}>
      <StatusBar style="dark" />
      {screen}

      {route !== 'home' && (
        <View style={styles.topBar}>
          <TouchableOpacity onPress={() => navigate('home')} style={styles.back}>
            <Text style={styles.backText}>‹ Home</Text>
          </TouchableOpacity>
          <Text style={styles.title}>{TITLES[route]}</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bgTop },
  topBar: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    paddingTop: 56,
    paddingBottom: spacing.sm,
    zIndex: 10,
  },
  back: { padding: spacing.xs },
  backText: { color: colors.teal, fontSize: 16, fontWeight: '700' },
  title: { marginLeft: spacing.md, color: colors.ink, fontSize: 17, fontWeight: '700' },
});
