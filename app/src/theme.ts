/**
 * Design tokens lifted from the "Group 2133" app mockup.
 *
 * The mockup's language: soft blue surfaces, mint/teal icon chips, one coral
 * primary action, dark teal ink. Rounded 12–24, generous but disciplined.
 */

export const colors = {
  bgTop: '#F5FAFF',
  bgBase: '#EAF4FC',
  card: '#FFFFFF',
  mint: '#CBE8ED',
  teal: '#3A656A',
  green: '#4E9F72',
  coral: '#FF9478',
  coralDark: '#9A452E',
  fieldBlue: '#EBF5FD',
  lightBlue: '#E5EFF7',
  ink: '#111111',
  inkTeal: '#131D22',
  grey: '#4D5559',
  greyLight: '#DAE4EC',
  white: '#FFFFFF',
} as const;

export const radius = {
  sm: 12,
  md: 16,
  lg: 24,
} as const;

export const spacing = {
  xs: 8,
  sm: 12,
  md: 16,
  lg: 24,
  xl: 32,
} as const;

export const type = {
  title: { fontSize: 28, fontWeight: '800' as const, color: colors.ink },
  heading: { fontSize: 20, fontWeight: '700' as const, color: colors.ink },
  body: { fontSize: 15, fontWeight: '400' as const, color: colors.grey },
  label: { fontSize: 13, fontWeight: '500' as const, color: colors.teal },
  caption: { fontSize: 12, fontWeight: '400' as const, color: colors.grey },
} as const;
