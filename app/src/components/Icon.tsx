/**
 * Icon — minimal flat line icons drawn with react-native-skia, matching the
 * Group 2133 mockup's flat vector-icon language (teal on mint chips).
 *
 * Each icon is a hand-drawn 48x48 stroke path; no emoji, no icon-font dep.
 */

import { Canvas, Group, Path, Skia, type SkPath } from '@shopify/react-native-skia';

export type IconName = 'hand' | 'text' | 'keys';

const PATHS: Record<IconName, string> = {
  // A simple open hand: palm + four fingers + thumb.
  hand:
    'M14 26v-8a3 3 0 0 1 6 0v6m0-9v-2a3 3 0 0 1 6 0v8m0-7v-1a3 3 0 0 1 6 0v8m-8 5h8a3 3 0 0 0 3-3v-5a3 3 0 0 1 6 0v7a11 11 0 0 1-11 11h-6a11 11 0 0 1-9-4l-6-7a3 3 0 0 1 4-4z',
  // A text/pen: an angled sheet with a writing stroke.
  text: 'M14 10h20a2 2 0 0 1 2 2v24a2 2 0 0 1-2 2H14a2 2 0 0 1-2-2V12a2 2 0 0 1 2-2zm2 8h16m-16 6h10m-10 6h14',
  // Keyboard / typing keys.
  keys:
    'M12 14h24a4 4 0 0 1 4 4v12a4 4 0 0 1-4 4H12a4 4 0 0 1-4-4V18a4 4 0 0 1 4-4zm6 10h.01M24 24h.01M30 24h.01M18 30h.01M24 30h.01M30 30h.01',
};

interface Props {
  name: IconName;
  size?: number;
  color?: string;
  strokeWidth?: number;
}

export function Icon({ name, size = 32, color = '#3A656A', strokeWidth = 2.4 }: Props) {
  const path: SkPath = Skia.Path.MakeFromSVGString(PATHS[name])!;
  const scale = size / 48;
  path.transform([scale, 0, 0, 0, scale, 0, 0, 0, 1]);
  return (
    <Canvas style={{ width: size, height: size }}>
      <Group>
        <Path path={path} style="stroke" strokeWidth={strokeWidth} strokeCap="round" strokeJoin="round" color={color} />
      </Group>
    </Canvas>
  );
}
