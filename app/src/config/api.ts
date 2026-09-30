/**
 * Backend API configuration.
 *
 * The backend base URL is set via app.json -> expo.extra.backendUrl. For a
 * local FastAPI server reachable from an emulator/device:
 *   - Android emulator:      http://10.0.2.2:8000
 *   - Physical phone (LAN):  http://<your-machine-ip>:8000
 */
export const BACKEND_URL =
  (process.env.EXPO_PUBLIC_BACKEND_URL as string | undefined) ??
  'http://10.0.2.2:8000';
