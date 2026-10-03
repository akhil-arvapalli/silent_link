/**
 * Backend API configuration.
 *
 * The base URL comes from the EXPO_PUBLIC_BACKEND_URL environment variable,
 * which Expo inlines at bundle time (see .env.example). For a local FastAPI
 * server reachable from an emulator/device:
 *   - Android emulator:      http://10.0.2.2:8000
 *   - Physical phone (LAN):  http://<your-machine-ip>:8000
 */
export const BACKEND_URL =
  (process.env.EXPO_PUBLIC_BACKEND_URL as string | undefined) ??
  'http://10.0.2.2:8000';
