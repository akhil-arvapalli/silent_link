/**
 * Minimal typed client for the Silent Link backend (FastAPI).
 *
 * Uses React Native's built-in fetch. The auth token is held in memory for the
 * session — no AsyncStorage dependency — so a fresh app start requires login.
 */

import { BACKEND_URL } from '../config/api';

let token: string | null = null;

export function setToken(t: string | null): void {
  token = t;
}

export function getToken(): string | null {
  return token;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(init.headers as Record<string, string> | undefined),
  };
  if (token) headers.Authorization = `Bearer ${token}`;

  let res: Response;
  try {
    res = await fetch(`${BACKEND_URL}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(0, `Cannot reach backend at ${BACKEND_URL}`);
  }

  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = (body as { detail?: string }).detail;
    throw new ApiError(res.status, detail ?? `Request failed (${res.status})`);
  }
  return body as T;
}

export interface TokenResponse {
  token: string;
}

export interface SynthesisResult {
  id: string;
  text: string;
  glosses: string[];
  status: string;
  result: { frames: number; samples: number; glosses: string[] } | null;
  error: string | null;
  created_at: string;
}

export const api = {
  health: () => request<{ status: string }>('/health'),
  signup: (username: string, password: string) =>
    request<TokenResponse>('/auth/signup', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    }),
  login: (username: string, password: string) =>
    request<TokenResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    }),
  synthesize: (text: string) =>
    request<SynthesisResult>('/synthesis', {
      method: 'POST',
      body: JSON.stringify({ text }),
    }),
};
