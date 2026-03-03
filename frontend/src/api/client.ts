import type { ModelInfo, SavingsData, ServerInfo } from '../types';

const BASE = import.meta.env.VITE_API_URL || '';  // relative to same origin by default

export async function fetchModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${BASE}/v1/models`);
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.status}`);
  const data = await res.json();
  return data.data || [];
}

export async function fetchSavings(): Promise<SavingsData> {
  const res = await fetch(`${BASE}/v1/savings`);
  if (!res.ok) throw new Error(`Failed to fetch savings: ${res.status}`);
  return res.json();
}

export async function fetchServerInfo(): Promise<ServerInfo> {
  const res = await fetch(`${BASE}/v1/info`);
  if (!res.ok) throw new Error(`Failed to fetch server info: ${res.status}`);
  return res.json();
}

export interface TranscriptionResult {
  text: string;
  language: string | null;
  confidence: number | null;
  duration_seconds: number;
}

export interface SpeechHealth {
  available: boolean;
  backend?: string;
  reason?: string;
}

export async function transcribeAudio(audioBlob: Blob, filename = 'recording.webm'): Promise<TranscriptionResult> {
  const formData = new FormData();
  formData.append('file', audioBlob, filename);
  const res = await fetch(`${BASE}/v1/speech/transcribe`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(`Transcription failed: ${res.status}`);
  return res.json();
}

export async function fetchSpeechHealth(): Promise<SpeechHealth> {
  const res = await fetch(`${BASE}/v1/speech/health`);
  if (!res.ok) return { available: false };
  return res.json();
}
