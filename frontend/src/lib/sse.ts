import type { SSEEvent } from '../types';
import { isTauri } from './api';

export interface ChatRequest {
  model: string;
  messages: Array<{ role: string; content: string }>;
  stream: true;
}

const DESKTOP_API = 'http://127.0.0.1:8000';

export async function* streamChat(
  request: ChatRequest,
  signal?: AbortSignal,
): AsyncGenerator<SSEEvent> {
  const base = import.meta.env.VITE_API_URL || (isTauri() ? DESKTOP_API : '');
  const response = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Chat request failed: ${response.status}`);
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      let currentEvent: string | undefined;

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') return;
          yield { event: currentEvent, data };
          currentEvent = undefined;
        } else if (line.trim() === '') {
          currentEvent = undefined;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
