import { useState, useRef, useEffect } from 'react';
import { Search, Cpu, X } from 'lucide-react';
import { useAppStore } from '../lib/store';

export function CommandPalette() {
  const [query, setQuery] = useState('');
  const [selectedIdx, setSelectedIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const models = useAppStore((s) => s.models);
  const selectedModel = useAppStore((s) => s.selectedModel);
  const setSelectedModel = useAppStore((s) => s.setSelectedModel);
  const setCommandPaletteOpen = useAppStore((s) => s.setCommandPaletteOpen);

  const filtered = query
    ? models.filter((m) =>
        m.id.toLowerCase().includes(query.toLowerCase()),
      )
    : models;

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    setSelectedIdx(0);
  }, [query]);

  const handleSelect = (modelId: string) => {
    setSelectedModel(modelId);
    setCommandPaletteOpen(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      setCommandPaletteOpen(false);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIdx((i) => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === 'Enter' && filtered.length > 0) {
      e.preventDefault();
      handleSelect(filtered[selectedIdx].id);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]"
      onClick={() => setCommandPaletteOpen(false)}
    >
      {/* Backdrop */}
      <div className="fixed inset-0" style={{ background: 'rgba(0,0,0,0.5)' }} />

      {/* Palette */}
      <div
        className="relative w-full max-w-lg rounded-xl overflow-hidden"
        style={{
          background: 'var(--color-surface)',
          border: '1px solid var(--color-border)',
          boxShadow: 'var(--shadow-lg)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search input */}
        <div
          className="flex items-center gap-3 px-4 py-3"
          style={{ borderBottom: '1px solid var(--color-border)' }}
        >
          <Search size={18} style={{ color: 'var(--color-text-tertiary)' }} />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search models..."
            className="flex-1 bg-transparent outline-none text-sm"
            style={{ color: 'var(--color-text)' }}
          />
          <button
            onClick={() => setCommandPaletteOpen(false)}
            className="p-1 rounded cursor-pointer"
            style={{ color: 'var(--color-text-tertiary)' }}
          >
            <X size={16} />
          </button>
        </div>

        {/* Results */}
        <div className="max-h-[300px] overflow-y-auto py-2">
          {filtered.length === 0 ? (
            <div className="px-4 py-6 text-center text-sm" style={{ color: 'var(--color-text-tertiary)' }}>
              {models.length === 0 ? 'No models available — is the server running?' : 'No matching models'}
            </div>
          ) : (
            filtered.map((model, idx) => {
              const isActive = model.id === selectedModel;
              const isSelected = idx === selectedIdx;
              return (
                <button
                  key={model.id}
                  onClick={() => handleSelect(model.id)}
                  className="flex items-center gap-3 w-full px-4 py-2.5 text-left transition-colors cursor-pointer"
                  style={{
                    background: isSelected ? 'var(--color-bg-secondary)' : 'transparent',
                  }}
                  onMouseEnter={() => setSelectedIdx(idx)}
                >
                  <Cpu size={16} style={{ color: isActive ? 'var(--color-accent)' : 'var(--color-text-tertiary)' }} />
                  <div className="flex-1 min-w-0">
                    <div
                      className="text-sm truncate"
                      style={{
                        color: isActive ? 'var(--color-accent)' : 'var(--color-text)',
                        fontWeight: isActive ? 500 : 400,
                      }}
                    >
                      {model.id}
                    </div>
                  </div>
                  {isActive && (
                    <span className="text-[10px] px-2 py-0.5 rounded-full" style={{
                      background: 'var(--color-accent-subtle)',
                      color: 'var(--color-accent)',
                    }}>
                      Active
                    </span>
                  )}
                </button>
              );
            })
          )}
        </div>

        {/* Footer */}
        <div
          className="flex items-center gap-4 px-4 py-2 text-[11px]"
          style={{ borderTop: '1px solid var(--color-border)', color: 'var(--color-text-tertiary)' }}
        >
          <span><kbd className="font-mono">↑↓</kbd> Navigate</span>
          <span><kbd className="font-mono">Enter</kbd> Select</span>
          <span><kbd className="font-mono">Esc</kbd> Close</span>
        </div>
      </div>
    </div>
  );
}
