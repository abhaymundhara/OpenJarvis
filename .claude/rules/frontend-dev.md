# Frontend / Desktop Development

## Stack

- **Frontend**: Vite-based web app in `frontend/`
- **Desktop**: Tauri app (Rust + web frontend) in `desktop/`

## Commands

```bash
# Frontend
cd frontend && npm install && npm run dev

# Desktop (Tauri)
cd desktop && npm install && npm run tauri dev
```

## Notes

- The desktop app wraps the frontend via Tauri, which uses a Rust backend with a webview.
- Frontend changes are reflected in the desktop app automatically during development.
