import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In docker-compose the backend is reachable as http://backend:8000;
// when running on the host it is http://localhost:8000.
const apiTarget = process.env.VITE_API_PROXY_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    // Proxying /api keeps the browser on one origin: no CORS in dev, and SSE just works.
    proxy: { "/api": { target: apiTarget, changeOrigin: true } },
  },
});
