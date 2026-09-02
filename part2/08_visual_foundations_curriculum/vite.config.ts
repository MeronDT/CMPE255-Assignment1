import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  // relative base so the built site works from any GitHub Pages project path
  // (https://<user>.github.io/<repo>/08_visual_foundations_curriculum/dist/)
  base: './',
  plugins: [react()],
})
