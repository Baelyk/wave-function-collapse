import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";

// https://vitejs.dev/config/
export default defineConfig(async () => ({
	base: "wave-function-collapse",
	plugins: [wasm(), topLevelAwait()],
}));
