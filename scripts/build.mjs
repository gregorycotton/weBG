import { build } from "esbuild";
import { copyFileSync, mkdirSync } from "node:fs";

mkdirSync("dist", { recursive: true });
await build({
  entryPoints: { index: "src/index.ts", worker: "src/worker.ts" },
  outdir: "dist",
  entryNames: "[name]",
  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2022",
  minify: true,
  splitting: false,
  conditions: ["onnxruntime-web-use-extern-wasm"],
});
for (const file of ["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) {
  copyFileSync(`node_modules/onnxruntime-web/dist/${file}`, `dist/${file}`);
}
