import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";

const root = resolve(import.meta.dirname, "..");
const { version } = JSON.parse(await readFile(join(root, "package.json"), "utf8"));
const release = join(root, "release", `v${version}`);
const name = `webg-${version}`;
const models = ["birefnet-lite-512", "yolos-tiny-416"];
const temporary = await mkdtemp(join(tmpdir(), "webg-release-"));

function run(command, args, cwd = root) {
  const result = spawnSync(command, args, { cwd, stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} exited with ${result.status}`);
}

async function sha256(path) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

try {
  run("npm", ["run", "build"]);
  run("npm", ["run", "verify:model"]);
  run("npm", ["pack", "--pack-destination", temporary, "--cache", join(temporary, "npm-cache")]);

  const staging = join(temporary, "browser");
  const webg = join(staging, "webg");
  const modelDir = join(staging, "models");
  await mkdir(webg, { recursive: true });
  await mkdir(modelDir);
  const packageFile = `${name}.tgz`;
  run("tar", ["-xzf", join(temporary, packageFile), "-C", webg, "--strip-components=1"]);
  for (const model of models) {
    for (const extension of ["json", "onnx"]) {
      const file = `${model}.${extension}`;
      await copyFile(join(root, "models", file), join(modelDir, file));
    }
  }

  const browserFile = `${name}-browser.tar.gz`;
  run("tar", ["-czf", join(temporary, browserFile), "--uid", "0", "--gid", "0", "--uname", "root", "--gname", "root", "webg", "models"], staging);
  await mkdir(release, { recursive: true });
  await mkdir(join(release, "models"), { recursive: true });
  for (const file of [packageFile, browserFile]) await copyFile(join(temporary, file), join(release, file));
  for (const model of models) {
    for (const extension of ["json", "onnx"]) {
      const file = `${model}.${extension}`;
      await copyFile(join(modelDir, file), join(release, "models", file));
    }
  }
  const checksums = [];
  for (const file of [packageFile, browserFile, ...models.flatMap(model => ["json", "onnx"].map(extension => `models/${model}.${extension}`))]) checksums.push(`${await sha256(join(release, file))}  ${file}`);
  await writeFile(join(release, "SHA256SUMS"), `${checksums.join("\n")}\n`);
  console.log(`Prepared ${release}`);
} finally {
  await rm(temporary, { recursive: true, force: true });
}
