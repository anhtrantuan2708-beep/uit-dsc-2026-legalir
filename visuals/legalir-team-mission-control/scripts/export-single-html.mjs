import { readFile, writeFile } from "node:fs/promises";
import { basename, dirname, extname, join, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const dist = join(root, "dist/client");
const htmlPath = join(dist, "index.html");

const mime = {
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
};

const asDataUri = async (path) => {
  const data = await readFile(path);
  return `data:${mime[extname(path)] || "application/octet-stream"};base64,${data.toString("base64")}`;
};

let html = await readFile(htmlPath, "utf8");
const jsUrl = html.match(/<script type="module" crossorigin src="([^"]+)"/)[1];
const cssUrl = html.match(/<link rel="stylesheet" crossorigin href="([^"]+)"/)[1];
const jsPath = join(dist, jsUrl.replace(/^\//, ""));
const cssPath = join(dist, cssUrl.replace(/^\//, ""));

let css = await readFile(cssPath, "utf8");
const assetUrls = [...new Set([...css.matchAll(/url\(([^)]+)\)/g)].map((match) => match[1].replace(/["']/g, "")))];
for (const assetUrl of assetUrls) {
  if (assetUrl.startsWith("data:")) continue;
  const assetPath = join(dirname(cssPath), basename(assetUrl));
  css = css.split(assetUrl).join(await asDataUri(assetPath));
}

let js = await readFile(jsPath, "utf8");
const markPath = join(dist, "assets/legalir-mark.png");
js = js.split("/assets/legalir-mark.png").join(await asDataUri(markPath));
js = js.replaceAll("</script", "<\\/script");

html = html
  .replace(/<script type="module" crossorigin src="[^"]+"><\/script>/, () => `<script type="module">${js}</script>`)
  .replace(/<link rel="stylesheet" crossorigin href="[^"]+">/, () => `<style>${css}</style>`)
  .replace('<html lang="en">', '<html lang="vi">')
  .replace("<title>Prototype</title>", "<title>LegalIR Team Mission Control</title>");

const localOutput = join(root, "legalir_team_mission_control.html");
const shareOutput = resolve(root, "../legalir_team_mission_control.html");
await Promise.all([writeFile(localOutput, html), writeFile(shareOutput, html)]);
console.log(`Exported ${localOutput}`);
console.log(`Exported ${shareOutput}`);
