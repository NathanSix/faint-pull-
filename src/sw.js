// Faint Pull service worker: makes the app open instantly and work offline.
// The app shell is cached per build; models are cached once and kept.
const VERSION = "__BUILD__";
const APP = "fp-app-" + VERSION, LIB = "fp-lib-__LIBHASH__", MODELS = "fp-models-v1", FONTS = "fp-fonts-v1";
const CORE = ["./", "index.html", "engine.js", "manifest.webmanifest", "icons/icon-192.png", "icons/apple-touch-icon.png"];
// The 12 MB MediaPipe runtime has its own cache, so app updates don't re-download it.
const CORE_LIB = ["lib/mediapipe/vision_bundle.mjs", "lib/mediapipe/wasm/vision_wasm_module_internal.js", "lib/mediapipe/wasm/vision_wasm_module_internal.wasm"];
const CORE_MODELS = ["models/efficientdet_lite0_int8.tflite"];

self.addEventListener("install", e => {
  e.waitUntil((async () => {
    await (await caches.open(APP)).addAll(CORE);
    const l = await caches.open(LIB);
    for (const u of CORE_LIB) if (!(await l.match(u))) await l.add(u);
    const m = await caches.open(MODELS);
    for (const u of CORE_MODELS) if (!(await m.match(u))) await m.add(u);
    await self.skipWaiting();
  })());
});
self.addEventListener("activate", e => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if ((k.startsWith("fp-app-") && k !== APP) || (k.startsWith("fp-lib-") && k !== LIB)) await caches.delete(k);
    await self.clients.claim();
  })());
});
self.addEventListener("fetch", e => {
  const req = e.request; if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin === location.origin){
    if (req.mode === "navigate" || url.pathname.endsWith("/index.html")) e.respondWith(networkFirst(req, APP));
    else if (url.pathname.includes("/models/")) e.respondWith(cacheFirst(req, MODELS));
    else if (url.pathname.includes("/lib/")) e.respondWith(cacheFirst(req, LIB));
    else e.respondWith(cacheFirst(req, APP));
  } else if (url.hostname === "fonts.googleapis.com" || url.hostname === "fonts.gstatic.com"){
    e.respondWith(staleWhileRevalidate(req, FONTS));
  }
});
async function cacheFirst(req, name){
  const c = await caches.open(name);
  // the Wasm loader is imported with a ?task= query so it runs once per task; the file itself is the same
  const hit = await c.match(req, { ignoreSearch: true }); if (hit) return hit;
  const res = await fetch(req);
  if (res.ok) c.put(req.url.split("?")[0], res.clone());
  return res;
}
async function networkFirst(req, name){
  const c = await caches.open(name);
  try {
    const res = await Promise.race([fetch(req), new Promise((_, rej) => setTimeout(() => rej(new Error("slow")), 4000))]);
    if (res.ok) c.put("index.html", res.clone());
    return res;
  } catch(e){ return (await c.match("index.html")) || (await c.match("./")) || Response.error(); }
}
async function staleWhileRevalidate(req, name){
  const c = await caches.open(name), hit = await c.match(req);
  const net = fetch(req).then(res => { if (res.ok) c.put(req, res.clone()); return res; }).catch(() => hit);
  return hit || net;
}
