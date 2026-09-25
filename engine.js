// Faint Pull inference engine. Runs MediaPipe object detection and image classification.
// Loaded as a module Web Worker so the camera view stays smooth; the page can also import it
// directly as a fallback if workers aren't available.

let vision = null, fileset = null, base = "", cfg = null;
const tasks = {};          // name -> Promise<task>
const canvases = {};
let lastTs = 0;

function oc(name, w, h){
  let c = canvases[name];
  if (!c) c = canvases[name] = new OffscreenCanvas(w, h);
  if (c.width !== w) c.width = w;
  if (c.height !== h) c.height = h;
  return c;
}
// Brighten with 'screen' blending: lifts shadows, keeps highlights from clipping.
function drawBoosted(ctx, src, sx, sy, sw, sh, w, h, boost){
  ctx.globalCompositeOperation = "source-over"; ctx.globalAlpha = 1;
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, w, h);
  ctx.drawImage(src, sx, sy, sw, sh, 0, 0, w, h);
  if (boost > 0.01){
    ctx.globalCompositeOperation = "screen";
    for (let b = boost; b > 0.01; b -= 1){ ctx.globalAlpha = Math.min(1, b); ctx.drawImage(src, sx, sy, sw, sh, 0, 0, w, h); }
    ctx.globalCompositeOperation = "source-over"; ctx.globalAlpha = 1;
  }
}
function meanLuma(src, w, h){
  const c = oc("lum", 16, 16), x = c.getContext("2d", { willReadFrequently: true });
  x.drawImage(src, 0, 0, w, h, 0, 0, 16, 16);
  const d = x.getImageData(0, 0, 16, 16).data; let s = 0;
  for (let i = 0; i < d.length; i += 4) s += 0.2126*d[i] + 0.7152*d[i+1] + 0.0722*d[i+2];
  return s / 256 / 255;
}

async function init(o){
  base = o.base; cfg = o;
  if (!vision) vision = await import(base + "lib/mediapipe/vision_bundle.mjs");
  if (!fileset) fileset = await vision.FilesetResolver.forVisionTasks(base + "lib/mediapipe/wasm", true);
  await getTask("live");
  return { ok: true };
}
function getTask(name){
  if (!tasks[name]){
    const model = base + "models/" + cfg.models[name];
    // Each task needs its own copy of the Wasm loader module; a query string makes the browser evaluate it again.
    const fs = Object.assign({}, fileset, { wasmLoaderPath: fileset.wasmLoaderPath + "?task=" + name });
    tasks[name] = (name === "classifier"
      ? vision.ImageClassifier.createFromOptions(fs, { baseOptions: { modelAssetPath: model, delegate: "CPU" }, runningMode: "IMAGE", maxResults: 5 })
      : vision.ObjectDetector.createFromOptions(fs, { baseOptions: { modelAssetPath: model, delegate: "CPU" }, runningMode: "VIDEO", scoreThreshold: name === "live" ? 0.35 : 0.3, maxResults: 25 })
    ).catch(e => { delete tasks[name]; throw e; });
  }
  return tasks[name];
}
function ts(){ const t = Math.max(performance.now(), lastTs + 1); lastTs = t; return t; }

// Detect objects in a frame. Boxes come back in the frame's own pixel coordinates.
async function detect({ bitmap, boost, which, maxSide }){
  const w0 = bitmap.width, h0 = bitmap.height, sc = Math.min(1, (maxSide || 512) / Math.max(w0, h0));
  const w = Math.round(w0*sc), h = Math.round(h0*sc), c = oc("det", w, h);
  const lum = meanLuma(bitmap, w0, h0);
  drawBoosted(c.getContext("2d"), bitmap, 0, 0, w0, h0, w, h, boost || 0);
  bitmap.close?.();
  const det = await getTask(which || "live");
  const t0 = performance.now();
  const r = det.detectForVideo(c, ts());
  const ms = performance.now() - t0;
  const out = r.detections.map(d => {
    const b = d.boundingBox, cat = d.categories[0];
    return { cls: cat.categoryName, score: cat.score, bbox: [b.originX/sc, b.originY/sc, b.width/sc, b.height/sc] };
  }).filter(d => d.cls && d.cls !== "???");
  return { dets: out, lum, ms };
}

// Classify one or more square crops of a frame. Returns the top 5 ImageNet classes for the best crop.
// With each=true, returns one result per box; otherwise returns the most confident box.
async function classify({ bitmap, boxes, boost, each }){
  const clf = await getTask("classifier"), c = oc("crop", 256, 256), x = c.getContext("2d");
  const list = [];
  for (const [bx, by, bw, bh] of boxes){
    const side = Math.max(bw, bh) * 1.08, cx = bx + bw/2, cy = by + bh/2;
    drawBoosted(x, bitmap, cx - side/2, cy - side/2, side, side, 256, 256, boost || 0);
    const r = clf.classify(c);
    list.push((r.classifications[0]?.categories || []).map(k => ({ index: k.index, name: k.categoryName, score: k.score })));
  }
  bitmap.close?.();
  if (each) return { list };
  let bi = 0; list.forEach((l, i) => { if ((l[0]?.score || 0) > (list[bi][0]?.score || 0)) bi = i; });
  return { cats: list[bi] || [], box: bi };
}
async function warm({ which }){ await getTask(which); return { ok: true }; }

const api = { init, detect, classify, warm };
export default api;

// Worker wiring
if (typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope){
  self.onmessage = async e => {
    const { id, type, payload } = e.data;
    try { self.postMessage({ id, ok: true, result: await api[type](payload) }); }
    catch(err){ self.postMessage({ id, ok: false, error: String(err && err.message || err) }); }
  };
}
