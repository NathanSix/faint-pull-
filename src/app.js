(() => {
"use strict";
const CONFIG = {
  models: { live: "efficientdet_lite0_int8.tflite", liveGpu: "efficientdet_lite0_fp16.tflite", photo: "efficientdet_lite2_fp16.tflite", classifier: "efficientnet_lite2_fp32.tflite" },
  liveMaxSide: 384,     // detector input for the live camera (EfficientDet-Lite0 itself works at 320 px)
  photoMaxSide: 1024,
  minDetectGap: 40      // ms between live detections, so the phone doesn't run hot
};
const BASE = new URL(".", location.href).href;
const G = 6.674e-11;
const $ = id => document.getElementById(id);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const store = {
  get(k, d){ try{ const v = localStorage.getItem("fp_"+k); return v == null ? d : JSON.parse(v); }catch(e){ return d; } },
  set(k, v){ try{ localStorage.setItem("fp_"+k, JSON.stringify(v)); }catch(e){} }
};

// ---------- catalog: name -> [typical mass kg, largest dimension m] ----------
const CAT = {
 "person":[70,1.7],"bicycle":[13,1.7],"car":[1500,4.5],"motorcycle":[200,2.1],"airplane":[40000,35],"bus":[12000,12],"train":[100000,25],"truck":[9000,7],"boat":[1000,6],
 "traffic light":[20,1],"fire hydrant":[70,0.8],"stop sign":[5,0.75],"parking meter":[30,1.5],"bench":[25,1.5],"bird":[0.3,0.25],"cat":[4.5,0.45],"dog":[25,0.7],
 "horse":[500,2.4],"sheep":[70,1.2],"cow":[700,2.2],"elephant":[5000,3.2],"bear":[250,1.8],"zebra":[350,2.3],"giraffe":[1000,5],"backpack":[3,0.45],"umbrella":[0.5,0.9],
 "handbag":[1,0.3],"tie":[0.05,0.5],"suitcase":[5,0.65],"frisbee":[0.18,0.27],"skis":[3,1.7],"snowboard":[4,1.5],"sports ball":[0.43,0.22],"kite":[0.3,1],
 "baseball bat":[0.9,0.84],"baseball glove":[0.6,0.3],"skateboard":[3,0.8],"surfboard":[6,2.2],"tennis racket":[0.3,0.68],"bottle":[0.8,0.25],"wine glass":[0.2,0.2],
 "cup":[0.35,0.1],"fork":[0.05,0.19],"knife":[0.08,0.22],"spoon":[0.04,0.17],"bowl":[0.4,0.15],"banana":[0.12,0.19],"apple":[0.18,0.08],"sandwich":[0.2,0.12],
 "orange":[0.14,0.08],"broccoli":[0.3,0.15],"carrot":[0.07,0.18],"hot dog":[0.1,0.17],"pizza":[0.8,0.35],"donut":[0.07,0.09],"cake":[1.2,0.25],"chair":[7,0.9],
 "couch":[45,2],"potted plant":[3,0.5],"bed":[60,2],"dining table":[30,1.5],"toilet":[40,0.75],"tv":[12,1],"laptop":[1.8,0.33],"mouse":[0.1,0.11],"remote":[0.15,0.18],
 "keyboard":[0.8,0.44],"cell phone":[0.18,0.14],"microwave":[13,0.5],"oven":[60,0.8],"toaster":[2,0.3],"sink":[10,0.6],"refrigerator":[80,1.75],"book":[0.5,0.24],
 "clock":[0.8,0.3],"vase":[1.2,0.3],"scissors":[0.1,0.2],"teddy bear":[0.4,0.35],"hair drier":[0.6,0.25],"toothbrush":[0.02,0.19],"bowling ball":[7.3,0.22]
};
// ImageNet class index -> [catalog name, mass kg, size m]. Anything unlisted keeps its ImageNet name with an unknown mass.
const IMN_SRC = `__IMN_SRC__`;
const IMN = {};
IMN_SRC.split("|").forEach(e => { const p = e.trim().split(" "), i = +p[0], sz = +p.pop(), m = +p.pop(), n = p.slice(1).join(" "); IMN[i] = n; if (!CAT[n]) CAT[n] = [m, sz]; });
for (let i = 151; i <= 268; i++) IMN[i] = "dog";
for (let i = 281; i <= 285; i++) IMN[i] = "cat";
const IN_LABELS = `__IN_LABELS__`.split("|");
// Detector classes generic enough that the identifier can name them more precisely
const REFINABLE = new Set(["bottle","cup","bowl","wine glass","book","chair","couch","dining table","tv","laptop","cell phone","remote","clock","vase","sports ball","handbag","backpack","suitcase","keyboard","oven","microwave","refrigerator","sink","toaster","bed","bench","potted plant","umbrella","tennis racket"]);
const REFS = [
  ["a bacterium", 1e-14, "bacterium"],["a red blood cell", 2.6e-13, "blood cell"],["a dust mote", 5e-12, "dust"],
  ["a pollen grain", 8e-11, "pollen"],["a grain of fine sand", 1.4e-8, "sand"],["a grain of salt", 6e-7, "salt"],["a mosquito", 2.5e-5, "mosquito"]
];

const S = {
  mode: "demo", objs: [], sel: null, nextId: 1,
  W: 1200, H: 900, f: 0,
  you: store.get("you", 70), lat: store.get("lat", 40.71), alt: store.get("alt", 10), camH: store.get("camH", 1.4),
  stream: null, track: null, loopId: 0, frozen: false, photo: null, pending: null,
  boost: 0, autoBoost: store.get("autoBoost", true), manualBoost: store.get("boost", 0.6), torch: false,
  fps: 0, inferMs: 0, lastPanel: 0, lastRank: 0, liveReady: false,
  motion: { mag:null, ema:null, buf:[], samples:null, up:null, dn:null },
  beta: null, gamma: null
};

const media = $("media"), cv = $("cv"), ctx = cv.getContext("2d"), ov = $("ov"), octx = ov.getContext("2d"), vid = $("vid");
const inFrame = (() => { try { return window.self !== window.top; } catch(e){ return true; } })();
if (inFrame) $("frameNote").hidden = false;

// ---------- inference engine (Web Worker, with an on-page fallback) ----------
const Engine = {
  worker: null, direct: null, ready: null, n: 0, pend: {},
  start(){
    if (!this.ready) this.ready = (async () => {
      try { await this.startWorker(); }
      catch(e){
        this.stopWorker();
        const m = await import(BASE + "engine.js"); this.direct = m.default;
        await this.direct.init({ base: BASE, models: CONFIG.models });
      }
    })().catch(e => { this.ready = null; throw e; });
    return this.ready;
  },
  startWorker(){
    return new Promise((res, rej) => {
      const w = new Worker(BASE + "engine.js", { type: "module" }); this.worker = w;
      w.onmessage = e => { const p = this.pend[e.data.id]; if (!p) return; delete this.pend[e.data.id]; e.data.ok ? p.res(e.data.result) : p.rej(new Error(e.data.error)); };
      w.onerror = e => { e.preventDefault?.(); const err = new Error(e.message || "worker error"); rej(err); this.failAll(err); this.stopWorker(); this.ready = null; };
      setTimeout(() => rej(new Error("worker start timed out")), 120000);
      this.rpc("init", { base: BASE, models: CONFIG.models }).then(res, rej);
    });
  },
  stopWorker(){ try { this.worker?.terminate(); } catch(e){} this.worker = null; },
  failAll(err){ Object.values(this.pend).forEach(p => p.rej(err)); this.pend = {}; },
  rpc(type, payload, transfer){ return new Promise((res, rej) => { const id = ++this.n; this.pend[id] = { res, rej }; this.worker.postMessage({ id, type, payload }, transfer || []); }); },
  async call(type, payload, transfer){ await this.start(); return this.worker ? this.rpc(type, payload, transfer) : this.direct[type](payload); }
};

// ---------- formatting ----------
const SUP = {"-":"⁻","0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹"};
function sci(x, p=2){
  if (!isFinite(x) || x === 0) return "0";
  const e = Math.floor(Math.log10(Math.abs(x))), m = x / Math.pow(10, e);
  if (e >= -2 && e <= 3) return (+x.toPrecision(p+1)).toString();
  return m.toFixed(p) + " × 10" + String(e).split("").map(c => SUP[c]).join("");
}
function fmtM(d){ return d < 1 ? (d*100).toFixed(0)+" cm" : d < 10 ? d.toFixed(2)+" m" : d.toFixed(1)+" m"; }
function fmtKg(m){ return m < 1 ? (m*1000).toFixed(m < 0.1 ? 1 : 0)+" g" : m < 1000 ? (+m.toPrecision(3))+" kg" : sci(m)+" kg"; }
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

// ---------- physics ----------
const force = (m1, m2, r) => G*m1*m2/(r*r);
function localG(latDeg, h){
  const s2 = Math.sin(latDeg*Math.PI/180)**2;
  return 9.7803253359*(1+0.00193185265241*s2)/Math.sqrt(1-0.00669437999013*s2) - 3.086e-6*h;
}
// iPhone SE (2nd gen) main camera: about 63° across the sensor's long side
function focal(W, H){ return (Math.max(W,H)/2)/Math.tan(31.55*Math.PI/180); }
const prior = o => CAT[o.cls] || [1, 0.3];
function mass(o){ return o.mass != null ? o.mass : prior(o)[0]; }
// ----- automatic distance: floor geometry, then "resting on", then apparent size -----
// Things that stand on the floor, so their box's bottom edge is where they touch it.
const FLOOR = new Set(["person","chair","couch","bed","dining table","desk","dog","cat","toilet","refrigerator","oven","stove","washing machine","dishwasher",
  "suitcase","bench","bicycle","bookcase","wardrobe","dresser","file cabinet","armchair","rocking chair","folding chair","barber chair","grand piano","upright piano",
  "trash can","space heater","vacuum cleaner","bowling ball","sports ball","basketball","soccer ball","tv stand","storage chest","crib","cradle","laundry hamper",
  "bucket","shopping cart","lawn mower","safe","radiator","pool table","china cabinet","barrel","crate","motorcycle","car","sneaker","sandal","backpack","potted plant","flowerpot"]);
// Surfaces other things rest on. An object whose base sits on one gets that surface's distance.
const SUPPORT = new Set(["dining table","desk","bed","couch","bench","chair","armchair","tv stand","dresser","storage chest","pool table","bookcase","china cabinet","file cabinet","refrigerator","oven","stove","washing machine","dishwasher","toilet"]);
// Horizontal distance to a point on the floor seen at image row y.
// The camera looks down depressionDeg below level from camH metres up; rows below centre look further down.
function floorDistance(y, H, f, depressionDeg, camH){
  const ray = depressionDeg + Math.atan((y - H/2) / f) * 180/Math.PI;
  return ray > 2.5 ? camH / Math.tan(ray*Math.PI/180) : null;   // near the horizon the answer is too unstable to use
}
function liveTilt(){
  if (S.beta == null) return null;
  const ang = (screen.orientation && screen.orientation.angle) ?? window.orientation ?? 0;
  return Math.abs(Math.abs(ang) === 90 ? S.gamma : S.beta);
}
function camDepression(){
  if (S.mode !== "live") return null;
  const t = S.frozen ? S.frozenTilt : liveTilt();
  return t == null ? null : 90 - t;
}
function floorDist(o){
  const dep = camDepression(); if (dep == null) return null;
  const base = o.bbox[1] + o.bbox[3];
  if (base > S.H * 0.985) return null;                           // bottom is cut off by the frame edge, so the floor contact isn't visible
  const d = floorDistance(base, S.H, S.f, dep, S.camH);
  return d && d < 40 ? d : null;
}
function supportOf(o){
  const bx = o.bbox[0] + o.bbox[2]/2, by = o.bbox[1] + o.bbox[3];
  let best = null;
  for (const s of S.objs){
    if (s === o || !SUPPORT.has(s.cls) || SUPPORT.has(o.cls) && s.bbox[2]*s.bbox[3] < o.bbox[2]*o.bbox[3]) continue;
    const [x,y,w,h] = s.bbox;
    if (bx < x || bx > x + w || by < y - h*0.08 || by > y + h*0.6) continue;
    const d = floorDist(s); if (!d) continue;
    if (!best || w*h < best.area) best = { s, d, area: w*h };
  }
  return best;
}
function sizeDist(o){ return clamp(prior(o)[1]*S.f/Math.max(o.bbox[2], o.bbox[3]), 0.15, 200); }
function distAuto(o){
  if (o.demoD) return { d: o.demoD, src: "demo" };
  const sup = supportOf(o); if (sup) return { d: sup.d, src: "on", on: sup.s.cls };
  if (FLOOR.has(o.cls)){ const d = floorDist(o); if (d) return { d, src: "floor" }; }
  return { d: sizeDist(o), src: "size" };
}
const distEst = o => distAuto(o).d;
function dist(o){ return o.dMode === "size" ? distEst(o) : o.d; }
function pos(o){
  const [x,y,w,h] = o.bbox, v = [(x+w/2-S.W/2)/S.f, (y+h/2-S.H/2)/S.f, 1], n = Math.hypot(v[0],v[1],v[2]), d = dist(o);
  return [v[0]/n*d, v[1]/n*d, d/n];
}
function between(a, b){ const p = pos(a), q = pos(b); return Math.max(0.05, Math.hypot(p[0]-q[0], p[1]-q[1], p[2]-q[2])); }
function pullOnYou(o){ return force(mass(o), S.you, dist(o)); }

// ---------- sky (low-precision Sun and Moon distances, Meeus ch. 25 and 47) ----------
function sky(date){
  const jd = date.getTime()/86400000 + 2440587.5, n = jd - 2451545, T = n/36525, r = Math.PI/180;
  const g = (357.529 + 0.98560028*n)*r;
  const sunAU = 1.00014 - 0.01671*Math.cos(g) - 0.00014*Math.cos(2*g);
  const D = (297.8501921 + 445267.1114034*T)*r, M = (357.5291092 + 35999.0502909*T)*r, Mp = (134.9633964 + 477198.8675055*T)*r, F = (93.2720950 + 483202.0175233*T)*r;
  const moonKm = 385000.56 - 20905.355*Math.cos(Mp) - 3699.111*Math.cos(2*D-Mp) - 2955.968*Math.cos(2*D) - 569.925*Math.cos(2*Mp)
    + 48.888*Math.cos(M) - 3.149*Math.cos(2*F) + 246.158*Math.cos(2*D-2*Mp) - 152.138*Math.cos(2*D-M-Mp) - 170.733*Math.cos(2*D+Mp)
    - 204.586*Math.cos(2*D-M) - 129.62*Math.cos(M-Mp) + 108.743*Math.cos(D) + 104.755*Math.cos(M+Mp);
  return { sunM: sunAU*1.495978707e11, moonM: moonKm*1000, sunAU, moonKm };
}

// ---------- demo scene ----------
const DEMO = [
  {cls:"tv", bbox:[470,140,260,160], d:3.6},
  {cls:"couch", bbox:[60,330,400,220], d:3.3},
  {cls:"potted plant", bbox:[1010,250,130,300], d:3.1},
  {cls:"dining table", bbox:[560,470,440,190], d:2.1},
  {cls:"cup", bbox:[640,418,46,54], d:2.15},
  {cls:"laptop", bbox:[770,380,150,92], d:2.25},
  {cls:"chair", bbox:[250,520,170,270], d:1.8},
  {cls:"bowling ball", bbox:[500,740,110,110], d:1.25},
  {cls:"cat", bbox:[860,720,210,130], d:1.3}
];
function newObj(p){
  const o = Object.assign({ id:S.nextId++, cls:"", detCls:null, mass:null, bbox:[0,0,1,1], dMode:"size", d:1, score:1, seen:performance.now(), src:"det", note:null, alts:null }, p);
  o.disp = o.bbox.slice(); return o;
}
function loadDemo(){
  setSize(1200, 900);
  S.objs = DEMO.map(o => newObj({ cls:o.cls, bbox:o.bbox.slice(), demoD:o.d, score:1, seen:Infinity, src:"demo" }));
  S.sel = S.objs.find(o => o.cls === "bowling ball").id;
}
function drawDemo(){
  const W = S.W, H = S.H, c = ctx, vx = 600, fy = 560;
  c.fillStyle = "#07070C"; c.fillRect(0,0,W,H);
  c.fillStyle = "#0c0c14"; c.fillRect(0,0,W,fy);
  c.strokeStyle = "rgba(233,236,242,.06)"; c.lineWidth = 1;
  for (let x = 0; x <= W; x += 60){ c.beginPath(); c.moveTo(x,0); c.lineTo(x,fy); c.stroke(); }
  c.fillStyle = "#0a0a11"; c.fillRect(0,fy,W,H-fy);
  c.strokeStyle = "rgba(0,231,255,.10)";
  for (let i = -14; i <= 14; i++){ c.beginPath(); c.moveTo(vx+i*40, fy); c.lineTo(vx+i*260, H); c.stroke(); }
  for (let k = 0; k < 9; k++){ const y = fy+(H-fy)*Math.pow(k/8,1.8); c.beginPath(); c.moveTo(0,y); c.lineTo(W,y); c.stroke(); }
  c.strokeStyle = "rgba(233,236,242,.18)"; c.beginPath(); c.moveTo(0,fy); c.lineTo(W,fy); c.stroke();
  const fill = "#1B1C28", edge = "rgba(233,236,242,.28)";
  const R = (x,y,w,h,r=6) => { c.beginPath(); c.roundRect ? c.roundRect(x,y,w,h,r) : c.rect(x,y,w,h); c.fillStyle = fill; c.fill(); c.strokeStyle = edge; c.stroke(); };
  const O = (x,y,r,f) => { c.beginPath(); c.arc(x,y,r,0,Math.PI*2); c.fillStyle = f || fill; c.fill(); c.strokeStyle = edge; c.stroke(); };
  R(470,140,260,150,4); c.fillStyle = "#07070C"; c.fillRect(480,150,240,130); c.fillStyle = "rgba(0,231,255,.07)"; c.fillRect(480,150,240,130); R(590,290,20,12,1);
  R(60,380,400,130,14); R(60,330,400,80,14); R(40,380,50,150,12); R(430,380,50,150,12);
  R(1040,450,80,100,6); O(1080,380,70,"#141824");
  R(560,470,440,26,3); R(585,496,18,164,2); R(957,496,18,164,2);
  R(640,420,40,50,6); c.beginPath(); c.arc(686,440,10,-Math.PI/2,Math.PI/2); c.strokeStyle = edge; c.stroke();
  R(780,380,130,82,4); c.fillStyle = "rgba(139,92,255,.10)"; c.fillRect(788,388,114,66); R(770,462,150,10,2);
  R(260,520,150,150,8); R(260,660,150,30,6); R(270,690,14,100,2); R(386,690,14,100,2);
  O(555,795,55,"#141420"); [[540,775],[560,770],[552,790]].forEach(([x,y]) => { c.beginPath(); c.arc(x,y,5,0,Math.PI*2); c.fillStyle = "#07070C"; c.fill(); });
  c.beginPath(); c.ellipse(965,800,95,45,0,0,Math.PI*2); c.fillStyle = fill; c.fill(); c.strokeStyle = edge; c.stroke();
  O(885,770,34);
  c.beginPath(); c.moveTo(862,748); c.lineTo(866,720); c.lineTo(882,740); c.moveTo(890,738); c.lineTo(904,716); c.lineTo(910,748); c.fillStyle = fill; c.fill(); c.stroke();
}

// ---------- layout: the media box keeps the frame's aspect ratio so taps map 1:1 ----------
function setSize(W, H){
  S.W = W; S.H = H; S.f = focal(W, H);
  if (cv.width !== W) cv.width = W; if (cv.height !== H) cv.height = H;
  fitMedia();
}
// Height budget uses the *small* viewport height (Safari toolbars showing), so scrolling
// and the toolbar sliding away never resize the camera view.
const vhProbe = Object.assign(document.createElement("div"), { style: "position:fixed;top:0;left:-9px;width:1px;height:74svh;visibility:hidden;pointer-events:none" });
document.body.appendChild(vhProbe);
let lastFit = "";
function fitMedia(){
  const stage = $("stage"), maxW = stage.clientWidth || 360, maxH = Math.max(260, vhProbe.offsetHeight || window.innerHeight * 0.74);
  const w = Math.min(maxW, maxH * S.W / S.H), h = w * S.H / S.W;
  media.style.width = Math.round(w) + "px"; media.style.height = Math.round(h) + "px"; media.style.aspectRatio = "auto";
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  ov.width = Math.round(w * dpr); ov.height = Math.round(h * dpr);
  S.cssW = w; lastFit = fitKey();
}
const fitKey = () => `${$("stage").clientWidth}|${vhProbe.offsetHeight}|${S.W}x${S.H}`;
window.addEventListener("resize", () => { if (fitKey() !== lastFit){ fitMedia(); drawOv(); } });

// The live picture is drawn into the canvas (not shown through the <video> element, which iOS can
// mis-size after scrolling). 'screen' blending lifts shadows without blowing out highlights.
function drawBoosted(c, src, w, h, boost){
  c.globalCompositeOperation = "source-over"; c.globalAlpha = 1; c.drawImage(src, 0, 0, w, h);
  if (boost > 0.02){
    c.globalCompositeOperation = "screen";
    for (let b = boost; b > 0.02; b -= 1){ c.globalAlpha = Math.min(1, b); c.drawImage(src, 0, 0, w, h); }
    c.globalCompositeOperation = "source-over"; c.globalAlpha = 1;
  }
}
function drawLiveFrame(){
  if (vid.readyState < 2) return false;
  if (vid.videoWidth && (vid.videoWidth !== S.W || vid.videoHeight !== S.H)){ setSize(vid.videoWidth, vid.videoHeight); S.objs = []; }
  drawBoosted(ctx, vid, S.W, S.H, S.boost); return true;
}
function applyDisplayBoost(){}   // kept as a hook; brightness is applied when each frame is drawn

// ---------- drawing ----------
function drawBase(){
  if (S.mode === "demo") drawDemo();
  else if (S.mode === "photo" && S.photo) ctx.drawImage(S.photo, 0, 0, S.W, S.H);
  else if (S.mode === "live" && S.frozen) {} // the frozen frame is already on the canvas
  else if (S.mode === "live" && S.stream && drawLiveFrame()) {}
  else { ctx.fillStyle = "#07070C"; ctx.fillRect(0, 0, S.W, S.H); }
}
function drawOv(){
  const c = octx, s = ov.width / S.W, u = S.W / Math.max(1, S.cssW), W = S.W, H = S.H, objs = S.objs;
  c.setTransform(1,0,0,1,0,0); c.clearRect(0, 0, ov.width, ov.height); c.setTransform(s,0,0,s,0,0);
  const ax = W/2, ay = H - 6*u, compact = S.cssW < 640;
  if (objs.length){
    const fs = objs.map(pullOnYou), lmin = Math.log10(Math.min(...fs)), lmax = Math.log10(Math.max(...fs));
    const sel = objs.find(o => o.id === S.sel);
    objs.forEach((o,i) => {
      const t = lmax - lmin < 1e-9 ? 1 : (Math.log10(fs[i]) - lmin)/(lmax - lmin), [x,y,w,h] = o.disp;
      c.strokeStyle = `rgba(0,231,255,${(0.14+0.76*t).toFixed(3)})`; c.lineWidth = (0.8+2*t)*u;
      c.beginPath(); c.moveTo(ax, ay); c.lineTo(x+w/2, y+h); c.stroke();
    });
    if (sel){
      const [sx,sy,sw,sh] = sel.disp;
      c.setLineDash([4*u, 5*u]); c.strokeStyle = "rgba(139,92,255,.8)"; c.lineWidth = 1.2*u;
      objs.forEach(o => { if (o === sel) return; const [x,y,w,h] = o.disp; c.beginPath(); c.moveTo(sx+sw/2, sy+sh/2); c.lineTo(x+w/2, y+h/2); c.stroke(); });
      c.setLineDash([]);
    }
    c.font = `500 ${11*u}px "Geist Mono", ui-monospace, monospace`; c.textBaseline = "middle";
    objs.forEach(o => {
      const [x,y,w,h] = o.disp, isSel = o === sel, L = Math.min(18*u, w/3, h/3);
      c.strokeStyle = isSel ? "#B9FF38" : o.src === "id" ? "rgba(139,92,255,.95)" : "rgba(233,236,242,.7)"; c.lineWidth = (isSel ? 2 : 1.3)*u;
      c.beginPath();
      c.moveTo(x,y+L); c.lineTo(x,y); c.lineTo(x+L,y);
      c.moveTo(x+w-L,y); c.lineTo(x+w,y); c.lineTo(x+w,y+L);
      c.moveTo(x+w,y+h-L); c.lineTo(x+w,y+h); c.lineTo(x+w-L,y+h);
      c.moveTo(x+L,y+h); c.lineTo(x,y+h); c.lineTo(x,y+h-L);
      c.stroke();
      if (isSel){ c.fillStyle = "rgba(185,255,56,.06)"; c.fillRect(x,y,w,h); }
      if (compact && !isSel) return;
      const txt = `${o.cls.toUpperCase()}  ${fmtM(dist(o))}`, tw = c.measureText(txt).width + 12*u, th = 18*u;
      let lx = clamp(x, 0, W - tw), ly = y - th - 3*u; if (ly < 34*u) ly = y + 3*u;
      c.fillStyle = isSel ? "#B9FF38" : "rgba(7,7,12,.82)"; c.fillRect(lx, ly, tw, th);
      c.fillStyle = isSel ? "#07070C" : "#E9ECF2"; c.fillText(txt, lx+6*u, ly+th/2+0.5*u);
    });
  }
  if (S.pending){
    const [x,y,w,h] = S.pending, t = (performance.now()/600) % 1;
    c.setLineDash([6*u,6*u]); c.lineDashOffset = -t*24*u; c.strokeStyle = "#00E7FF"; c.lineWidth = 1.6*u; c.strokeRect(x,y,w,h); c.setLineDash([]); c.lineDashOffset = 0;
  }
  if (S.mode === "live" && S.stream){
    const x = W/2, y = H/2; c.strokeStyle = "rgba(233,236,242,.85)"; c.lineWidth = 1*u;
    c.beginPath(); c.moveTo(x-14*u,y); c.lineTo(x-4*u,y); c.moveTo(x+4*u,y); c.lineTo(x+14*u,y);
    c.moveTo(x,y-14*u); c.lineTo(x,y-4*u); c.moveTo(x,y+4*u); c.lineTo(x,y+14*u); c.stroke();
  }
  c.fillStyle = "#07070C"; c.strokeStyle = "#00E7FF"; c.lineWidth = 1.5*u;
  c.beginPath(); c.arc(ax, ay-2*u, 6*u, 0, Math.PI*2); c.fill(); c.stroke();
  c.font = `500 ${10*u}px "Geist Mono", monospace`; c.fillStyle = "#00E7FF"; c.textAlign = "center"; c.textBaseline = "alphabetic";
  c.fillText("YOU", ax, ay-18*u); c.textAlign = "left";
}
function draw(){ drawBase(); snapDisp(); drawOv(); }
function snapDisp(){ S.objs.forEach(o => { o.disp = o.bbox.slice(); }); }
// Glide boxes toward their latest detection so they move smoothly between detector updates.
function glide(){
  let moving = false;
  S.objs.forEach(o => { for (let i = 0; i < 4; i++){ const d = o.bbox[i] - o.disp[i]; if (Math.abs(d) > 0.5){ o.disp[i] += d*0.35; moving = true; } else o.disp[i] = o.bbox[i]; } });
  return moving;
}

// ---------- selection panel ----------
$("catalog").innerHTML = Object.keys(CAT).sort().map(k => `<option value="${esc(k)}"></option>`).join("");
const typeInp = $("fType");
let panelFor = null, scaleBuilt = false;
const DMIN = 0.1, DMAX = 30;
const d2r = d => Math.round(1000*Math.log(d/DMIN)/Math.log(DMAX/DMIN));
const r2d = r => DMIN*Math.pow(DMAX/DMIN, r/1000);
const cur = () => S.objs.find(x => x.id === S.sel);

function renderScale(F){
  const lo = -14.5, hi = -4, p = v => (Math.log10(v)-lo)/(hi-lo)*100, pf = clamp(p(F), 0, 100);
  if (!scaleBuilt){
    $("sScale").innerHTML = `<div class="track"></div><div class="fill" id="scFill"></div>` +
      REFS.map((r,i) => `<div class="tick" style="left:${p(r[1])}%"></div><div class="tl ${i%2?'dn':'up'}" style="left:${clamp(p(r[1]),6,94)}%">${r[2]}</div>`).join("") +
      `<div class="pin" id="scPin"></div>`;
    scaleBuilt = true;
  }
  $("scFill").style.width = pf + "%"; $("scPin").style.left = pf + "%";
}
function compare(F){
  let best = REFS[0];
  for (const r of REFS) if (r[1] <= F) best = r;
  const ratio = F / best[1];
  if (ratio < 1) return `Less than the weight of ${best[0]}`;
  if (ratio < 1.6) return `About the weight of ${best[0]}`;
  return `About ${ratio < 10 ? ratio.toFixed(1) : Math.round(ratio).toLocaleString()}× the weight of ${best[0]}`;
}
function renderSel(full){
  const o = cur();
  $("selEmpty").hidden = !!o; $("selBody").hidden = !o;
  if (!o){ panelFor = null; return; }
  const m = mass(o), d = dist(o), F = pullOnYou(o), gl = localG(S.lat, S.alt);
  $("sName").textContent = o.cls;
  const au = o.dMode === "size" ? distAuto(o) : null;
  const dsrc = o.dMode === "tilt" ? ["tilt","Distance: rangefinder"] : o.dMode === "manual" ? ["manual","Distance: set by hand"]
    : au.src === "demo" ? ["","Distance: demo layout"] : au.src === "floor" ? ["tilt","Distance: floor geometry"]
    : au.src === "on" ? ["tilt",`Distance: resting on ${au.on}`] : o.src === "id" ? ["","Distance: rough, from tap area"] : ["","Distance: apparent size"];
  const chips = [`<span class="chip">${esc(o.src === "demo" ? "demo object" : o.src === "id" ? `identified ${Math.round(o.score*100)}%` : `detected ${Math.round(o.score*100)}%`)}</span>`,
    `<span class="chip ${dsrc[0]}">${dsrc[1]}</span>`];
  if (o.note) chips.push(`<span class="chip">${esc(o.note)}</span>`);
  if (o.mass != null) chips.push(`<span class="chip manual">mass set by hand</span>`);
  else if (!CAT[o.cls]) chips.push(`<span class="chip" style="color:var(--coral);border-color:rgba(255,65,108,.45)">mass unknown, set it</span>`);
  const chipHtml = chips.join("");
  if ($("sChips").dataset.h !== chipHtml){ $("sChips").innerHTML = chipHtml; $("sChips").dataset.h = chipHtml; }
  const alts = (o.alts || []).filter(a => a.name !== o.cls);
  $("sAlts").hidden = !alts.length;
  const altHtml = alts.length ? `<span>Other guesses:</span>` + alts.map(a => `<button data-n="${esc(a.name)}">${esc(a.name)} ${Math.round(a.score*100)}%</button>`).join("") : "";
  if ($("sAlts").dataset.h !== altHtml){ $("sAlts").innerHTML = altHtml; $("sAlts").dataset.h = altHtml; }
  $("sForce").innerHTML = `${sci(F)}<small>N</small>`;
  $("sCmp").textContent = compare(F);
  $("sCmp2").textContent = `Earth pulls you ${sci(S.you*gl/F,1)} times harder.`;
  renderScale(F);
  if ($("mathBox").open) $("sMath").textContent =
`F = G · m₁ · m₂ / r²
  = 6.674×10⁻¹¹ · ${+m.toPrecision(4)} kg · ${S.you} kg / (${d.toFixed(2)} m)²
  = ${sci(F,3)} N

m₁ is the ${o.cls}, m₂ is you, r is the distance between you.`;
  const others = S.objs.filter(x => x !== o).map(x => { const r = between(o, x); return { x, r, F: force(m, mass(x), r) }; }).sort((a,b) => b.F - a.F).slice(0, 4);
  $("sNb").innerHTML = others.length ? others.map(e => `<li><span style="text-transform:capitalize">${esc(e.x.cls)} · ${fmtM(e.r)}</span><span>${sci(e.F)} N</span></li>`).join("")
    : `<li><span class="small" style="margin:0">No other objects in view.</span></li>`;
  const active = document.activeElement;
  if ((full || panelFor !== o.id) && active !== typeInp) typeInp.value = o.cls;
  if (active !== $("fMass")) $("fMass").value = +m.toPrecision(4);
  if (active !== $("fDist")) $("fDist").value = d.toFixed(2);
  if (active !== $("fDistR")) $("fDistR").value = d2r(d);
  $("fSrc").textContent = o.dMode === "size" ? "Drag to set the distance yourself." : `Estimate was ${fmtM(distEst(o))}.`;
  $("fReset").hidden = o.dMode === "size" && o.mass == null;
  $("refine").hidden = S.mode === "demo" || o.src === "demo";
  panelFor = o.id;
}
$("sAlts").addEventListener("click", e => {
  const b = e.target.closest("button[data-n]"), o = cur(); if (!b || !o) return;
  o.cls = b.dataset.n; o.mass = null; o.note = "picked from guesses"; renderAll(true); drawOv();
});
$("mathBox").addEventListener("toggle", () => renderSel());

// Ranking rows are reused by id, so a tap never lands on a row that was just replaced.
const rankRows = new Map();
function renderRank(){
  const ul = $("rank"), list = S.objs.map(o => ({ o, F: pullOnYou(o) })).sort((a,b) => b.F - a.F);
  $("rCount").textContent = `${list.length} tracked`;
  $("rankEmpty").hidden = list.length > 0;
  const ids = new Set(list.map(e => e.o.id));
  for (const [id, li] of rankRows) if (!ids.has(id)){ li.remove(); rankRows.delete(id); }
  if (!list.length) return;
  const lmax = Math.log10(list[0].F), lmin = Math.log10(list[list.length-1].F) - 1;
  list.forEach(({o, F}, i) => {
    let li = rankRows.get(o.id);
    if (!li){
      li = document.createElement("li");
      li.innerHTML = `<button data-id="${o.id}"><span class="nm"></span><span class="fv"></span><span class="meta"></span><span></span><span class="bar"><i></i></span></button>`;
      rankRows.set(o.id, li);
    }
    const b = li.firstChild;
    b.setAttribute("aria-current", o.id === S.sel);
    b.children[0].textContent = o.cls;
    b.children[1].textContent = `${sci(F)} N`;
    b.children[2].textContent = `${fmtKg(mass(o))} · ${fmtM(dist(o))}`;
    b.children[4].firstChild.style.width = Math.max(3, (Math.log10(F)-lmin)/(lmax-lmin)*100).toFixed(1) + "%";
    if (ul.children[i] !== li) ul.insertBefore(li, ul.children[i] || null);
  });
}
function renderSky(){
  const sk = sky(new Date()), fs = force(1.989e30, S.you, sk.sunM), fm = force(7.342e22, S.you, sk.moonM), gl = localG(S.lat, S.alt);
  const lmx = Math.log10(fs), lmn = Math.log10(fm) - 1;
  $("skyRank").innerHTML = [["Sun", fs, `${sk.sunAU.toFixed(4)} AU`], ["Moon", fm, `${Math.round(sk.moonKm).toLocaleString()} km`]].map(([n,F,d]) =>
    `<li><div class="skyrow"><span class="nm">${n}</span><span class="fv">${sci(F,3)} N</span><span class="meta">${d}</span><span></span><span class="bar"><i style="width:${((Math.log10(F)-lmn)/(lmx-lmn)*100).toFixed(1)}%"></i></span></div></li>`).join("");
  $("earthCmp").textContent = `For comparison, Earth pulls you with ${(S.you*gl).toFixed(1)} N.`;
  $("skyDate").textContent = new Date().toLocaleDateString(undefined, { month:"short", day:"numeric", year:"numeric" });
  $("sunF").innerHTML = `${sci(fs,3)}<small>N from the Sun</small>`;
  $("skyKv").innerHTML = `<dt>Sun distance</dt><dd>${sk.sunAU.toFixed(4)} AU</dd><dt>Moon pull</dt><dd>${sci(fm,3)} N</dd><dt>Moon distance</dt><dd>${Math.round(sk.moonKm).toLocaleString()} km</dd><dt>Sun vs Moon</dt><dd>${Math.round(fs/fm)}×</dd>`;
}
function renderAll(full){ renderSel(full); renderRank(); renderTilt(); }

$("rank").addEventListener("click", e => { const b = e.target.closest("button[data-id]"); if (!b) return; S.sel = +b.dataset.id; renderAll(true); drawOv(); });
typeInp.addEventListener("change", () => {
  const o = cur(), v = typeInp.value.trim().toLowerCase(); if (!o || !v) return;
  o.cls = v; o.mass = null; o.note = null; renderAll(true); drawOv();
});
$("fMass").addEventListener("input", e => { const o = cur(), v = parseFloat(e.target.value); if (!o || !(v > 0)) return; o.mass = v; renderAll(); drawOv(); });
$("fDist").addEventListener("input", e => { const o = cur(), v = parseFloat(e.target.value); if (!o || !(v >= 0.05)) return; o.dMode = "manual"; o.d = v; renderAll(); drawOv(); });
$("fDistR").addEventListener("input", e => { const o = cur(); if (!o) return; o.dMode = "manual"; o.d = +r2d(+e.target.value).toFixed(2); renderAll(); drawOv(); });
$("fReset").addEventListener("click", () => { const o = cur(); if (!o) return; o.dMode = "size"; o.mass = null; renderAll(true); drawOv(); });
$("youMass").value = S.you;
$("youMass").addEventListener("input", e => { const v = parseFloat(e.target.value); if (v > 0){ S.you = v; store.set("you", v); renderAll(); renderSky(); drawOv(); } });

// ---------- tap: select a box (with a finger-sized margin), or identify whatever is under an empty spot ----------
ov.addEventListener("click", e => {
  const r = ov.getBoundingClientRect(), x = (e.clientX - r.left)*S.W/r.width, y = (e.clientY - r.top)*S.H/r.height;
  const pad = 12 * S.W / Math.max(1, S.cssW);
  const hits = S.objs.filter(o => { const [bx,by,bw,bh] = o.disp; return x >= bx-pad && x <= bx+bw+pad && y >= by-pad && y <= by+bh+pad; })
    .map(o => { const [bx,by,bw,bh] = o.disp, inside = x >= bx && x <= bx+bw && y >= by && y <= by+bh; return { o, inside, area: bw*bh }; })
    .sort((a,b) => (b.inside - a.inside) || (a.area - b.area));
  if (hits[0]){ S.sel = hits[0].o.id; renderAll(true); drawOv(); return; }
  if (S.mode === "demo") return;
  identifyAt(x, y);
});

// ---------- identify ----------
function pickCats(cats){
  if (!cats.length) return null;
  // prefer the best guess with a known mass, if it's reasonably close to the top guess
  const known = cats.find(c => IMN[c.index] != null && c.score >= cats[0].score * 0.5);
  const top = known || cats[0];
  const name = c => IMN[c.index] || (IN_LABELS[c.index] || c.name || "object").toLowerCase();
  const seen = new Set(), alts = [];
  cats.forEach(c => { const n = name(c); if (!seen.has(n) && c.score >= 0.03){ seen.add(n); alts.push({ name: n, score: c.score }); } });
  return { name: name(top), score: top.score, known: IMN[top.index] != null, alts: alts.slice(0, 4) };
}
async function frameBitmap(){
  if (S.mode === "photo" && S.photo) return createImageBitmap(S.photo);
  if (S.mode === "live") return createImageBitmap(cv);   // the frozen frame lives on the base canvas
  return null;
}
let identifying = false;
async function identifyAt(x, y){
  if (identifying) return; identifying = true;
  try {
    if (S.mode === "live" && !S.frozen) setFrozen(true);
    const short = Math.min(S.W, S.H);
    const boxes = [0.2, 0.34, 0.5].map(f => { const s = short*f; return [clamp(x - s/2, 0, S.W - s), clamp(y - s/2, 0, S.H - s), s, s]; });
    S.pending = boxes[1]; animatePending();
    setHud("busy", S.clfReady ? "Identifying" : "Loading identifier (24 MB, first time only)");
    const bm = await frameBitmap(); if (!bm) return;
    const r = await Engine.call("classify", { bitmap: bm, boxes, boost: 0 }, [bm]);   // the frozen frame is already brightened
    S.clfReady = true;
    const p = pickCats(r.cats);
    if (!p || p.score < 0.1){
      setHud("warn", "Not sure what that is");
      $("hudB").textContent = "Try tapping the middle of the object, or step closer so it fills more of the frame." + (p ? ` Best guess was ${p.name} (${Math.round(p.score*100)}%).` : "");
      return;
    }
    const o = newObj({ cls: p.name, bbox: boxes[r.box], score: p.score, seen: Infinity, src: "id", alts: p.alts, note: p.score < 0.25 ? "low confidence" : null });
    S.objs.push(o); S.sel = o.id;
    setHud("on", `Identified: ${p.name}`);
    $("hudB").textContent = p.score < 0.25 ? "Not sure about this one. Pick another guess in the panel, or type the right name." : "Set the distance with the rangefinder or the slider for a better number.";
  } catch(e){
    setHud("warn", "Identifier unavailable");
    $("hudB").textContent = "Couldn't load the identifier. Check your connection and tap again.";
  } finally {
    identifying = false; S.pending = null; renderAll(true); drawOv();
  }
}
function animatePending(){ if (!S.pending) return; drawOv(); requestAnimationFrame(animatePending); }
async function refineObjs(objs, quiet){
  const bm = await frameBitmap(); if (!bm) return 0;
  const r = await Engine.call("classify", { bitmap: bm, boxes: objs.map(o => o.bbox), boost: 0, each: true }, [bm]);
  S.clfReady = true;
  let n = 0;
  r.list.forEach((cats, i) => {
    const o = objs[i], p = pickCats(cats); if (!p) return;
    o.alts = p.alts;
    if (p.known && p.score >= (quiet ? 0.35 : 0.12) && p.name !== o.cls){ o.note = `was ${o.detCls || o.cls}`; o.cls = p.name; o.mass = null; n++; }
    if (!quiet) $("hudB").textContent = p.known && p.score >= 0.12 ? `Identified as ${p.name} (${Math.round(p.score*100)}%).` : `Not sure. Kept "${o.cls}". Other guesses are in the panel.`;
  });
  return n;
}
$("refine").addEventListener("click", async () => {
  const o = cur(); if (!o) return;
  const b = $("refine"); b.disabled = true; b.textContent = "…";
  try {
    if (S.mode === "live" && !S.frozen) setFrozen(true);
    setHud("busy", S.clfReady ? "Identifying" : "Loading identifier");
    await refineObjs([o], false); setHud("on", "Done");
  } catch(e){ setHud("warn", "Identifier unavailable"); $("hudB").textContent = "Couldn't load the identifier. Check your connection and try again."; }
  b.disabled = false; b.textContent = "Identify"; renderAll(true); drawOv();
});
$("identify").addEventListener("click", () => {
  // if something is already detected under the crosshair, name it more precisely; otherwise identify the spot
  const x = S.W/2, y = S.H/2;
  const under = S.objs.filter(o => { const [bx,by,bw,bh] = o.bbox; return x >= bx && x <= bx+bw && y >= by && y <= by+bh; }).sort((a,b) => a.bbox[2]*a.bbox[3] - b.bbox[2]*b.bbox[3])[0];
  if (under){ S.sel = under.id; $("refine").click(); } else identifyAt(x, y);
});

// ---------- tracking ----------
function iou(a, b){
  const x1 = Math.max(a[0],b[0]), y1 = Math.max(a[1],b[1]), x2 = Math.min(a[0]+a[2],b[0]+b[2]), y2 = Math.min(a[1]+a[3],b[1]+b[3]);
  const inter = Math.max(0,x2-x1)*Math.max(0,y2-y1); return inter/(a[2]*a[3]+b[2]*b[3]-inter);
}
function merge(dets){
  const now = performance.now(), used = new Set(), reach = Math.max(S.W, S.H) * 0.15;
  dets.sort((a,b) => b.score - a.score).forEach(d => {
    let best = null, bs = 0;
    S.objs.forEach(o => {
      if (used.has(o) || o.detCls !== d.class) return;
      const v = iou(o.bbox, d.bbox), cd = Math.hypot(o.bbox[0]+o.bbox[2]/2 - d.bbox[0]-d.bbox[2]/2, o.bbox[1]+o.bbox[3]/2 - d.bbox[1]-d.bbox[3]/2);
      const sc = v > 0.15 ? 1 + v : cd < reach ? 1 - cd/reach : 0;
      if (sc > bs){ bs = sc; best = o; }
    });
    if (best){ used.add(best); best.bbox = d.bbox; best.score = best.score*0.6 + d.score*0.4; best.seen = now; }
    else { const o = newObj({ cls: d.class, detCls: d.class, bbox: d.bbox, score: d.score, seen: now }); S.objs.push(o); used.add(o); }
  });
  S.objs = S.objs.filter(o => now - o.seen < (o.id === S.sel ? 3000 : 1000));
  if (!S.objs.some(o => o.id === S.sel)) S.sel = S.objs.length ? S.objs.reduce((a,b) => pullOnYou(b) > pullOnYou(a) ? b : a).id : null;
}

// ---------- modes ----------
function setMode(m){
  if (S.mode === "live" && m !== "live") stopCam();
  S.mode = m; S.frozen = false; S.pending = null;
  document.querySelectorAll(".seg button").forEach(b => b.setAttribute("aria-pressed", b.dataset.mode === m));
  ["freeze","identify","torch","newPhoto","boostWrap","evWrap","tiltBtn"].forEach(id => $(id).hidden = true);
  $("tools").hidden = m === "demo"; $("drop").hidden = true;
  vid.hidden = m !== "live";
  if (m === "demo"){ loadDemo(); $("hudL").textContent = "Demo · schematic room"; setHud("on","Ready"); $("hudB").textContent = "Tap any object. Cyan lines are its pull on you; brighter is stronger."; }
  if (m === "live"){ S.objs = []; S.sel = null; setSize(1280, 720); $("hudL").textContent = "Live · rear camera"; $("hudB").textContent = "Point at a room. Tap anything unlabeled to identify it."; startCam(); }
  if (m === "photo"){
    S.objs = []; S.sel = null; S.photo = null; setSize(1200, 900); $("hudL").textContent = "Photo"; setHud("", "Waiting for photo");
    $("hudB").textContent = "Take or choose a photo of a room.";
    showDrop("Scan a photo", "Take a picture of a room or pick one from your library. Detection runs on this device.", "Take or choose photo", () => $("file").click());
    Engine.call("warm", { which: "photo" }).catch(() => {});
  }
  applyDisplayBoost(); renderAll(true); draw();
}
function showDrop(title, text, btn, fn){
  const d = $("drop"); d.hidden = false;
  d.innerHTML = `<div><div class="eyebrow">${esc(title)}</div><p>${esc(text)}</p>${btn ? `<button class="cta" id="dropBtn">${esc(btn)}</button>` : ""}</div>`;
  if (btn) $("dropBtn").onclick = fn;
}
document.querySelectorAll(".seg button").forEach(b => b.addEventListener("click", () => { if (b.dataset.mode !== S.mode || b.dataset.mode === "photo") setMode(b.dataset.mode); }));

async function startCam(){
  const id = ++S.loopId;
  if (!navigator.mediaDevices?.getUserMedia){ camFail("This browser doesn't allow camera access here."); return; }
  setHud("busy", "Starting camera");
  Engine.start().catch(() => {});   // start downloading the detector while the camera spins up
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } } });
  } catch(e){
    camFail(inFrame ? "This preview blocks the camera. Use Photo here, or open the GitHub version for live mode."
      : "Camera access was denied. Allow it for this site in Settings → Apps → Safari → Camera, then reload.");
    return;
  }
  if (S.mode !== "live" || id !== S.loopId){ stopCam(); return; }
  S.track = S.stream.getVideoTracks()[0];
  vid.srcObject = S.stream; vid.hidden = false;
  await vid.play().catch(() => {});
  if (!vid.videoWidth) await new Promise(r => vid.addEventListener("loadedmetadata", r, { once:true }));
  setSize(vid.videoWidth, vid.videoHeight);
  setupCamControls();
  ["freeze","identify","boostWrap"].forEach(k => $(k).hidden = false); $("freeze").textContent = "Freeze";
  $("tiltBtn").hidden = S.beta != null;
  requestWake();
  renderLoop(id);
  setHud("busy", "Loading detector");
  try { await Engine.start(); } catch(e){ setHud("warn", "Detector failed to load"); $("hudB").textContent = "The detector couldn't load. Check your connection, then switch modes and back to retry."; return; }
  detectLoop(id);
}
function camFail(msg){ setHud("warn", "Camera unavailable"); showDrop("Live camera", msg, "Switch to Photo", () => setMode("photo")); }
function stopCam(){
  S.loopId++; S.stream?.getTracks().forEach(t => t.stop()); S.stream = null; S.track = null; vid.srcObject = null; S.torch = false;
  try { S.wake?.release(); } catch(e){} S.wake = null;
}
async function requestWake(){ try { S.wake = await navigator.wakeLock?.request("screen"); } catch(e){} }
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible" && S.stream) requestWake(); });

// The detector runs in the worker; the page only grabs frames and draws boxes.
async function detectLoop(id){
  let frames = 0, t0 = performance.now();
  while (id === S.loopId && S.mode === "live" && S.stream){
    if (S.frozen || document.visibilityState !== "visible" || vid.readyState < 2){ await sleep(150); continue; }
    const start = performance.now();
    try {
      const bm = await createImageBitmap(vid);
      const r = await Engine.call("detect", { bitmap: bm, boost: S.boost, which: "live", maxSide: CONFIG.liveMaxSide }, [bm]);
      if (id !== S.loopId || S.frozen) continue;
      merge(r.dets.map(d => ({ class: d.cls, score: d.score, bbox: d.bbox })));
      if (S.autoBoost) autoGain(r.lum);
      S.inferMs = S.inferMs ? S.inferMs*0.8 + r.ms*0.2 : r.ms;
      frames++;
      const now = performance.now();
      if (r.backend) S.backend = r.backend;
      if (now - t0 > 1000){ S.fps = frames * 1000 / (now - t0); frames = 0; t0 = now; setHud("on", `Scanning · ${S.fps.toFixed(0)} fps · ${Math.round(S.inferMs)} ms · ${S.backend || "CPU"}`); }
    } catch(e){ setHud("warn", "Detector hiccup, retrying"); await sleep(800); }
    const dt = performance.now() - start; if (dt < CONFIG.minDetectGap) await sleep(CONFIG.minDetectGap - dt);
  }
}
function renderLoop(id){
  if (id !== S.loopId || S.mode !== "live" || !S.stream) return;
  if (!S.frozen){
    if (vid.currentTime !== S.lastVT){ S.lastVT = vid.currentTime; drawLiveFrame(); }
    glide(); drawOv();
    const now = performance.now();
    if (now - S.lastPanel > 300){ S.lastPanel = now; renderSel(); }
    if (now - S.lastRank > 700){ S.lastRank = now; renderRank(); renderTilt(); }
  }
  requestAnimationFrame(() => renderLoop(id));
}
function setFrozen(on){
  S.frozen = on; $("freeze").textContent = on ? "Resume" : "Freeze";
  if (on){
    S.frozenTilt = liveTilt();
    drawLiveFrame(); snapDisp();
    S.objs.forEach(o => o.seen = Infinity); setHud("", "Frozen · tap anything to identify it");
  } else {
    const n = performance.now(); S.objs = S.objs.filter(o => o.src !== "id"); S.objs.forEach(o => o.seen = n); setHud("on", "Scanning");
  }
  applyDisplayBoost(); renderAll(true); drawOv();
}
$("freeze").addEventListener("click", () => setFrozen(!S.frozen));

// Hardware controls where the camera exposes them; software brightness always.
function setupCamControls(){
  const caps = S.track?.getCapabilities?.() || {};
  $("torch").hidden = !caps.torch; $("torch").setAttribute("aria-pressed", "false");
  const c = caps.exposureCompensation;
  if (c && c.max > c.min){
    const ev = $("ev"); ev.min = c.min; ev.max = c.max; ev.step = c.step || 0.1; ev.value = S.track.getSettings?.().exposureCompensation ?? 0;
    $("evWrap").hidden = false;
  }
  $("boost").value = Math.round((S.autoBoost ? S.boost : S.manualBoost)*100); $("autoBoost").checked = S.autoBoost; $("boost").disabled = S.autoBoost;
  if (!S.autoBoost) S.boost = S.manualBoost;
  applyDisplayBoost();
}
$("torch").addEventListener("click", async () => {
  if (!S.track) return; S.torch = !S.torch;
  try { await S.track.applyConstraints({ advanced: [{ torch: S.torch }] }); } catch(e){ S.torch = false; }
  $("torch").setAttribute("aria-pressed", S.torch);
});
$("ev").addEventListener("input", e => { S.track?.applyConstraints({ advanced: [{ exposureMode: "continuous", exposureCompensation: +e.target.value }] }).catch(() => {}); });
$("autoBoost").addEventListener("change", e => { S.autoBoost = e.target.checked; store.set("autoBoost", S.autoBoost); $("boost").disabled = S.autoBoost; if (!S.autoBoost){ S.boost = S.manualBoost; applyDisplayBoost(); } });
$("boost").addEventListener("input", e => { S.manualBoost = +e.target.value/100; store.set("boost", S.manualBoost); if (!S.autoBoost){ S.boost = S.manualBoost; applyDisplayBoost(); } });
// Auto brightness: the worker reports the raw frame's mean luminance; lift it toward mid-grey.
function autoGain(mean){
  const target = 0.42, want = mean >= target ? 0 : (target - mean) / Math.max(0.02, mean*(1-mean));
  const next = S.boost*0.75 + clamp(want, 0, 2)*0.25;
  if (Math.abs(next - S.boost) > 0.02){ S.boost = next; applyDisplayBoost(); $("boost").value = Math.round(S.boost*100); }
}

$("file").addEventListener("change", e => {
  const file = e.target.files[0]; e.target.value = ""; if (!file) return;
  const url = URL.createObjectURL(file), img = new Image();
  img.onerror = () => { URL.revokeObjectURL(url); showDrop("Photo", "That file couldn't be opened as an image. Try a JPEG or PNG.", "Choose another", () => $("file").click()); };
  img.onload = async () => {
    const sc = Math.min(1, 1600/Math.max(img.naturalWidth, img.naturalHeight));
    const c = document.createElement("canvas"); c.width = Math.round(img.naturalWidth*sc); c.height = Math.round(img.naturalHeight*sc);
    c.getContext("2d").drawImage(img, 0, 0, c.width, c.height); URL.revokeObjectURL(url);
    S.photo = c; setSize(c.width, c.height);
    $("drop").hidden = true; $("newPhoto").hidden = false; S.objs = []; S.sel = null; draw();
    setHud("busy", "Detecting");
    let r;
    try { const bm = await createImageBitmap(c); r = await Engine.call("detect", { bitmap: bm, boost: 0, which: "photo", maxSide: CONFIG.photoMaxSide }, [bm]); }
    catch(err){ showDrop("Photo", "The detector couldn't load. Check your connection and try again.", "Try again", () => $("file").click()); return; }
    if (S.photo !== c) return;
    merge(r.dets.map(d => ({ class: d.cls, score: d.score, bbox: d.bbox }))); S.objs.forEach(o => o.seen = Infinity);
    S.sel = S.objs.length ? S.objs.reduce((a,b) => pullOnYou(b) > pullOnYou(a) ? b : a).id : null;
    setHud("on", `${S.objs.length} objects found`);
    $("hudB").textContent = S.objs.length ? "Tap any object, or tap anything unlabeled to identify it." : "Nothing detected. Tap on things to identify them one at a time.";
    renderAll(true); draw();
    // name generic detections more precisely (bottle → wine bottle, cup → coffee mug)
    const todo = S.objs.filter(o => REFINABLE.has(o.detCls));
    if (todo.length){
      setHud("busy", "Naming objects more precisely");
      try { const n = await refineObjs(todo, true); if (S.photo === c){ setHud("on", `${S.objs.length} objects · ${n} renamed`); renderAll(true); drawOv(); } }
      catch(err){ if (S.photo === c) setHud("on", `${S.objs.length} objects found`); }
    }
  };
  img.src = url;
});
$("newPhoto").addEventListener("click", () => $("file").click());

// ---------- sensors ----------
const spark = $("spark"), sctx = spark.getContext("2d");
function setStat(el, cls, text){ el.className = "status " + cls; el.innerHTML = `<span class="dot ${cls}"></span>${esc(text)}`; }
function enableSensors(){
  const btn = $("enableSensors"); btn.disabled = true;
  // iOS needs both permission requests inside the same tap, before any await.
  const asks = [];
  if (typeof DeviceMotionEvent !== "undefined" && DeviceMotionEvent.requestPermission) asks.push(DeviceMotionEvent.requestPermission());
  if (typeof DeviceOrientationEvent !== "undefined" && DeviceOrientationEvent.requestPermission) asks.push(DeviceOrientationEvent.requestPermission());
  Promise.allSettled(asks).then(rs => {
    const denied = rs.some(r => r.status === "rejected" || r.value === "denied");
    window.addEventListener("devicemotion", onMotion);
    window.addEventListener("deviceorientation", onOrient);
    setStat($("aStat"), "busy", "Listening");
    setTimeout(() => {
      if (S.motion.mag == null){
        setStat($("aStat"), "warn", denied ? "Permission denied" : "No sensor data");
        $("gDiff").textContent = inFrame ? "Blocked in this preview" : denied ? "Allow Motion & Orientation Access for this site" : "This device reports no motion data";
        btn.disabled = false; btn.textContent = "Try again";
      } else btn.textContent = "Sensors on";
    }, 1500);
  });
}
function onMotion(e){
  const a = e.accelerationIncludingGravity; if (!a || a.x == null) return;
  const mag = Math.hypot(a.x, a.y, a.z), M = S.motion;
  if (M.mag == null){ setStat($("aStat"), "on", "Live"); $("capUp").disabled = false; $("capDn").disabled = false; startSensorUi(); }
  M.mag = mag; M.ema = M.ema == null ? mag : M.ema*0.9 + mag*0.1;
  M.buf.push(mag); if (M.buf.length > 240) M.buf.shift();
  if (M.samples) M.samples.push(Math.abs(a.z));
}
function onOrient(e){
  if (e.beta == null) return;
  if (S.beta == null){ setStat($("tStat"), "on", "Live tilt"); $("tManWrap").hidden = true; $("tiltBtn").hidden = true; }
  S.beta = e.beta; S.gamma = e.gamma;
  const now = performance.now(); if (now - (S.lastTilt || 0) > 150){ S.lastTilt = now; renderTilt(); }
}
function capture(side){
  const M = S.motion, btn = side === "up" ? $("capUp") : $("capDn"), label = btn.textContent;
  btn.disabled = true; M.samples = []; btn.textContent = "Hold still…";
  setTimeout(() => {
    const s = M.samples; M.samples = null; btn.disabled = false; btn.textContent = label;
    if (s.length) { M[side] = s.reduce((a,b) => a+b, 0)/s.length; renderG(); }
  }, 2000);
}
$("enableSensors").addEventListener("click", enableSensors);
$("tiltBtn").addEventListener("click", enableSensors);
$("capUp").addEventListener("click", () => capture("up"));
$("capDn").addEventListener("click", () => capture("dn"));
function renderG(){
  const gp = localG(S.lat, S.alt), M = S.motion;
  $("gPred").textContent = gp.toFixed(4) + " m/s²";
  if (M.ema != null) $("aVal").innerHTML = `${M.ema.toFixed(3)}<small>m/s²</small>`;
  let cal = null;
  if (M.up != null && M.dn != null){ cal = (M.up + M.dn)/2; $("gCal").textContent = cal.toFixed(4) + " m/s²"; }
  else $("gCal").textContent = M.up != null ? "Now capture face down" : M.dn != null ? "Now capture face up" : "—";
  const ref = cal ?? M.ema;
  if (ref != null){ const pct = (ref - gp)/gp*100; $("gDiff").textContent = `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}% vs predicted`; }
}
function drawSpark(){
  const W = spark.width, H = spark.height, b = S.motion.buf, c = sctx, gp = localG(S.lat, S.alt), lo = gp - 1.5, hi = gp + 1.5;
  const y = v => H - (clamp(v, lo, hi) - lo)/(hi - lo)*H;
  c.clearRect(0, 0, W, H);
  c.strokeStyle = "rgba(233,236,242,.08)"; c.lineWidth = 1;
  [lo+0.5, hi-0.5].forEach(v => { c.beginPath(); c.moveTo(0,y(v)); c.lineTo(W,y(v)); c.stroke(); });
  c.strokeStyle = "rgba(139,92,255,.6)"; c.setLineDash([6,6]); c.beginPath(); c.moveTo(0,y(gp)); c.lineTo(W,y(gp)); c.stroke(); c.setLineDash([]);
  if (b.length > 1){
    c.strokeStyle = "#00E7FF"; c.lineWidth = 2; c.beginPath();
    b.forEach((v,i) => { const x = i/239*W; i ? c.lineTo(x, y(v)) : c.moveTo(x, y(v)); }); c.stroke();
    c.fillStyle = "#B9FF38"; c.beginPath(); c.arc((b.length-1)/239*W, y(b[b.length-1]), 3.5, 0, Math.PI*2); c.fill();
  } else { c.fillStyle = "#858799"; c.font = "22px 'Geist Mono', monospace"; c.fillText("predicted g ─ ─", 12, y(gp) - 10); }
}
function startSensorUi(){ if (S.sensorTimer) return; S.sensorTimer = setInterval(() => { if (document.visibilityState === "visible"){ renderG(); drawSpark(); } }, 100); }

function tiltAngle(){ const t = liveTilt(); return t == null ? parseFloat($("tMan").value) : t; }
function tiltDist(){ const b = tiltAngle(); return b > 3 && b < 87 ? S.camH * Math.tan(b*Math.PI/180) : null; }
function renderTilt(){
  const b = tiltAngle(), d = tiltDist();
  $("tAng").textContent = isFinite(b) ? b.toFixed(1) + "°" : "—";
  $("tDist").innerHTML = d ? `${d.toFixed(2)}<small>m</small>` : `—<small>aim lower</small>`;
  const ex = d ? Math.min(280, 30 + d/(S.camH*5)*250) : 280;
  $("ray").setAttribute("x2", ex); $("rayEnd").setAttribute("cx", ex);
  const o = cur(); $("tApply").textContent = o ? `Apply to ${o.cls}` : "Apply to selected"; $("tApply").disabled = !o || !d;
}
$("tH").value = S.camH;
$("tH").addEventListener("input", e => { const v = parseFloat(e.target.value); if (v > 0.1){ S.camH = v; store.set("camH", v); renderTilt(); } });
$("tMan").addEventListener("input", renderTilt);
$("tApply").addEventListener("click", () => { const o = cur(), d = tiltDist(); if (!o || !d) return; o.dMode = "tilt"; o.d = +d.toFixed(2); renderAll(true); drawOv(); });

$("lat").value = S.lat; $("alt").value = S.alt;
$("lat").addEventListener("input", e => { const v = parseFloat(e.target.value); if (v >= -90 && v <= 90){ S.lat = v; store.set("lat", v); renderG(); renderSky(); drawSpark(); } });
$("alt").addEventListener("input", e => { const v = parseFloat(e.target.value); if (isFinite(v)){ S.alt = v; store.set("alt", v); renderG(); renderSky(); drawSpark(); } });
$("geo").addEventListener("click", () => {
  if (!navigator.geolocation){ $("geoMsg").textContent = "Location isn't available here. Type your latitude."; return; }
  $("geoMsg").textContent = "Locating…";
  navigator.geolocation.getCurrentPosition(p => {
    S.lat = +p.coords.latitude.toFixed(4); $("lat").value = S.lat; store.set("lat", S.lat);
    if (p.coords.altitude != null){ S.alt = Math.round(p.coords.altitude); $("alt").value = S.alt; store.set("alt", S.alt); }
    $("geoMsg").textContent = "Updated."; renderG(); renderSky(); drawSpark();
  }, () => { $("geoMsg").textContent = inFrame ? "Blocked in this preview. Type your latitude." : "Location was denied. Type your latitude."; }, { enableHighAccuracy:true, timeout:8000 });
});

function setHud(state, text){ $("hudR").innerHTML = `<span class="dot ${state}"></span>${esc(text)}`; }

// ---------- install and offline ----------
const standalone = navigator.standalone === true || matchMedia("(display-mode: standalone)").matches;
const isIOS = /iPhone|iPad|iPod/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
if (isIOS && !standalone && !inFrame && !store.get("installHidden", false)) $("install").hidden = false;
$("installX").addEventListener("click", () => { $("install").hidden = true; store.set("installHidden", true); });
if ("serviceWorker" in navigator && !inFrame && (location.protocol === "https:" || location.hostname === "localhost")){
  let hadController = !!navigator.serviceWorker.controller;
  navigator.serviceWorker.register("sw.js").catch(() => {});
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    if (hadController){ $("toastMsg").textContent = "A new version is ready."; $("toast").hidden = false; }
    hadController = true;
  });
}
$("toastBtn").addEventListener("click", () => location.reload());

// ---------- boot ----------
window.__faintPull = { S, Engine, floorDistance, distAuto, setMode, newObj };   // handy for debugging from the console
setMode("demo"); renderSky(); renderG(); drawSpark();
document.fonts?.ready.then(() => drawOv());
})();
