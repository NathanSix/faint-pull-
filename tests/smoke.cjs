// Smoke tests for Faint Pull. Runs the real page in a phone-sized Chromium with a fake camera.
// Usage: node tests/smoke.cjs   (serves the repo on a local port by itself)
const { chromium } = require("playwright");
const http = require("http"), fs = require("fs"), path = require("path");

const ROOT = path.resolve(__dirname, "..");
const TYPES = { ".html":"text/html", ".js":"text/javascript", ".mjs":"text/javascript", ".wasm":"application/wasm", ".json":"application/json",
  ".webmanifest":"application/manifest+json", ".png":"image/png", ".jpg":"image/jpeg", ".tflite":"application/octet-stream" };
const server = http.createServer((req, res) => {
  let p = decodeURIComponent(new URL(req.url, "http://x").pathname); if (p.endsWith("/")) p += "index.html";
  const f = path.join(ROOT, p);
  if (!f.startsWith(ROOT) || !fs.existsSync(f) || fs.statSync(f).isDirectory()){ res.writeHead(404); return res.end(); }
  res.writeHead(200, { "Content-Type": TYPES[path.extname(f)] || "application/octet-stream" }); fs.createReadStream(f).pipe(res);
});

const results = []; let failed = 0;
async function check(name, fn){
  const t = Date.now();
  try { const note = await fn(); results.push(`PASS  ${name}${note ? "  (" + note + ")" : ""}  ${Date.now()-t}ms`); }
  catch(e){ failed++; results.push(`FAIL  ${name}\n      ${String(e.message || e).split("\n")[0]}`); }
}
const assert = (c, msg) => { if (!c) throw new Error(msg); };

async function testImage(){
  if (process.env.TEST_IMAGE && fs.existsSync(process.env.TEST_IMAGE)) return process.env.TEST_IMAGE;
  const url = process.env.TEST_IMAGE_URL; if (!url) return null;
  try { const r = await fetch(url); if (!r.ok) return null; const f = path.join(require("os").tmpdir(), "fp-test.jpg"); fs.writeFileSync(f, Buffer.from(await r.arrayBuffer())); return f; }
  catch(e){ return null; }
}

(async () => {
  await new Promise(r => server.listen(0, r));
  const base = `http://localhost:${server.address().port}/`;
  const browser = await chromium.launch({ args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"] });
  const ctx = await browser.newContext({ viewport: { width: 375, height: 667 }, deviceScaleFactor: 2, permissions: ["camera"] });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", e => errors.push(e.message));
  page.on("console", m => { if (m.type() === "error" && !/^INFO:|fonts\.g|ERR_(TUNNEL|NAME|INTERNET|CONNECTION)|net::/.test(m.text())) errors.push(m.text()); });
  await page.addInitScript(() => { window.__long = []; try { new PerformanceObserver(l => l.getEntries().forEach(e => window.__long.push(e.duration))).observe({ type: "longtask", buffered: true }); } catch(e){} });
  const hud = () => page.$eval("#hudR", e => e.textContent);
  const hudB = () => page.$eval("#hudB", e => e.textContent);

  await check("page loads without script errors", async () => {
    await page.goto(base); await page.waitForTimeout(800);
    assert((await page.title()) === "Faint Pull", "wrong title");
    assert(errors.length === 0, "errors: " + errors.join(" | "));
  });
  await check("hidden elements are really hidden", async () => {
    const shown = await page.$$eval("[hidden]", els => els.filter(e => getComputedStyle(e).display !== "none").map(e => e.id || e.className));
    assert(!shown.length, "visible despite hidden: " + shown.join(", "));
  });
  await check("nothing covers the scene", async () => {
    const top = await page.evaluate(() => { const r = document.getElementById("media").getBoundingClientRect(); return document.elementFromPoint(r.left + r.width/2, r.top + r.height/2)?.id; });
    assert(top === "ov", "element on top of the scene is #" + top);
  });
  await check("demo room lists its objects", async () => {
    const n = await page.$eval("#rCount", e => e.textContent); assert(n === "9 tracked", n);
  });
  await check("tapping an object selects it", async () => {
    const pt = await page.evaluate(() => { const { S } = window.__faintPull, o = S.objs.find(o => o.cls === "cup"), r = document.getElementById("ov").getBoundingClientRect();
      return { x: r.left + (o.bbox[0] + o.bbox[2]/2) * r.width / S.W, y: r.top + (o.bbox[1] + o.bbox[3]/2) * r.height / S.H }; });
    await page.mouse.click(pt.x, pt.y);
    const name = await page.$eval("#sName", e => e.textContent); assert(name === "cup", "selected " + name);
  });
  await check("floor geometry math", async () => {
    const r = await page.evaluate(() => {
      const { floorDistance: fd } = window.__faintPull;
      return { level30: fd(450, 900, 800, 30, 1.4), below: fd(450 + 800*Math.tan(15*Math.PI/180), 900, 800, 30, 1.4), horizon: fd(450, 900, 800, 1, 1.4) };
    });
    assert(Math.abs(r.level30 - 1.4/Math.tan(Math.PI/6)) < 0.01, "30° down gave " + r.level30);
    assert(Math.abs(r.below - 1.4) < 0.01, "45° ray gave " + r.below);
    assert(r.horizon === null, "near-horizon should be rejected");
  });
  await check("objects on a table take the table's distance", async () => {
    const r = await page.evaluate(() => {
      const { S, distAuto, newObj } = window.__faintPull, saved = { mode: S.mode, objs: S.objs, beta: S.beta, gamma: S.gamma };
      S.mode = "live"; S.frozen = false; S.beta = 60; S.gamma = 0;
      const table = newObj({ cls: "dining table", bbox: [300, 450, 500, 250] }), cup = newObj({ cls: "cup", bbox: [500, 400, 60, 70] });
      S.objs = [table, cup];
      const out = { table: distAuto(table), cup: distAuto(cup) };
      Object.assign(S, saved); return out;
    });
    assert(r.table.src === "floor", "table used " + r.table.src);
    assert(r.cup.src === "on" && Math.abs(r.cup.d - r.table.d) < 1e-9, "cup used " + r.cup.src);
  });

  const img = await testImage();
  if (!img) results.push("SKIP  photo tests (no test image available)");
  else {
    await check("photo mode detects objects", async () => {
      await page.click("#m-photo"); await page.setInputFiles("#file", img);
      await page.waitForFunction(() => /objects/.test(document.getElementById("hudR").textContent) && !/Naming|Detecting/.test(document.getElementById("hudR").textContent), null, { timeout: 120000 });
      const n = await page.evaluate(() => window.__faintPull.S.objs.length); assert(n >= 3, "only " + n + " objects");
      return n + " objects";
    });
    await check("tapping an empty spot identifies it", async () => {
      // pick the spot farthest from every detected box
      const spot = await page.evaluate(() => { const S = window.__faintPull.S; let best = null, bd = -1;
        for (let gy = 0.15; gy < 0.88; gy += 0.03) for (let gx = 0.08; gx < 0.92; gx += 0.03){ const x = gx*S.W, y = gy*S.H;
          const d = Math.min(...S.objs.map(o => Math.hypot(Math.max(o.bbox[0]-x, 0, x-o.bbox[0]-o.bbox[2]), Math.max(o.bbox[1]-y, 0, y-o.bbox[1]-o.bbox[3]))));
          if (d > bd){ bd = d; best = [gx, gy]; } }
        return bd > 20 ? best : null; });
      assert(spot, "no empty spot in the test photo");
      const r = await page.$eval("#ov", e => { const b = e.getBoundingClientRect(); return [b.left, b.top, b.width, b.height]; });
      await page.mouse.click(r[0] + r[2]*spot[0], r[1] + r[3]*spot[1]);
      await page.waitForFunction(() => /Identified|Not sure|unavailable/.test(document.getElementById("hudR").textContent), null, { timeout: 120000 });
      const h = await hud(); assert(!/unavailable/.test(h), h + " / " + await hudB()); return h;
    });
  }

  await check("live camera scans without blocking the page", async () => {
    await page.evaluate(() => { window.__long.length = 0; });
    await page.click("#m-live");
    await page.waitForFunction(() => /Scanning · \d+ fps/.test(document.getElementById("hudR").textContent), null, { timeout: 90000 });
    await page.waitForTimeout(4000);
    const long = await page.evaluate(() => window.__long.slice());
    const worst = long.length ? Math.max(...long) : 0;
    assert(worst < 300, "main thread blocked for " + Math.round(worst) + " ms");
    await page.waitForTimeout(6000);   // give the GPU trial time to finish
    const st = await page.evaluate(() => window.__faintPull.Engine.call("status", {}));
    return (await hud()) + ", worst stall " + Math.round(worst) + " ms, GPU trial: " + JSON.stringify(st);
  });
  await check("live picture fills the scene", async () => {
    const r = await page.evaluate(() => {
      const cv = document.getElementById("cv"), m = document.getElementById("media").getBoundingClientRect(), c = cv.getBoundingClientRect();
      const d = cv.getContext("2d").getImageData(Math.floor(cv.width/2), Math.floor(cv.height/2), 1, 1).data;
      return { sameBox: Math.abs(m.width - c.width) < 1 && Math.abs(m.height - c.height) < 1, lit: d[0] + d[1] + d[2] > 30, visible: getComputedStyle(cv).display !== "none" };
    });
    assert(r.visible && r.sameBox, "canvas doesn't cover the scene box");
    assert(r.lit, "live picture isn't being drawn");
  });
  await check("scrolling doesn't resize the scene", async () => {
    const before = await page.$eval("#media", e => e.style.width + "x" + e.style.height);
    await page.mouse.wheel(0, 900); await page.waitForTimeout(300); await page.mouse.wheel(0, -900); await page.waitForTimeout(300);
    const after = await page.$eval("#media", e => e.style.width + "x" + e.style.height);
    assert(before === after, before + " became " + after);
  });
  await check("freeze and resume", async () => {
    await page.click("#freeze"); assert(/Frozen/.test(await hud()), await hud());
    await page.click("#freeze"); await page.waitForTimeout(300); assert(/Scanning/.test(await hud()), await hud());
  });
  await check("works offline after the first visit", async () => {
    await page.click("#m-demo");
    await page.evaluate(() => navigator.serviceWorker.ready);
    await page.reload(); await page.waitForTimeout(500);
    await page.waitForFunction(() => !!navigator.serviceWorker.controller, null, { timeout: 20000 });
    await ctx.setOffline(true);
    try {
      await page.reload(); await page.waitForTimeout(800);
      const n = await page.$eval("#rCount", e => e.textContent); assert(n === "9 tracked", "offline page shows " + n);
      await page.click("#m-live");
      await page.waitForFunction(() => /Scanning · \d+ fps/.test(document.getElementById("hudR").textContent), null, { timeout: 60000 });
      return "demo and live scanning work offline";
    } finally { await ctx.setOffline(false); }
  });
  await check("no script errors during the run", async () => { assert(errors.length === 0, errors.slice(0, 3).join(" | ")); });

  await browser.close(); server.close();
  console.log(results.join("\n"));
  console.log(failed ? `\n${failed} check(s) failed` : "\nAll checks passed");
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
