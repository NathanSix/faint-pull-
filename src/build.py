"""Builds index.html from src/page.html + src/app.js. Run: python3 src/build.py"""
import os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = lambda f: open(os.path.join(root, 'src', f), encoding='utf-8').read()
js = src('app.js')
js = js.replace('__IMN_SRC__', '|'.join(l.strip() for l in src('imagenet-masses.txt').splitlines() if l.strip()))
js = js.replace('__IN_LABELS__', '|'.join(l.strip() for l in src('imagenet-labels.txt').splitlines()))
frag = src('page.html').rstrip() + '\n\n<script>\n' + js + '</script>\n'
cut = frag.index('<div class="wrap">')
full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<meta name="theme-color" content="#07070C">\n' + frag[:cut] + '</head>\n<body>\n' + frag[cut:] + '</body>\n</html>\n')
open(os.path.join(root, 'index.html'), 'w', encoding='utf-8').write(full)
# service worker: a new build id makes phones fetch the new app shell
import hashlib
h = hashlib.sha256(full.encode())
for f in ['engine.js', 'lib/mediapipe/vision_bundle.mjs', 'manifest.webmanifest']:
    h.update(open(os.path.join(root, f), 'rb').read())
build = h.hexdigest()[:10]
lh = hashlib.sha256()
for f in ['lib/mediapipe/vision_bundle.mjs', 'lib/mediapipe/wasm/vision_wasm_module_internal.js', 'lib/mediapipe/wasm/vision_wasm_module_internal.wasm']:
    lh.update(open(os.path.join(root, f), 'rb').read())
open(os.path.join(root, 'sw.js'), 'w', encoding='utf-8').write(src('sw.js').replace('__BUILD__', build).replace('__LIBHASH__', lh.hexdigest()[:10]))
os.makedirs(os.path.join(root, 'dist'), exist_ok=True)
open(os.path.join(root, 'dist', 'fragment.html'), 'w', encoding='utf-8').write(frag)
print('built index.html', len(full), 'bytes, build', build)
