# Faint Pull

A gravity scanner for iPhone. Point the camera at a room and it estimates how hard each object pulls on you with gravity.

Live site: https://nathansix.github.io/faint-pull-/

## How it works
- **Scanning:** MediaPipe EfficientDet-Lite0 finds 80 common object types, running in a Web Worker so the camera view stays smooth.
- **Photos:** EfficientDet-Lite2, which is slower but more accurate.
- **Tap to identify:** EfficientNet-Lite2 names 1,000 kinds of things. Masses come from `src/imagenet-masses.txt`.
- **Instruments:** accelerometer g with two-sided calibration, a tilt rangefinder, and today's Sun and Moon pulls.

Everything runs on the phone. The bundled MediaPipe library has its usage logging switched off, so nothing is sent anywhere.

## Editing
Edit `src/page.html` (markup and styles) or `src/app.js` (logic), then run `python3 src/build.py` to rebuild `index.html`.
Model files live in `models/` and are downloaded by the "Fetch on-device models" workflow.
