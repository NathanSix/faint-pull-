# Faint Pull

A gravity scanner for iPhone. Point the camera at a room and it estimates how hard each object pulls on you with gravity.

Live site: https://nathansix.github.io/faint-pull-/

- **Detection:** TensorFlow.js COCO-SSD (80 object types), with weights vendored in `model/coco-ssd/`.
- **Tap to identify:** MobileNet (1,000 ImageNet classes), downloaded from TF Hub the first time you use it.
- **Instruments:** accelerometer g with two-sided calibration, a tilt rangefinder, and today's Sun and Moon pulls.

Everything runs on the phone. Camera frames are never uploaded.
