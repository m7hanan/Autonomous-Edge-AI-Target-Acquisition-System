# Autonomous Face-Tracking Vision System (Phase 1)

![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%205-c51a4a?logo=raspberrypi&logoColor=white)
![OS](https://img.shields.io/badge/OS-Raspberry%20Pi%20OS%2064--bit%20(Trixie)-A81D33?logo=debian&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-DNN%20Module-5C3EE8?logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/status-Phase%201%20Complete-success)
![License](https://img.shields.io/badge/license-MIT-blue)

Real-time, low-latency facial centroid tracking on edge hardware. This system runs a deep learning detection pipeline entirely on-device, computes a corrected landmark offset for stable subject-centroid lock, and emits smoothed (X, Y) error telemetry intended to drive a future pan-tilt servo gimbal.

---

## 1. Project Overview

This project is **Phase 1** of a two-phase autonomous camera-tracking system. The goal of this phase was to solve the perception and signal-stability problem — reliable, jitter-free face tracking at real-time frame rates on constrained edge compute — before any mechanical actuation is introduced.

Core capabilities delivered in this phase:

- On-device deep learning face detection (no cloud dependency, no network round-trip)
- A custom geometric correction that re-centers the tracking point from the default detector centroid to a stable, consistent facial landmark
- A tracking reticle overlay rendered live with detector confidence telemetry
- An anti-jitter deadzone filter that converts noisy per-frame bounding boxes into smooth, gimbal-safe (X, Y) error signals

The output of this phase — clean `(Delta X, Delta Y)` telemetry — is the direct input to the Phase 2 servo control loop.

---

## 2. Tech Stack

| Layer | Component |
|---|---|
| **Compute** | Raspberry Pi 5 (8GB) |
| **Vision Input** | Standard USB Web Camera |
| **Display** | Waveshare 10.1" Capacitive Touch External Display |
| **OS** | Raspberry Pi OS 64-bit (Debian "Trixie") |
| **Language** | Python 3 |
| **CV / Math** | OpenCV (`cv2`), NumPy |
| **AI Model** | ResNet-10 Single Shot MultiBox Detector (SSD), Caffe-trained, run via OpenCV's DNN module |

---

## 3. Key Engineering Features & Logic

### 3.1 High-FPS Edge Detection — Haar Cascade → DNN SSD Migration
The initial prototype used Haar Cascade classifiers. In practice, Haar Cascades exhibited significant detection dropout during rapid subject movement and side-profile rotation, which is unacceptable for a system feeding a downstream physical actuator. I replaced the classifier with the **OpenCV DNN module running a ResNet-10 SSD (Caffe)**, which maintains lock through fast lateral motion and partial profile turns while sustaining real-time inference on the Pi 5 CPU.

### 3.2 Facial Landmark Offset Correction
By default, the SSD bounding box centroid tracks roughly to the nose/mid-face region. For a stable, repeatable lock point (required for downstream gimbal aiming accuracy), I engineered a deterministic vertical offset to bias the tracking point upward within the bounding box:
