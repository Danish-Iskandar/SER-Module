# Adaptive VR Training System for Quay Crane Operators

![Unity](https://img.shields.io/badge/Unity-VR-black?logo=unity)
![Python](https://img.shields.io/badge/Python-SER%20Backend-3776AB?logo=python&logoColor=white)
![WebSocket](https://img.shields.io/badge/Socket.io-WebSocket-010101?logo=socketdotio)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)

An emotion-aware Virtual Reality training system for Ship-to-Shore (STS) quay crane operators. It uses **voice-based Speech Emotion Recognition (SER)** to detect the trainee's emotional state and dynamically adjust training scenarios in real time.

> Final Year Project, Bachelor of Computer Science (Hons.), Universiti Teknologi PETRONAS (UTP)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Emotion Classes](#emotion-classes)
- [Tech Stack](#tech-stack)
- [Datasets](#datasets)
- [Instructor Dashboard](#instructor-dashboard)
- [Development Methodology](#development-methodology)
- [Evaluation Plan](#evaluation-plan)
- [Project Status](#project-status)
- [Scope](#scope)
- [Getting Started](#getting-started)
- [Future Work](#future-work)
- [References](#references)
- [Author](#author)

---

## Overview

Quay crane operators perform millimetre-precision container transfers from cabins about 40 metres in the air, under sustained cognitive load, fatigue, and environmental stress. VR simulators are the current standard for zero-risk training, but they are **passive**: they scale difficulty linearly without considering the trainee's psychological or emotional state.

This project bridges that gap. The system listens to the operator through the VR headset microphone, classifies their emotional state, and adapts the simulation accordingly, treating an overwhelmed novice differently from a confident expert.

## Features

- **Speech Emotion Recognition:** MFCC feature extraction with a CNN-LSTM classifier
- **Real-time adaptation:** scenario difficulty adjusts based on detected emotion and confidence level
- **Emotion-reactive avatar:** a 3D operator avatar in Unity that changes colour with the detected state (e.g. red for stress)
- **Confidence thresholding:** prevents the detected state from flickering on noisy or ambiguous input
- **Instructor dashboard:** live emotional state, rolling emotion timeline, and an auditable adaptive event log
- **Non-intrusive sensing:** voice only, with no biosensors required

## System Architecture

```
Voice Input (VR headset microphone, short overlapping frames)
        │
        ▼
SER Processing (MFCC extraction + CNN-LSTM classification)
        │
        ▼
Middleware (Socket.io / WebSocket broadcasts emotion state as JSON)
        │
        ├──► Avatar Colour      (changes with emotion state)
        ├──► Adaptive Engine    (adjusts scenario parameters)
        └──► Instructor Dashboard (live state + session logs)
```

| Layer | Description |
| --- | --- |
| **Input** | Audio captured from the VR headset microphone and segmented into short, overlapping frames for real-time updates. |
| **Processing** | Each frame is converted to MFCC features and passed to the CNN-LSTM model, which outputs a probability distribution across emotion classes. A confidence threshold decides whether to accept a new state or retain the previous one. |
| **Middleware** | A Socket.io/WebSocket server connects the Python SER backend to Unity, publishing lightweight JSON messages. |
| **Output** | Avatar colour change, adaptive scenario adjustments (e.g. reducing wind speed when frustration crosses a threshold, or enabling a voice-assistant tip when stress is detected), and logging to the instructor dashboard. |

## Emotion Classes

| Class | Description |
| --- | --- |
| Neutral | Baseline operator state |
| Stressed / Anxious | Elevated stress, may trigger supportive adaptations |
| Frustrated | May trigger reduced difficulty (e.g. lower wind speed) |
| Confident | Operator is coping well, difficulty can be maintained or increased |

## Tech Stack

- **Simulator and avatar:** Unity (C#)
- **SER backend:** Python (CNN-LSTM, MFCC features)
- **Middleware:** Socket.io / WebSocket (JSON messaging)
- **Dashboard:** 2D secondary-display monitoring interface

## Datasets

The SER model is trained on three widely benchmarked emotional speech datasets:

- **RAVDESS:** Ryerson Audio-Visual Database of Emotional Speech and Song
- **IEMOCAP:** Interactive Emotional Dyadic Motion Capture
- **SAVEE:** Surrey Audio-Visual Expressed Emotion

## Instructor Dashboard

The dashboard runs on a secondary 2D display so instructors can monitor trainees without disrupting VR immersion.

- **Status indicator cards:** current emotional state, stress index, fatigue index, and adaptation cycle timing
- **Real-time emotion line chart:** stress, frustration, and confidence plotted over a rolling 3-minute window
- **Adaptive event log:** timestamped record of every automatic adaptation, for post-session debriefing

## Development Methodology

The project follows an **iterative, incremental** approach so the SER model, avatar, and middleware can be built and tested in parallel.

| Phase | Focus |
| --- | --- |
| **1. Data Preparation and Model Training** | Extract MFCC features, train and validate the CNN-LSTM, tune confidence thresholds using accuracy and confusion matrices |
| **2. Avatar and Middleware Development** | Build the Unity avatar and the Socket.io/WebSocket bridge to the SER backend |
| **3. System Integration** | Integrate SER, avatar, and dashboard into the Unity quay crane simulator with the adaptive engine |
| **4. Evaluation** | Within-subject comparison of adaptive vs. non-adaptive training |

## Evaluation Plan

A within-subject study will compare adaptive and non-adaptive training sessions, measuring:

- Operator performance KPIs
- Perceived cognitive load via the **NASA Task Load Index (NASA-TLX)**

## Project Status

| Component | Status |
| --- | --- |
| Unity avatar | Prototype complete (colour-based emotion feedback) |
| SER module | In active development (MFCC pipeline, model training, threshold tuning) |
| Instructor dashboard | Designed, not yet integrated with live SER output |
| System integration | Planned |
| Evaluation | Planned |

## Scope

- **VR development:** limited to the 3D operator avatar with colour-based emotion feedback. Crane mechanics and physics simulation are part of the existing simulator and out of scope.
- **SER module:** audio capture, MFCC extraction, and CNN-LSTM classification into four emotion classes.
- **Dashboard:** live monitoring and historical session logging for post-training review.

## Getting Started

> Setup instructions will be added as the project develops.

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

## Future Work

- Expand avatar expressiveness from colour changes to full facial blendshape animation
- Add physiological signals (e.g. heart rate variability) alongside voice to cross-validate predictions
- Evaluate with real or trainee quay crane operators for greater ecological validity
- Generalise the approach to other high-risk operator roles

## References

- Dhalmahapatra, K., Singh, S., Niyogi, S., & Vinod Kumar, M. (2021). Assessment of virtual reality based safety training simulator for electric overhead crane operations. *Safety Science, 139*, 105241.
- Nasri, M. (2025). Towards intelligent VR training: A physiological adaptation framework for cognitive load and stress detection. *arXiv*. https://arxiv.org/pdf/2504.06461
- Ouyang, Q. (2025). Speech emotion detection based on MFCC and CNN-LSTM architecture. *arXiv*. https://arxiv.org/abs/2501.10666
- Scorgie, D., Feng, Z., Paes, D., Parisi, F., Yiu, T. W., & Lovreglio, R. (2024). Virtual reality for safety training: A systematic literature review and meta-analysis. *Safety Science, 171*, 106372.

## Author

**Muhammad Danish Iskandar Bin Zahrul-Ikram**
Bachelor of Computer Science (Hons.)
Universiti Teknologi PETRONAS

## Supervisor

Assoc. Prof. Ts. Dr. Noreen Izza Bt Arshad
Associate Professor (Computing)
Faculty of Science, Management & Computing
Universiti Teknologi PETRONAS
