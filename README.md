# Multi-Factor Authentication System

A robust **AI-powered Multi-Factor Authentication (MFA)** system that combines **Face Recognition**, **Voice Analysis**, **Liveness Detection**, and **Lip Sync Verification** to provide high-security, real-time, and user-friendly authentication.

---

## Features

- **Face Recognition** using deep learning and real-time webcam input
- **Voice Matching** and speech-to-text conversion
- **Liveness Detection** with randomized text prompts and response validation
- **Lip Sync Verification** to ensure the speaker matches the lip movement
- **AI-powered biometric security** with anti-spoofing
- **Local data storage** (Cloud integration planned)

---

## Technical Stack

| Category          | Technologies/Tools                                                                 |
|-------------------|------------------------------------------------------------------------------------|
| **Language**       | Python                                                                             |
| **Libraries**      | `OpenCV`, `face-recognition`, `whisper`, `fuzzywuzzy`, `noisereduce`, `mediapipe`, `sounddevice`, `soundfile` |
| **Storage**        | Local directory-based storage (planning migration to cloud)                       |
| **AI/ML/DL**       | Used for face encoding, voice embedding, and dynamic lip movement verification    |

---

## Authentication Flow

```text
[1] User opens the app 
      ↓  
[2] Face Recognition → Verified?  
      ↓  
[3] Random Text Prompt Generated  
      ↓  
[4] User Speaks Prompt → Voice Verified?  
      ↓  
[5] Lip Movements Verified (Lip Sync Match)?  
      ↓  
[6] Authentication Success / Denied
