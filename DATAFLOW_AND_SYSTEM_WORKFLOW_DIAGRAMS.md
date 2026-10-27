# SentinelVision AI: System Workflow & Data Flow Diagrams (DFD)
### Academic Project Report Documentation (Architecture & Engineering Design)

---

## 1. Executive Summary & Architecture Overview

The **AI Surveillance and Suspicious Activity Detection System (SentinelVision AI)** is an enterprise-grade intelligent video analytics architecture designed to process multi-channel surveillance feeds in real time. The platform integrates:
1. **YOLOv11** high-efficiency person and multi-object spatial localization.
2. **Google MediaPipe Pose 3D** biomechanical kinematic analysis (33 skeletal landmarks and joint angle evaluation).
3. **Calibrated Locomotion & Posture Engine** accurately differentiating `sitting`, `standing`, `walking`, `running`, `sleeping`, and device interactions without jitter.
4. **Hysteresis Debounce Filter** within temporal tracking to eliminate state flipping and single-frame flicker.
5. **Photometrically-Gated Fire & Flame Hazard Engine** combining deep neural classification with chromatic and luminance validation to reject false positives.
6. **Vector Polygon Ray-Casting & Spatial-Temporal Behaviour Engine** for perimeter intrusion, loitering, and violence detection.
7. **Real-Time SOC Command Center** built on FastAPI, React 18, TypeScript, and WebSockets.

---

## 2. High-Level End-to-End System Workflow Architecture

```mermaid
flowchart TD
    subgraph S1["1. Video Ingestion Layer"]
        A1["Physical Webcams (Browser MediaDevices)"]
        A2["CCTV & IP Cameras (RTSP / H.264 / HTTP)"]
        A3["Forensic Video Files (MP4 / AVI / MOV)"]
        A4["Static Forensic Images (JPEG / PNG)"]
    end

    subgraph S2["2. Preprocessing & Optical Enhancement"]
        B1["OpenCV Stream Decoder & Frame Resizer"]
        B2["Luminance & Lux Estimation Engine"]
        B3["Night Vision Processing (Tactical Green / IR / HDR)"]
        B4["Contrast Limited Adaptive Histogram Equalization (CLAHE)"]
    end

    subgraph S3["3. Deep Learning Multi-Model Inference Pipeline"]
        C1["YOLOv11 Neural Detector (Person & Context Objects)"]
        C2["Google MediaPipe Pose 3D (33 Skeletal Landmarks)"]
        C3["Biomechanical Kinematics Engine (Knee, Hip, Spine Angles)"]
        C4["FireNet & YOLOv8 Flame Module (Photometrically Gated)"]
    end

    subgraph S4["4. Tracking, Locomotion & Debouncing Engine"]
        D1["Multi-Person Tracker (IoU + Centroid EMA Smoothing)"]
        D2["Velocity & Spatial Displacement Calculator"]
        D3["Locomotion Classifier (Walking, Running, Stationary)"]
        D4["Multi-Frame Hysteresis Debounce & Voting Filter"]
    end

    subgraph S5["5. Contextual Threat & Behaviour Engine"]
        E1["Ray-Casting Point-in-Polygon Zone Perimeter Check"]
        E2["Loitering & Dwell-Time Counter (30s+ Threshold)"]
        E3["Violence & Fight Anomaly Detection"]
        E4["Safe Posture Guard (Sitting, Standing, Laptop, Phone)"]
    end

    subgraph S6["6. Incident Dispatch & Response Layer"]
        F1["Threat Severity Classifier (CRITICAL, HIGH, MEDIUM, LOW)"]
        F2["Real-Time Audio Siren Synthesizer (Web Audio API)"]
        F3["Alert Broadcast Manager (WebSocket Event Push)"]
        F4["Snapshot Evidence & Forensic Timeline Logger"]
    end

    subgraph S7["7. User Interface & SOC Command Dashboard"]
        G1["React 18 + Vite SOC Command Dashboard"]
        G2["Live Surveillance Quad Wall & Bounding Box Canvas"]
        G3["Forensic Activity Timeline & Video Player"]
        G4["Interactive Restricted Zone Polygon Drawer"]
        G5["Analytics, Spatial Heatmaps & Project Results Portal"]
    end

    %% Workflow Connections
    S1 --> S2
    B1 --> B2 --> B3 --> B4
    B4 --> C1
    C1 --> C2 --> C3
    B4 --> C4
    C1 --> D1
    C3 --> D3
    D1 --> D2 --> D3 --> D4
    D4 --> S5
    E1 --> F1
    E2 --> F1
    E3 --> F1
    E4 --> F1
    C4 --> F1
    F1 --> F2
    F1 --> F3
    F1 --> F4
    F3 --> G1
    F4 --> G3
    D4 --> G2
    G4 --> E1
    F4 --> G5
```

---

## 3. Data Flow Diagram (DFD) Level 0: Context-Level Diagram

The Level 0 DFD illustrates the boundary of the entire **SentinelVision AI System**, depicting external inputs, primary external entities (Surveillance Operator, Camera Stream Feeds, Physical Alarm Systems), and data exchanges.

```mermaid
graph TD
    %% External Entities
    CAM["Video Capture Sources (Webcam, RTSP IP Cameras, File Uploads)"]
    OP["Security Operator / SOC Administrator"]
    SYS["External Alerting & Audio Output Systems"]

    %% Core System Process
    CORE["0.0 SentinelVision AI Surveillance & Suspicious Activity Detection System"]

    %% External Data Flows
    CAM -->|"Raw Video Frames, Base64 Streams, Video Files"| CORE
    OP -->|"Zone Coordinates, Camera Settings, Sensitivity Thresholds"| CORE
    CORE -->|"Live Bounding Boxes, Posture Wireframes, Telemetry HUD"| OP
    CORE -->|"Incident Reports, Analytical Heatmaps, Academic Results"| OP
    CORE -->|"Audio Alarm Triggers, WebSocket Alerts, Siren Notifications"| SYS

    classDef entity fill:#1e293b,stroke:#94a3b8,stroke-width:2px,color:#f8fafc;
    classDef process fill:#0f172a,stroke:#38bdf8,stroke-width:3px,color:#38bdf8;
    class CAM,OP,SYS entity;
    class CORE process;
```

---

## 4. Data Flow Diagram (DFD) Level 1: System Decomposition

The Level 1 DFD decomposes the system into 7 operational processes and 5 persistent data stores.

```mermaid
flowchart TD
    %% External Entities
    ENT_CAM["External Cameras & Webcams"]
    ENT_OP["SOC Security Operator"]
    ENT_ALARM["Audio & Alarm Dispatcher"]

    %% Data Stores
    subgraph DataStores["System Data Stores"]
        DS1[("D1: Camera Registry & Stream Configurations")]
        DS2[("D2: Restricted Polygon Vector Zones")]
        DS3[("D3: Incident Logs & Forensic Audit Dossier")]
        DS4[("D4: AI Neural Weights & Class Maps")]
        DS5[("D5: Forensic Video Archive & Snapshot Cache")]
    end

    %% Processes
    P1["1.0 Stream Ingestion & Night Vision Enhancement"]
    P2["2.0 Neural Object & Person Spatial Localization"]
    P3["3.0 Biomechanical Skeletal & Posture Kinematics"]
    P4["4.0 Temporal Multi-Person Tracking & Debounce Voting"]
    P5["5.0 Contextual Behaviour & Threat Rule Engine"]
    P6["6.0 Incident Classification & Real-Time Alert Dispatch"]
    P7["7.0 Analytics, Spatial Heatmaps & Reporting"]

    %% Data Flow Pathways
    ENT_CAM -->|"Raw Frames / Stream Packets"| P1
    DS1 -->|"Stream URI, RTSP Credentials, Lux Config"| P1
    P1 -->|"Normalized RGB Frame (CLAHE / Night Vision)"| P2
    P1 -->|"Downscaled Candidate Frame (320px)"| P5

    DS4 -->|"YOLOv11 & MediaPipe Weights"| P2
    P2 -->|"Person Bounding Boxes [x, y, w, h]"| P3
    P2 -->|"Context Objects (Laptop, Phone, TV, Cup)"| P4
    P2 -->|"Person Crops & Bounding Boxes"| P4

    P3 -->|"33 3D Landmarks & Joint Angles (Hips, Knees, Spine)"| P4
    P4 -->|"Tracked Person Object (EMA Centroid, Speed, Posture)"| P5

    DS2 -->|"Polygon Coordinates [Xn, Yn]"| P5
    P5 -->|"Threat Event (Fight, Intrusion, Loitering, Fire)"| P6

    P6 -->|"Incident Snapshot & Metadata"| DS3
    P6 -->|"Forensic Video Evidence"| DS5
    P6 -->|"Acoustic Siren Signal"| ENT_ALARM
    P6 -->|"Live WebSocket Push Notification"| ENT_OP

    DS3 -->|"Historical Audit Data"| P7
    DS5 -->|"Archived Keyframes"| P7
    P7 -->|"Heatmaps, Confusion Matrices, Performance Curves"| ENT_OP
    ENT_OP -->|"Define Restricted Zones & Calibration"| DS2
```

---

## 5. Data Flow Diagram (DFD) Level 2: Detailed AI Vision & Locomotion Pipeline

The Level 2 DFD details the exact transformations occurring within the **Deep Learning & Locomotion Classification** subsystem (Process 3.0 & 4.0), showing how raw frames are transformed into debounced, stable activity states without glitching.

```mermaid
flowchart TD
    %% Inputs
    IN_FRAME["Incoming Video Frame (BGR Matrix)"]
    IN_BBOX["Detected Person Bounding Box [x, y, w, h]"]
    IN_OBJS["Interacting Objects (Laptop, Phone, Table)"]

    %% Step 1: Crop and Padding
    subgraph PreCrop["1. Spatial Padding & Normalization"]
        T1["Add 16% Limb Padding Margin (x1, y1, x2, y2)"]
        T2["Scale Crop to Max 384px (Optimization)"]
        T3["Color Conversion: BGR to RGB"]
    end

    %% Step 2: MediaPipe 3D Landmark Extraction
    subgraph PoseExtraction["2. MediaPipe Pose 3D Processing"]
        T4["Extract 33 3D Anatomical Landmarks"]
        T5["Compute Joint Angles: Knee (Hip-Knee-Ankle)"]
        T6["Compute Joint Angles: Hip (Shoulder-Hip-Knee)"]
        T7["Calculate Spine Incline Angle from Vertical"]
        T8["Calculate Upper-Body Vertical Placement in Full Frame"]
    end

    %% Step 3: Biomechanical Decision Logic
    subgraph BioDecide["3. Posture Kinematic Decision Tree"]
        D_SLEEP{"Spine Incline > 45° or Recumbent?"}
        D_ANGLES{"Hips ≥ 134° or Knees ≥ 140°?"}
        D_SIT_ANGLES{"Hips ≤ 126° or Knees ≤ 130°?"}
        D_WEBCAM{"Shoulders Elevated & Aspect ≥ 1.32?"}

        OUT_SLEEP["Candidate Posture: SLEEPING"]
        OUT_STAND_BIO["Candidate Posture: STANDING (Extended Joints)"]
        OUT_SIT_BIO["Candidate Posture: SITTING (Seated Hip Flexion)"]
        OUT_STAND_CAM["Candidate Posture: STANDING (Upper-Body View)"]
        OUT_SIT_CAM["Candidate Posture: SITTING (Desk Occluded)"]
    end

    %% Step 4: Locomotion & Velocity Integration
    subgraph LocomotionEngine["4. Velocity & Locomotion Fusion"]
        V1["Calculate Centroid Displacement: disp = sqrt(dx² + dy²)"]
        V2["Exponential Moving Average: vel = 0.65*disp + 0.35*vel"]
        V_CHECK{"Velocity Magnitude Threshold"}

        OUT_RUN["Activity State: RUNNING (vel ≥ 20.0 px/frame)"]
        OUT_WALK["Activity State: WALKING (5.5 ≤ vel < 20.0 px/frame)"]
        OUT_STATIONARY["Stationary Person (vel < 5.5 px/frame)"]
    end

    %% Step 5: Temporal Hysteresis & Debounce Filter
    subgraph DebounceFilter["5. Rolling Window Hysteresis Debouncing"]
        H1["Append (Activity, Conf) to Rolling 10-Frame History"]
        H2["Stride Pause Guard (Maintain Walking across foot plants)"]
        H3["Posture Debounce (Require 3-4 Consecutive Frames to Flip)"]
        H4["Confidence-Weighted Vote Aggregation"]
        FINAL_ACT["Rock-Solid Stable Output: Activity & Conf"]
    end

    %% Routing
    IN_FRAME --> T1
    IN_BBOX --> T1
    T1 --> T2 --> T3 --> T4
    T4 --> T5 & T6 & T7 & T8

    T7 --> D_SLEEP
    D_SLEEP -->|Yes| OUT_SLEEP
    D_SLEEP -->|No| D_ANGLES
    D_ANGLES -->|Yes| OUT_STAND_BIO
    D_ANGLES -->|No| D_SIT_ANGLES
    D_SIT_ANGLES -->|Yes| OUT_SIT_BIO
    D_SIT_ANGLES -->|Inconclusive| D_WEBCAM
    D_WEBCAM -->|Yes| OUT_STAND_CAM
    D_WEBCAM -->|No| OUT_SIT_CAM

    IN_BBOX --> V1 --> V2 --> V_CHECK
    V_CHECK -->|Fast Motion| OUT_RUN
    V_CHECK -->|Moderate Motion| OUT_WALK
    V_CHECK -->|Stationary| OUT_STATIONARY

    OUT_RUN --> H1
    OUT_WALK --> H1
    OUT_STATIONARY --> H1
    OUT_SLEEP --> H1
    OUT_STAND_BIO --> H1
    OUT_STAND_CAM --> H1
    OUT_SIT_BIO --> H1
    OUT_SIT_CAM --> H1
    IN_OBJS --> H1

    H1 --> H2 --> H3 --> H4 --> FINAL_ACT
```

---

## 6. Photometric Fire Hazard Detection Data Flow

This diagram documents the two-stage flame hazard pipeline that eliminates false positives on warm clothes, skin tones, and ambient lighting:

```mermaid
flowchart TD
    A["Raw Frame (RGB/BGR)"] --> B{"Is detect_fire Enabled?"}
    B -->|"False (Webcam Default)"| C["Immediate Bypass: Return Fire=False, Boxes=[] (0ms CPU Cost)"]
    B -->|"True (Explicit Mode)"| D["YOLOv8 Flame Object Detection (imgsz=320, conf=0.50)"]

    D --> E["Extract Candidate Bounding Box Crop"]
    E --> F["Convert Crop to HSV and Compute Color Means"]

    F --> G{"Luminance & Flame Check"}
    G -->|"Brightness V < 115 or Red R < 115"| H["Reject False Positive (Clothes / Skin / Wood)"]
    G -->|"Red Channel Dominance R > B + 15"| I["Accept Candidate Flame Box"]

    I --> J{"Accumulate Detections Across Frames"}
    J -->|"Consecutive Frames < 3"| K["Transient Alert Indicator"]
    J -->|"Consecutive Frames ≥ 3"| L["Trigger CRITICAL Fire Incident Siren & Dispatch Alert"]
```

---

## 7. System Sequence Diagram: Live Client-Server Interaction

The sequence diagram below illustrates the real-time interaction between the Web Browser, FastAPI Backend, AI Pipeline, and Alert Engine during live surveillance monitoring:

```mermaid
sequenceDiagram
    autonumber
    actor Operator as SOC Operator
    participant UI as React Frontend (LiveWebcam.tsx)
    participant API as FastAPI Router (/api/detection/frame)
    participant Pipeline as InferencePipeline
    participant YOLO as YOLOv11 Detector
    participant Pose as MediaPipe Pose 3D
    participant Tracker as MultiPersonTracker
    participant Alert as AlertEngine
    participant Audio as Web Audio Synthesizer

    Operator->>UI: Clicks "START LIVE WEBCAM"
    UI->>UI: Initializes navigator.mediaDevices.getUserMedia()
    loop Every 60ms (Adaptive Adaptive Low-Latency Loop)
        UI->>UI: Captures Frame on HTML5 Canvas (640x360 JPEG)
        UI->>API: POST /api/detection/frame (base64, night_vision, detect_fire=false)
        API->>Pipeline: process_frame(frame, accuracy_mode, detect_fire)
        Pipeline->>YOLO: detect(frame, imgsz=384, conf=0.35)
        YOLO-->>Pipeline: Returns [people_bboxes, scene_objects, fire_clean]
        Pipeline->>Tracker: update(people_bboxes)
        Tracker-->>Pipeline: Returns [TrackedObjects with EMA velocity]
        loop For Each Tracked Person
            Pipeline->>Pose: analyze_person(frame, bbox)
            Pose-->>Pipeline: 33 3D Landmarks, Joint Angles, Posture
            Pipeline->>Tracker: update_activity(raw_activity, confidence)
            Tracker-->>Pipeline: debounced_activity (Sitting/Standing/Walking/Running)
        end
        Pipeline->>Alert: analyze_person(bbox, activity, zones, history)
        Alert-->>Pipeline: Threat Level & Generated Alerts
        Pipeline-->>API: Processed Results JSON (Tracks, Landmarks, Angles, Alerts)
        API-->>UI: 200 OK JSON Payload
        UI->>UI: Render Bounding Boxes, Skeleton Wireframes, Telemetry HUD
        opt Threat Detected (e.g., Fighting or Perimeter Intrusion)
            UI->>Audio: playCriticalAlarm()
            Audio-->>Operator: Emits Acoustic Tactical Siren Tone
        end
    end
    Operator->>UI: Clicks "STOP WEBCAM"
    UI->>UI: Releases MediaStream tracks & clears state
```

---

## 8. Summary Table of Subsystem Specifications

| Subsystem Component | Framework / Technology | Primary Operational Function | Output Metric / Artifact |
| :--- | :--- | :--- | :--- |
| **Stream Ingestion** | OpenCV 4.8 / Web MediaDevices | Decodes RTSP, WebRTC, MP4, and JPEG frames at target dimensions. | Standard BGR matrix ($640 \times 360$ / $1280 \times 720$). |
| **Night Vision Engine** | Custom NumPy + CLAHE | Dynamic luminance sensor, lux estimation, and tactical green/IR filters. | Enhanced low-light frame + lux telemetry. |
| **Object & Person Detector** | YOLOv11-Nano (Ultralytics) | Spatial localization of people and everyday contextual objects. | Normalized bounding boxes $[x, y, w, h]$ with confidence. |
| **Biomechanical Engine** | Google MediaPipe Pose 3D | 33 3D skeletal landmark extraction and trigonometric joint angle computation. | Anatomical joint degrees (hip, knee, spine incline). |
| **Locomotion Engine** | Displacement Vector Math | Frame-to-frame centroid velocity with scale-invariant normalization. | Velocity (px/frame) and classified locomotion state. |
| **Temporal Debouncer** | Dwell-Time Hysteresis | 10-frame voting window with state-dwell locks to stop flip-flop jitter. | Stabilized activity tag (`sitting`, `standing`, `walking`, `running`). |
| **Fire Hazard Module** | YOLOv8 + HSV Chromatic Filter | Dual deep learning and photometric verification of incandescent flame. | Incandescent flame box + continuous alert flag. |
| **Behavior & Zone Engine** | 2D Vector Ray-Casting | Point-in-polygon intrusion testing, loitering counters, and violence checks. | Threat level (`NORMAL`, `SUSPICIOUS`, `CRITICAL`). |
| **Alert Engine** | Python Async Dispatcher | Thread-safe threat deduplication, audit persistence, and WebSocket broadcasting. | JSON Alert Payload + Evidence Snapshot. |
| **SOC Dashboard** | React 18, TypeScript, Tailwind | Live surveillance wall, forensic timeline, zone designer, and analytics. | Interactive Web Client HUD at 30+ FPS. |

---

## 9. Academic Report Inclusion Instructions

For inclusion in a formal thesis, capstone report, or conference submission:
1. **Chapter 3 (System Analysis & Requirement Specifications)**: Embed the **Level 0 and Level 1 Data Flow Diagrams (Section 3 & Section 4)** to illustrate procedural data interchange.
2. **Chapter 4 (System Architecture & Design)**: Embed the **High-Level System Workflow Diagram (Section 2)** and **Sequence Diagram (Section 7)** to demonstrate asynchronous service topology.
3. **Chapter 5 (Implementation & Algorithmic Modules)**: Embed the **Level 2 Locomotion & Posture Kinematic Flowchart (Section 5)** and **Photometric Fire Hazard Flowchart (Section 6)** to document the novel algorithmic contributions for zero-jitter posture classification and false-alarm suppression.
