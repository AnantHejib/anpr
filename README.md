# TRINETRA-VIG · Urban Vehicle Intelligence Platform
### Smart India Hackathon 2026 | Problem Statement ID: 26127
**Title:** City-Wide AI Engine for Multi-Camera ANPR Trajectory Tracking and Urban Traffic Analytics  
**Theme:** Smart Automation | **Team:** Tech_Innovation_Crew  
**Jurisdiction:** MH-14 (Lonavala · Maval · Pimpri-Chinchwad Corridor)  
**Hardware Integration:** Luxonis OAK-D Pro Spatial AI Camera (DepthAI v3)  

---

## 🌟 Overview & Key Innovation

Traditional ANPR systems fail in real-world urban networks because they evaluate isolated camera frames with greedy decision-making. A blurred character or low-light reflection breaks the vehicle trail, while cloned license plates slip through unchecked.

**TRINETRA-VIG** connects uncertain camera sightings into a verified, explainable vehicle journey:
- **Top-K Candidate Hypotheses:** Keeps multiple character-level plate hypotheses rather than forcing a single brittle answer (e.g., `KA03MN11B8` at 46% vs `KA03MN1188` at 42%).
- **Visual Appearance ReID:** Matches vehicle color, body classification (Sedan, SUV, Commercial, 2-Wheeler), and feature embeddings.
- **Road Network & Time Feasibility:** Assesses road distances and transit times to compute physical feasibility ($v = \frac{\Delta d}{\Delta t}$).
- **Retroactive Weak-Read Repair (Key Differentiator):** Later high-confidence evidence from downstream cameras **repairs** earlier ambiguous sightings while preserving original raw sensor readings in an immutable audit trail.
- **Cloned Plate Anomaly Detection:** Instantly flags impossible travel times (e.g., 27.2 km in 0.8 min $\rightarrow$ required speed > 2000 km/h) or visual mismatches on identical plates.

---

## 📷 Hardware: Luxonis OAK-D Pro Integration

The prototype interfaces directly with the connected **Luxonis OAK-D Pro** camera using the **DepthAI v3** API (`depthai 3.10.0`):
- **RGB Sensor:** High-resolution color stream processed in real-time with YOLOv8 vehicle detection and plate localization.
- **Active Stereo IR:** Infrared dot projector and flood illuminator for low-light and nighttime license plate capture.
- **Hardware Telemetry:** Real-time monitoring of device ID, USB 3.0 throughput, temperature, and frame rates (up to 30 FPS).
- **Edge Deployment:** Operates as a local command post / checkpoint camera seamlessly integrated into the city-wide network.

---

## 🚀 Running the Prototype

1. **Start the TRINETRA-VIG Server:**
   ```bash
   python app.py
   ```
2. **Access the Web Dashboard:**
   Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

---

## 🎯 Interactive PPT Demonstration Scenarios

The web dashboard includes interactive one-click scenario buttons corresponding to the presentation slides:

| Scenario | Slide Reference | Description | Observed Result |
| :--- | :--- | :--- | :--- |
| **1. Weak-Read Retroactive Repair** | Slide 2 & Slide 6 | Camera 1 records ambiguous plate `KA03MN11B8` (46%). Downstream Camera 2 records `KA03MN1188` (94%). | Identity Graph resolves journey to `KA03MN1188` with **CONFIRMED (94%)**. Camera 1 reading repaired; raw read preserved in audit log. |
| **2. Cloned Plate Detection** | Slide 5 | Plate `DL01AB1234` observed at Camera 1 (White Sedan) and Camera 3 (Dark SUV) within 50 seconds (9.5 km apart). | **CRITICAL ALERT:** Cloned plate detected due to impossible travel time (> 680 km/h) and visual clash. |
| **3. Police Watchlist Interception** | Slide 3 & Slide 4 | Vehicle `MH14DX9921` detected at Maval Junction. | **HIGH-PRIORITY ALERT:** Flagged by Maharashtra Police (Stolen Vehicle FIR #412/2026). Recommended officer action displayed. |
| **4. Live OAK-D Pro Capture** | Hardware Demonstration | Snaps a live frame from the user's connected Luxonis OAK-D Pro camera. | Detects vehicle in view, runs color/class inference, extracts plate hypotheses, and injects sighting into the live identity graph. |

---

## 🗺️ PostGIS Spatial Intelligence & Topographic Terrain Engine

TRINETRA-VIG integrates an enterprise-grade **PostGIS Spatial Engine** (powered by GEOS 3.12 and Shapely 2.1) configured for **EPSG:4326 (WGS84)** and projected Web Mercator:
- **Native PostGIS Geometries:** Point geometries (`ST_MakePoint`), FOV coverage cone sector polygons (`ST_Buffer_Cone`), highway corridor LineStrings, and 4D spatio-temporal trajectories (`LineString ZM` with coordinates, elevation $Z$, and epoch timestamps $M$).
- **Spatial Calculations:** Geodesic Great-Circle distance (`ST_DistanceSphere`), spatial proximity queries (`ST_DWithin`), and road corridor lengths (`ST_Length`).
- **Topographic Terrain Mapping:** Default high-contrast mountain terrain layer (Esri World Topographic DEM & OpenTopoMap) displaying 10-meter elevation contours, hillshading, mountain passes, and escarpments across the **Western Ghats (MH-14)**.
- **Dynamic Current Location & GPS Sync:** The dashboard pins the active **Mobile Command Post** (`ST_MakePoint(73.4100, 18.7530)` at $622\text{m}$ ASL) and supports 1-click GPS synchronization with the operator's laptop via the HTML5 Geolocation API.
- **Elevation Profile & Ghat Gradient:** Interactively models the altitude climb from Khopoli ($180\text{m}$) up to Khandala ($550\text{m}$) and Lonavala ($622\text{m}$), calculating slope gradient and physics-based vehicle transit feasibility ($v = \frac{\Delta d}{\Delta t}$).
- **Interactive PostGIS SQL Terminal:** Built-in SQL terminal on the dashboard for evaluators and officers to test live PostGIS queries in real time.

### PostGIS REST API Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/postgis/status` | `GET` | PostGIS spatial engine health, SRID, layers, and active Command Post position. |
| `/api/postgis/cameras/geojson` | `GET` | Camera nodes and camera FOV coverage cone polygons as standard PostGIS GeoJSON. |
| `/api/postgis/corridors/geojson` | `GET` | Highway corridors and mountain passes as PostGIS GeoJSON LineStrings with speed limits. |
| `/api/postgis/trajectories/geojson`| `GET` | Reconstructed 4D vehicle trajectories (`LineString ZM`) with elevation and speed vectors. |
| `/api/postgis/location/current` | `GET` | Active Command Post location, elevation, and distance/bearing vectors to all cameras. |
| `/api/postgis/location/set` | `POST` | Sets Command Post coordinates (from laptop GPS, map click, or hardware rig). |
| `/api/postgis/spatial_query/within_radius` | `POST` | Executes `ST_DWithin` spatial query returning matching cameras and buffer circle polygon. |
| `/api/postgis/terrain/elevation_profile` | `GET` | Topographic elevation slice, min/max altitude, and mountain slope gradient data. |
| `/api/postgis/sql_console` | `POST` | Executes custom spatial SQL queries and returns tabular results with execution times. |

---

## 🏗️ Architecture & Technology Stack

- **Capture & Edge AI:**
  - Hardware: Luxonis OAK-D Pro (3-Camera Spatial Perception: Center Color RGB, Left IR Mono, Right IR Mono, StereoDepth, Active IR Laser)
  - Vision Pipeline: Ultralytics YOLOv8, OpenCV 4.13, Local HSRP Character OCR Template Matcher (60 FPS Decoupled Worker)
- **Spatial Intelligence & GIS:**
  - PostGIS Spatial Engine: WGS84 ST_* geometry algorithms, R-Tree indexing, Digital Elevation Model (DEM)
  - Topographic Map: Leaflet.js with Esri World Topo, OpenTopoMap, Satellite Hybrid, and Dark Matter
- **Backend Intelligence & Fusion:**
  - FastAPI, Python 3.14, Uvicorn
  - Temporal graph reasoning & road feasibility engine
- **Frontend Command Center:**
  - Glassmorphism tactical dashboard with 3D polar spatial radar, live telemetry, and theater mode
  - Explainable AI Reasoning Inspector with real-time telemetry
- **Audit & Governance:**
  - Preserved raw sensor readings for judicial evidence
  - Human-in-the-loop action controls (Verify / Dispatch / Dismiss)

