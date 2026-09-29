import time
import os
import cv2
import json
import logging
from typing import Optional
from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.camera_manager import CameraManager, CITY_CAMERAS
from core.anpr_engine import ANPREngine
from core.evidence_fusion import EvidenceFusionEngine
from core.analytics_engine import AnalyticsEngine
from core.vahan_service import VahanService
from core.postgis_engine import PostGISSpatialEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TRINETRA.App")

app = FastAPI(
    title="TRINETRA-VIG Urban Vehicle Intelligence Engine",
    description="SIH 2026 Problem Statement 26127: Multi-Camera ANPR Trajectory Tracking & Urban Analytics",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Subsystems
cam_mgr = CameraManager()
anpr_engine = ANPREngine()
fusion_engine = EvidenceFusionEngine()
analytics_engine = AnalyticsEngine()
postgis_engine = PostGISSpatialEngine()

# Seed PPT Demonstration Scenarios
def seed_initial_demonstration_data():
    now = time.time()
    
    # 1. PPT Primary Scenario: The Ambiguous Weak-Read & Retroactive Repair
    # Camera A (Toll Plaza): Weak read KA03MN11B8 (46%) vs KA03MN1188 (42%)
    sighting_1 = {
        "camera_id": "CAM_01",
        "timestamp": now - 780, # 13 mins ago
        "class": "Car (Sedan)",
        "color": "White",
        "hypotheses": [
            {"plate": "KA03MN11B8", "confidence": 0.46, "rank": 1},
            {"plate": "KA03MN1188", "confidence": 0.42, "rank": 2},
            {"plate": "KA03MN1108", "confidence": 0.12, "rank": 3}
        ]
    }
    fusion_engine.ingest_sighting(sighting_1)

    # Camera B (Kumar Resort): Clear read KA03MN1188 (94%) -> REPAIRS Camera A!
    sighting_2 = {
        "camera_id": "CAM_02",
        "timestamp": now - 120, # 2 mins ago (6.2 km in 11 mins = 33.8 km/h)
        "class": "Car (Sedan)",
        "color": "White",
        "hypotheses": [
            {"plate": "KA03MN1188", "confidence": 0.94, "rank": 1},
            {"plate": "KA03MN11B8", "confidence": 0.04, "rank": 2},
            {"plate": "KA03MN118B", "confidence": 0.02, "rank": 3}
        ]
    }
    fusion_engine.ingest_sighting(sighting_2)

    # 2. PPT Cloned Plate Scenario (Slide 5)
    # Cam A: White Sedan DL01AB1234 at 08:12 AM
    # Cam B: Dark SUV DL01AB1234 at 08:13 AM (27.2 km in 0.8 min = impossible!)
    sighting_clone_1 = {
        "camera_id": "CAM_01",
        "timestamp": now - 300,
        "class": "Car (Sedan)",
        "color": "White",
        "hypotheses": [
            {"plate": "DL01AB1234", "confidence": 0.92, "rank": 1}
        ]
    }
    fusion_engine.ingest_sighting(sighting_clone_1)

    sighting_clone_2 = {
        "camera_id": "CAM_03",
        "timestamp": now - 250, # only 50 seconds later across 9.5 km!
        "class": "SUV",
        "color": "Dark SUV / Grey",
        "hypotheses": [
            {"plate": "DL01AB1234", "confidence": 0.91, "rank": 1}
        ]
    }
    fusion_engine.ingest_sighting(sighting_clone_2)

    # 3. PPT Watchlist Hit (Stolen Vehicle MH14DX9921)
    sighting_watch = {
        "camera_id": "CAM_04",
        "timestamp": now - 60,
        "class": "Car (Sedan)",
        "color": "Silver",
        "hypotheses": [
            {"plate": "MH14DX9921", "confidence": 0.96, "rank": 1}
        ]
    }
    fusion_engine.ingest_sighting(sighting_watch)

seed_initial_demonstration_data()

# Video Streaming Generator with 3D Spatial Perception & Odometry HUD (60 FPS Optimized)
def generate_mjpeg_stream(camera_id: str, mode: str = "rgb_hud"):
    """Encodes real-time camera frames with multi-camera perception, 3D odometry, and tactical HUD at up to 60 FPS."""
    last_ingest_time = 0.0
    while True:
        try:
            if camera_id == "CAM_OAKD":
                if mode == "depth_color":
                    annotated = cam_mgr.get_depth_colored_frame()
                elif mode == "left_ir":
                    annotated = cam_mgr.get_left_mono_frame()
                elif mode == "right_ir":
                    annotated = cam_mgr.get_right_mono_frame()
                elif mode == "triple_cam":
                    raw_rgb = cam_mgr.get_raw_frame("CAM_OAKD")
                    anpr_engine.submit_frame(raw_rgb)
                    det = anpr_engine.get_latest_detection()
                    top_bbox = det["vehicles"][0]["bbox"] if det["vehicle_detected"] else None
                    spatial_info = cam_mgr.update_spatial_odometry(top_bbox)
                    center_hud = anpr_engine.annotate_frame_with_hud(
                        raw_rgb, det, camera_name="CAM_OAKD", fps=cam_mgr.oakd_fps, spatial_info=spatial_info
                    )
                    annotated = cam_mgr.get_triple_panorama_frame(center_hud)
                else:
                    # Default: "rgb_hud"
                    raw_frame = cam_mgr.get_raw_frame("CAM_OAKD")
                    anpr_engine.submit_frame(raw_frame)
                    det_result = anpr_engine.get_latest_detection()
                    top_bbox = det_result["vehicles"][0]["bbox"] if det_result["vehicle_detected"] else None
                    spatial_info = cam_mgr.update_spatial_odometry(top_bbox)

                    now = time.time()
                    if det_result["plate_detected"] and (now - last_ingest_time > 8.0):
                        s = {
                            "camera_id": "CAM_OAKD",
                            "timestamp": now,
                            "class": det_result.get("primary_vehicle") or "Car",
                            "color": "White",
                            "hypotheses": det_result["plates"][0]["hypotheses"],
                            "distance_m": spatial_info["distance_m"]
                        }
                        fusion_engine.ingest_sighting(s)
                        analytics_engine.record_detection("Car", "CAM_OAKD", speed_est=spatial_info["speed_kmh"])
                        last_ingest_time = now

                    cam_name = CITY_CAMERAS.get(camera_id, {}).get("name", camera_id)
                    annotated = anpr_engine.annotate_frame_with_hud(
                        raw_frame, det_result, camera_name=cam_name, fps=cam_mgr.oakd_fps, spatial_info=spatial_info
                    )
            else:
                raw_frame = cam_mgr.get_raw_frame(camera_id)
                det_result = anpr_engine.detect_vehicles_and_plates(raw_frame, synchronous=False)
                cam_name = CITY_CAMERAS.get(camera_id, {}).get("name", camera_id)
                annotated = anpr_engine.annotate_frame_with_hud(raw_frame, det_result, camera_name=cam_name, fps=25.0)

            # High-performance JPEG encode (Quality 78 for low bandwidth + high framerate)
            ret, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            logger.debug(f"Streaming error: {e}")
        # Frame pacing for ~60 FPS
        time.sleep(0.010)

# Live OAK-D Pro 3D Perception & Odometry Telemetry Endpoint
@app.get("/api/oakd/live_detection")
def get_oakd_live_detection():
    """Returns instant real-time detection, 3D metric distance, and visual odometry velocity."""
    frame = cam_mgr.get_raw_frame("CAM_OAKD")
    anpr_engine.submit_frame(frame)
    det_result = anpr_engine.get_latest_detection()
    top_bbox = det_result["vehicles"][0]["bbox"] if det_result["vehicle_detected"] else None
    spatial_info = cam_mgr.update_spatial_odometry(top_bbox)

    return {
        "vehicle_detected": det_result["vehicle_detected"],
        "plate_detected": det_result["plate_detected"],
        "primary_vehicle": det_result["primary_vehicle"] or "No Vehicle in View",
        "primary_plate": det_result["primary_plate"] or "No Plate in View",
        "count_vehicles": det_result["count_vehicles"],
        "count_plates": det_result["count_plates"],
        "plates": [
            {
                "plate_text": p.get("plate_text", ""),
                "confidence": p.get("confidence", 0.90),
                "bbox": p.get("bbox", [])
            }
            for p in det_result.get("plates", [])
        ],
        "spatial_odometry": spatial_info,
        "cameras": {
            "cam_a_rgb": True,
            "cam_b_left_ir": True,
            "cam_c_right_ir": True,
            "stereo_depth": True,
            "ir_laser_projector": True
        },
        "fps": cam_mgr.oakd_fps,
        "ai_fps": anpr_engine.inference_fps,
        "ai_latency_ms": anpr_engine.inference_latency_ms,
        "device_id": cam_mgr.oakd_info.get("device_id", "19443010E1A4FB4800")
    }

# API Endpoints
@app.get("/api/status")
def get_system_status():
    """Returns system status, OAK-D Pro hardware telemetry, and engine health."""
    telemetry = cam_mgr.get_telemetry()
    return {
        "status": "OPERATIONAL",
        "system": "TRINETRA-VIG v1.0",
        "problem_statement": "SIH 2026 - ID 26127",
        "team": "Tech_Innovation_Crew",
        "jurisdiction": "MH-14 (Lonavala - Maval - Pimpri-Chinchwad)",
        "oakd_pro": telemetry["oakd"],
        "active_cameras": len(telemetry["network"]),
        "total_sightings": len(fusion_engine.sightings),
        "active_journeys": len(fusion_engine.identity_graph),
        "active_alerts": len(fusion_engine.alerts)
    }

@app.get("/api/cameras")
def get_cameras():
    """Returns all camera nodes in the city network with locations and live status."""
    return list(CITY_CAMERAS.values())

@app.get("/api/sightings")
def get_sightings():
    """Returns chronological list of all camera sightings with Top-K candidates."""
    return fusion_engine.sightings[-40:]

@app.get("/api/journeys")
def get_journeys():
    """Returns all reconstructed vehicle journeys in the city."""
    return fusion_engine.get_all_journeys()

@app.get("/api/journey/{plate}")
def get_journey_detail(plate: str):
    """Returns full trajectory, timeline, and Explainable AI Reasoning Inspector data for a vehicle."""
    journey = fusion_engine.get_journey(plate)
    if not journey:
        return JSONResponse(status_code=404, content={"error": f"Vehicle with plate '{plate}' not found in identity graph."})
    return journey

@app.get("/api/alerts")
def get_alerts():
    """Returns all active alerts (Cloned plates, Watchlist hits, Speed violations)."""
    return fusion_engine.get_alerts()

class AlertActionRequest(BaseModel):
    action: str # "VERIFY", "DISPATCH_PATROL", "DISMISS"
    notes: Optional[str] = ""

@app.post("/api/alerts/{alert_id}/action")
def take_alert_action(alert_id: str, req: AlertActionRequest):
    """Applies human-in-the-loop action to an alert."""
    success = fusion_engine.resolve_alert(alert_id, req.action, req.notes or "")
    if success:
        return {"status": "SUCCESS", "message": f"Alert {alert_id} updated: {req.action}"}
    return JSONResponse(status_code=404, content={"error": "Alert ID not found."})

@app.get("/api/analytics")
def get_traffic_analytics():
    """Returns city-wide traffic intelligence analytics, density, speeds, and modal split."""
    return analytics_engine.get_summary()

@app.get("/api/audit/repairs")
def get_repair_audit_log():
    """Returns full law enforcement audit log showing preserved original readings vs resolved identities."""
    return fusion_engine.repair_audit_log

@app.get("/api/stream/oakd")
def stream_oakd_video(mode: str = "rgb_hud"):
    """Live MJPEG video stream from the Luxonis OAK-D Pro camera with multi-view modes."""
    return StreamingResponse(generate_mjpeg_stream("CAM_OAKD", mode=mode),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/stream/{cam_id}")
def stream_cam_video(cam_id: str):
    """Live MJPEG stream for any city camera node."""
    if cam_id not in CITY_CAMERAS:
        cam_id = "CAM_01"
    return StreamingResponse(generate_mjpeg_stream(cam_id),
                             media_type="multipart/x-mixed-replace; boundary=frame")

# PostGIS Spatial & Terrain Intelligence Endpoints
class LocationUpdateRequest(BaseModel):
    lat: float
    lng: float
    name: Optional[str] = "Live Command Unit / GPS"
    source: Optional[str] = "CLIENT_GPS"

class SpatialProximityRequest(BaseModel):
    lat: float
    lng: float
    radius_m: float = 3000.0

class PostGISSQLRequest(BaseModel):
    query: str

@app.get("/api/postgis/status")
def get_postgis_status():
    """Returns PostGIS spatial engine health, SRID, spatial reference, and active command post."""
    return {
        "status": "OPERATIONAL",
        "engine": "PostGIS Spatial Engine (GEOS 3.12 / Shapely 2.1)",
        "srid_primary": "EPSG:4326 (WGS84 Ellipsoidal)",
        "srid_projected": "EPSG:3857 (Spherical Mercator)",
        "current_command_post": postgis_engine.get_current_location_summary(),
        "registered_layers": [
            {"name": "anpr_cameras_points", "type": "Point", "count": len(postgis_engine.cameras)},
            {"name": "camera_fov_coverage_cones", "type": "Polygon", "count": len(postgis_engine.cameras)},
            {"name": "road_corridors_network", "type": "LineString", "count": len(postgis_engine.corridors)},
            {"name": "vehicle_trajectories", "type": "LineStringZM", "count": len(fusion_engine.identity_graph)}
        ],
        "dem_terrain": {
            "model": "Western Ghats Sahyadri DEM (10m Contours)",
            "elevation_min_m": 120.0,
            "elevation_max_m": 820.0,
            "datum": "Mean Sea Level (MSL)"
        }
    }

@app.get("/api/postgis/cameras/geojson")
def get_postgis_cameras():
    """Returns all camera nodes and their FOV coverage cone polygons as PostGIS GeoJSON."""
    return postgis_engine.get_cameras_geojson()

@app.get("/api/postgis/corridors/geojson")
def get_postgis_corridors():
    """Returns highway and ghat road corridors as PostGIS GeoJSON LineStrings."""
    return postgis_engine.get_corridors_geojson()

@app.get("/api/postgis/trajectories/geojson")
def get_postgis_trajectories():
    """Returns reconstructed vehicle trajectories as 4D PostGIS GeoJSON LineStrings (X, Y, Z_elevation, M_timestamp)."""
    sightings_by_plate = {p: j.get("sightings", []) for p, j in fusion_engine.identity_graph.items()}
    return postgis_engine.get_trajectories_geojson(sightings_by_plate)

@app.get("/api/postgis/location/current")
def get_current_location():
    """Returns the active Command Post location and spatial distance/bearing vectors to all cameras."""
    return postgis_engine.get_current_location_summary()

@app.post("/api/postgis/location/set")
def set_current_location(req: LocationUpdateRequest):
    """Sets the active Command Post location (from laptop GPS, user click, or OAK-D rig)."""
    res = postgis_engine.set_current_location(req.lat, req.lng, req.name or "Live Command Unit", req.source or "CLIENT_GPS")
    cam_mgr.cameras["CAM_OAKD"]["location"]["lat"] = req.lat
    cam_mgr.cameras["CAM_OAKD"]["location"]["lng"] = req.lng
    return res

@app.post("/api/postgis/spatial_query/within_radius")
def query_within_radius(req: SpatialProximityRequest):
    """Executes a PostGIS ST_DWithin proximity search returning cameras and circular buffer polygon."""
    return postgis_engine.execute_spatial_proximity_query(req.lat, req.lng, req.radius_m)

@app.get("/api/postgis/terrain/elevation_profile")
def get_terrain_elevation_profile(corridor_id: str = "CORR_NH48"):
    """Returns topographic elevation profile with ascent/descent gradient along a corridor."""
    return postgis_engine.get_terrain_elevation_profile(corridor_id)

@app.post("/api/postgis/sql_console")
def run_postgis_sql(req: PostGISSQLRequest):
    """Executes a simulated PostGIS spatial SQL query and returns result rows + GeoJSON."""
    q = req.query.strip()
    t0 = time.time()
    q_lower = q.lower()
    rows = []
    columns = []
    geojson = None
    
    if "st_dwithin" in q_lower or "distance" in q_lower:
        cur = postgis_engine.current_location
        res = postgis_engine.execute_spatial_proximity_query(cur["lat"], cur["lng"], 5000.0)
        columns = ["camera_id", "name", "distance_meters", "elevation_m", "is_hardware"]
        for m in res["matches"]:
            rows.append({
                "camera_id": m["camera_id"],
                "name": m["name"],
                "distance_meters": m["distance_meters"],
                "elevation_m": m["elevation_m"],
                "is_hardware": m["is_hardware"]
            })
        geojson = res["buffer_polygon_geojson"]
    elif "corridor" in q_lower or "st_length" in q_lower:
        columns = ["corridor_id", "name", "type", "speed_limit_kmh", "length_km"]
        for c in postgis_engine.get_corridors_geojson()["features"]:
            props = c["properties"]
            rows.append({
                "corridor_id": props["corridor_id"],
                "name": props["name"],
                "type": props["type"],
                "speed_limit_kmh": props["speed_limit_kmh"],
                "length_km": props["length_km"]
            })
    else:
        columns = ["camera_id", "name", "lat", "lng", "elevation_m", "postgis_wkt"]
        for c in postgis_engine.get_cameras_geojson()["features"]:
            if c["properties"].get("feature_type") == "ANPR_CAMERA_NODE":
                props = c["properties"]
                rows.append({
                    "camera_id": props["camera_id"],
                    "name": props["name"],
                    "lat": c["geometry"]["coordinates"][1],
                    "lng": c["geometry"]["coordinates"][0],
                    "elevation_m": props["elevation_m"],
                    "postgis_wkt": props["postgis_geom_wkt"]
                })
                
    t_exec = round((time.time() - t0) * 1000, 2)
    return {
        "status": "SUCCESS",
        "query": q,
        "execution_time_ms": t_exec,
        "row_count": len(rows),
        "columns": columns,
        "rows": rows,
        "geojson": geojson
    }

# Scenario Trigger Endpoint
class ScenarioRequest(BaseModel):
    scenario: str

@app.post("/api/scenarios/run")
def run_scenario(req: ScenarioRequest):
    """Triggers live interactive demonstration scenarios from the PPT."""
    now = time.time()
    sc = req.scenario

    if sc == "weak_read_repair":
        # Simulates an ambiguous sighting repaired by downstream camera
        plate_a = "KA03MN11B8"
        s1 = {
            "camera_id": "CAM_01",
            "timestamp": now - 360,
            "class": "Car (Sedan)",
            "color": "White",
            "hypotheses": [
                {"plate": "KA03MN11B8", "confidence": 0.46, "rank": 1},
                {"plate": "KA03MN1188", "confidence": 0.42, "rank": 2},
                {"plate": "KA03MN1108", "confidence": 0.12, "rank": 3}
            ]
        }
        fusion_engine.ingest_sighting(s1)
        time.sleep(0.5)

        s2 = {
            "camera_id": "CAM_02",
            "timestamp": now,
            "class": "Car (Sedan)",
            "color": "White",
            "hypotheses": [
                {"plate": "KA03MN1188", "confidence": 0.94, "rank": 1},
                {"plate": "KA03MN11B8", "confidence": 0.04, "rank": 2}
            ]
        }
        res2 = fusion_engine.ingest_sighting(s2)
        analytics_engine.record_detection("Car", "CAM_02", speed_est=34.0)

        return {
            "status": "COMPLETED",
            "scenario": "Weak-Read Retroactive Repair",
            "repaired": res2.get("retroactively_repaired", True),
            "resolved_plate": "KA03MN1188",
            "original_cam_a_raw": "KA03MN11B8",
            "explanation": (
                "Camera A recorded ambiguous reading 'KA03MN11B8' (46%). Downstream Camera B recorded "
                "'KA03MN1188' (94%). TRINETRA-VIG fused visual appearance (White Sedan) and road-time "
                "feasibility, retroactively repairing Camera A's journey while preserving the raw sensor read."
            )
        }

    elif sc == "cloned_plate":
        s1 = {
            "camera_id": "CAM_01",
            "timestamp": now - 50,
            "class": "Car (Sedan)",
            "color": "White",
            "hypotheses": [{"plate": "DL01AB1234", "confidence": 0.93, "rank": 1}]
        }
        fusion_engine.ingest_sighting(s1)

        s2 = {
            "camera_id": "CAM_03",
            "timestamp": now, # Only 50s later over 9.5 km highway distance!
            "class": "SUV",
            "color": "Dark SUV / Grey",
            "hypotheses": [{"plate": "DL01AB1234", "confidence": 0.92, "rank": 1}]
        }
        fusion_engine.ingest_sighting(s2)

        return {
            "status": "COMPLETED",
            "scenario": "Cloned Plate Detection",
            "alert": "ALERT: PROBABLE CLONED PLATE (DL01AB1234)",
            "explanation": "Identical plate observed at Cam 1 and Cam 3 within 50 seconds (implied speed > 680 km/h) with visual clash (White Sedan vs Dark SUV)."
        }

    elif sc == "watchlist_hit":
        s1 = {
            "camera_id": "CAM_04",
            "timestamp": now,
            "class": "Car (Sedan)",
            "color": "Silver",
            "hypotheses": [{"plate": "MH14DX9921", "confidence": 0.95, "rank": 1}]
        }
        fusion_engine.ingest_sighting(s1)
        return {
            "status": "COMPLETED",
            "scenario": "Police Watchlist Interception",
            "alert": "POLICE WATCHLIST HIT: MH14DX9921 (Stolen Vehicle FIR #412/2026)"
        }

    elif sc == "oakd_live_capture":
        # Capture current frame from live OAK-D Pro camera and inject into system
        frame = cam_mgr.get_raw_frame("CAM_OAKD")
        detections = anpr_engine.detect_vehicles(frame)
        plate_str = "MH14EX9900"
        if detections:
            v = detections[0]
            v_class = v.get("class", "Car")
            v_color = v.get("color", "White")
        else:
            v_class = "Car (Sedan)"
            v_color = "Silver"

        s1 = {
            "camera_id": "CAM_OAKD",
            "timestamp": now,
            "class": v_class,
            "color": v_color,
            "hypotheses": [
                {"plate": plate_str, "confidence": 0.91, "rank": 1},
                {"plate": "MH14EX990D", "confidence": 0.06, "rank": 2}
            ]
        }
        res = fusion_engine.ingest_sighting(s1)
        analytics_engine.record_detection(v_class, "CAM_OAKD", speed_est=30.0)

        return {
            "status": "COMPLETED",
            "scenario": "Live OAK-D Pro Hardware Capture",
            "resolved_plate": res["resolved_plate"],
            "resolution_status": res["resolution_status"],
            "composite_score": res["composite_score"],
            "camera": "CAM_OAKD (Luxonis OAK-D Pro)"
        }

    return {"status": "UNKNOWN_SCENARIO"}

# RTO / VAHAN Database Lookup Endpoint (Slide 6 & Slide 3)
@app.get("/api/vahan/{plate}")
def get_vahan_details(plate: str):
    """Returns official RTO / VAHAN registration details for MH-14 or interstate vehicles."""
    return VahanService.lookup_vehicle(plate)

# City-Wide Vehicle & Trajectory Search Endpoint (Slide 3 & 5)
@app.get("/api/search")
def search_vehicles(q: str):
    """Searches vehicle journeys by plate number, color, or vehicle class."""
    query = q.strip().upper()
    journeys = fusion_engine.get_all_journeys()
    results = []
    for j in journeys:
        if (query in j["plate"] or 
            query in j.get("vehicle_class", "").upper() or 
            query in j.get("color", "").upper()):
            results.append(j)
    return {"query": q, "count": len(results), "results": results}

# Mobile / Patrol Unit Ingestion Endpoint (Slide 3)
class MobileUploadRequest(BaseModel):
    plate: Optional[str] = "MH14AX1001"
    vehicle_class: Optional[str] = "Car (Sedan)"
    color: Optional[str] = "White"
    location_desc: Optional[str] = "Lonavala Patrol Unit #4"

@app.post("/api/mobile/upload")
def mobile_patrol_upload(req: MobileUploadRequest):
    """Handles field observation uploads from mobile or patrol units."""
    now = time.time()
    s = {
        "camera_id": "CAM_OAKD",
        "timestamp": now,
        "class": req.vehicle_class,
        "color": req.color,
        "hypotheses": anpr_engine.generate_top_k_hypotheses(req.plate or "MH14AX1001", base_confidence=0.92)
    }
    res = fusion_engine.ingest_sighting(s)
    analytics_engine.record_detection(req.vehicle_class or "Car", "CAM_OAKD", speed_est=35.0)
    return {
        "status": "SUCCESS",
        "sighting_id": res["sighting_id"],
        "resolved_plate": res["resolved_plate"],
        "resolution_status": res["resolution_status"],
        "composite_score": res["composite_score"]
    }

# Background Heartbeat: Generates realistic city traffic flow
def start_traffic_heartbeat():
    import threading, random
    sample_pool = [
        ("MH14BZ3321", "Car (Sedan)", "White", ["CAM_01", "CAM_02"]),
        ("MH14KT5050", "SUV", "Dark SUV / Grey", ["CAM_02", "CAM_03"]),
        ("MH12PQ9000", "Truck", "Silver", ["CAM_01", "CAM_04"]),
        ("MH14CC7788", "Motorcycle", "Red", ["CAM_02", "CAM_OAKD"])
    ]
    def loop():
        time.sleep(5)
        while True:
            try:
                time.sleep(12)
                plate, v_class, color, path = random.choice(sample_pool)
                cam = random.choice(path)
                s = {
                    "camera_id": cam,
                    "timestamp": time.time(),
                    "class": v_class,
                    "color": color,
                    "hypotheses": anpr_engine.generate_top_k_hypotheses(plate, base_confidence=0.88)
                }
                fusion_engine.ingest_sighting(s)
                analytics_engine.record_detection(v_class, cam, speed_est=float(random.randint(35, 75)))
            except Exception as e:
                logger.debug(f"Heartbeat error: {e}")
    t = threading.Thread(target=loop, daemon=True)
    t.start()

start_traffic_heartbeat()

# Serve output assets (zoomed plates and OCR verification cards)
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
os.makedirs(os.path.join(OUTPUT_DIR, "plates"), exist_ok=True)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

@app.get("/api/output/latest_detection")
def get_latest_output_detection():
    """Returns the most recent plate detection, OCR result, and saved image links from /output."""
    det_file = os.path.join(OUTPUT_DIR, "latest_detection.json")
    if os.path.exists(det_file):
        try:
            with open(det_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "plate_text": "SEARCHING",
        "confidence": 0.0,
        "source": "CAM_OAKD",
        "timestamp": time.time(),
        "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        "zoomed_image": "/output/latest_plate.png",
        "ocr_card": "/output/latest_ocr.png",
        "status": "SCANNING_ACTIVE"
    }

@app.get("/api/output/latest_plate")
def get_latest_output_plate():
    """Serves the latest zoomed number plate crop image."""
    p = os.path.join(OUTPUT_DIR, "latest_plate.png")
    if os.path.exists(p):
        return FileResponse(p, media_type="image/png")
    return JSONResponse(status_code=404, content={"error": "No plate image saved yet."})

@app.get("/api/output/latest_ocr")
def get_latest_output_ocr():
    """Serves the latest verified OCR recognition card image."""
    p = os.path.join(OUTPUT_DIR, "latest_ocr.png")
    if os.path.exists(p):
        return FileResponse(p, media_type="image/png")
    return JSONResponse(status_code=404, content={"error": "No OCR card saved yet."})

@app.get("/api/output/history")
def get_output_history():
    """Returns chronological list of all saved plate crops and OCR cards in /output/plates."""
    h_file = os.path.join(OUTPUT_DIR, "history.json")
    if os.path.exists(h_file):
        try:
            with open(h_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    plates_dir = os.path.join(OUTPUT_DIR, "plates")
    items = []
    if os.path.exists(plates_dir):
        files = sorted(os.listdir(plates_dir), reverse=True)[:30]
        for f in files:
            if f.endswith(".png"):
                items.append({
                    "filename": f,
                    "url": f"/output/plates/{f}"
                })
    return items

# Serve frontend static assets
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print(">> TRINETRA-VIG URBAN VEHICLE INTELLIGENCE PLATFORM")
    print("   Smart India Hackathon 2026 | PS 26127 | Theme: Smart Automation")
    print("   Connected Hardware: Luxonis OAK-D Pro")
    print("   Dashboard: http://localhost:8000")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
