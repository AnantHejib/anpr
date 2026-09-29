import time
import math
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("TRINETRA.Fusion")

# Known road distances (in km) and typical travel times between Lonavala/Maval camera nodes
ROAD_GRAPH_DISTANCES = {
    ("CAM_01", "CAM_02"): 6.2,   # NH-48 Toll to Kumar Resort Chowk
    ("CAM_02", "CAM_01"): 6.2,
    ("CAM_02", "CAM_03"): 4.8,   # Kumar Resort to Khandala Interchange
    ("CAM_03", "CAM_02"): 4.8,
    ("CAM_01", "CAM_03"): 9.5,   # Toll to Khandala Interchange
    ("CAM_03", "CAM_01"): 9.5,
    ("CAM_02", "CAM_04"): 8.4,   # Kumar Resort to Maval Junction
    ("CAM_04", "CAM_02"): 8.4,
    ("CAM_01", "CAM_04"): 14.1,  # Toll to Maval Junction
    ("CAM_04", "CAM_01"): 14.1,
    ("CAM_03", "CAM_04"): 13.0,  # Khandala to Maval Junction
    ("CAM_04", "CAM_03"): 13.0,
    ("CAM_OAKD", "CAM_02"): 1.8, # Local OAK-D Checkpoint to Kumar Resort
    ("CAM_02", "CAM_OAKD"): 1.8,
    ("CAM_OAKD", "CAM_01"): 4.9,
    ("CAM_01", "CAM_OAKD"): 4.9
}

# Watchlist / Blacklist of vehicles of interest for MH-14 jurisdiction
POLICE_WATCHLIST = {
    "MH14DX9921": {
        "plate": "MH14DX9921",
        "reason": "WANTED: Stolen Vehicle FIR #412/2026 (Lonavala Police)",
        "priority": "HIGH",
        "flagged_by": "Maharashtra Police Dept",
        "action_required": "Intercept at nearest toll/checkpoint and notify control room."
    },
    "DL01AB1234": {
        "plate": "DL01AB1234",
        "reason": "SUSPICIOUS: Suspected Plate Duplication / Cross-State Toll Evasion",
        "priority": "CRITICAL",
        "flagged_by": "State Highway Patrol",
        "action_required": "Flag for cloned plate multi-camera verification."
    },
    "MH12AB9999": {
        "plate": "MH12AB9999",
        "reason": "UNPAID CHALLANS: 14 High-Speed Corridor Violations",
        "priority": "MEDIUM",
        "flagged_by": "RTO Pune / Pimpri-Chinchwad",
        "action_required": "Issue e-challan notice and detain vehicle at toll plaza."
    }
}

class EvidenceFusionEngine:
    def __init__(self):
        # All ingested raw sightings chronologically
        self.sightings: List[Dict[str, Any]] = []
        # Vehicle identity graph: resolved_plate -> list of sightings belonging to this vehicle journey
        self.identity_graph: Dict[str, Dict[str, Any]] = {}
        # Active alerts list
        self.alerts: List[Dict[str, Any]] = []
        # Audit log of retroactive repairs
        self.repair_audit_log: List[Dict[str, Any]] = []

    def get_road_distance(self, cam_a: str, cam_b: str) -> float:
        """Returns road distance between two cameras in kilometers."""
        if cam_a == cam_b:
            return 0.1
        return ROAD_GRAPH_DISTANCES.get((cam_a, cam_b), 7.5)

    def calculate_visual_similarity(self, v1: Dict[str, Any], v2: Dict[str, Any]) -> float:
        """Calculates visual appearance similarity between two vehicle sightings (0.0 to 1.0)."""
        score = 0.0
        # Color match
        c1 = v1.get("color", "").lower()
        c2 = v2.get("color", "").lower()
        if c1 and c2:
            if c1 == c2:
                score += 0.50
            elif (c1 in ["silver", "grey", "white"] and c2 in ["silver", "grey", "white"]):
                score += 0.35
            else:
                score += 0.05
        else:
            score += 0.30

        # Class match (Sedan vs SUV vs Truck)
        k1 = v1.get("class", "").lower()
        k2 = v2.get("class", "").lower()
        if k1 and k2:
            if k1 == k2:
                score += 0.30
            elif "car" in k1 and "car" in k2:
                score += 0.25
            else:
                score += 0.0
        else:
            score += 0.15

        # Feature embedding cosine similarity
        e1 = v1.get("embedding", [])
        e2 = v2.get("embedding", [])
        if e1 and e2 and len(e1) == len(e2) and len(e1) > 0:
            dot = sum(a * b for a, b in zip(e1, e2))
            norm1 = math.sqrt(sum(a * a for a in e1))
            norm2 = math.sqrt(sum(b * b for b in e2))
            if norm1 > 0 and norm2 > 0:
                cos_sim = max(0.0, min(1.0, dot / (norm1 * norm2)))
                score += 0.20 * cos_sim
        else:
            score += 0.10

        return round(min(1.0, score), 2)

    def calculate_road_time_feasibility(self, s1: Dict[str, Any], s2: Dict[str, Any]) -> Tuple[float, float, float, bool]:
        """
        Calculates travel feasibility between two sightings.
        Returns: (feasibility_score, distance_km, delta_minutes, is_physically_impossible)
        """
        cam_a = s1["camera_id"]
        cam_b = s2["camera_id"]
        dt_seconds = abs(s2["timestamp"] - s1["timestamp"])
        dt_minutes = max(0.05, dt_seconds / 60.0)

        dist_km = self.get_road_distance(cam_a, cam_b)
        speed_kmh = (dist_km / (dt_minutes / 60.0))

        # Check impossible physics
        # If speed > 180 km/h on urban corridor, or extreme impossible jump (e.g. 27.2 km in 0.8 min)
        if speed_kmh > 180.0 and cam_a != cam_b:
            feasibility_score = 0.0
            is_impossible = True
        elif speed_kmh > 120.0:
            feasibility_score = 0.30
            is_impossible = False
        elif speed_kmh < 5.0 and dist_km > 3.0 and dt_minutes < 180.0:
            feasibility_score = 0.70 # Normal traffic delay
            is_impossible = False
        else:
            feasibility_score = 0.95 # Highly feasible
            is_impossible = False

        return round(feasibility_score, 2), round(dist_km, 1), round(dt_minutes, 1), is_impossible

    def ingest_sighting(self, sighting: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core ingestion pipeline:
        1. Analyzes Top-K candidate hypotheses from ANPR.
        2. Evaluates evidence against existing vehicle identity graph.
        3. Executes Retroactive Weak-Read Repair if stronger later evidence clarifies earlier sightings.
        4. Detects Cloned Plates and Watchlist violations.
        5. Updates and stores identity journey.
        """
        sighting_id = f"SGT_{len(self.sightings) + 1:05d}"
        sighting["sighting_id"] = sighting_id
        if "timestamp" not in sighting or sighting["timestamp"] is None:
            sighting["timestamp"] = time.time()

        # Top-K candidate hypotheses
        hypotheses = sighting.get("hypotheses", [])
        if not hypotheses:
            top_plate = sighting.get("raw_plate", "MH14AB1234")
            hypotheses = [{"plate": top_plate, "confidence": 0.85, "rank": 1}]
            sighting["hypotheses"] = hypotheses

        primary_hypothesis = hypotheses[0]
        raw_plate = primary_hypothesis["plate"]
        plate_conf = primary_hypothesis["confidence"]
        sighting["raw_reading"] = raw_plate

        # Check for Watchlist alert
        self._check_watchlist_hit(sighting, raw_plate)

        # Multi-camera evidence matching against known vehicle journeys
        best_match_plate = raw_plate
        best_composite_score = plate_conf
        repaired = False
        reasoning_trail = []

        # Check all candidate hypotheses against existing journeys
        candidate_plates = [h["plate"] for h in hypotheses]
        for candidate in hypotheses:
            c_plate = candidate["plate"]
            c_conf = candidate["confidence"]

            if c_plate in self.identity_graph:
                journey = self.identity_graph[c_plate]
                last_sighting = journey["sightings"][-1]

                # 1. Visual Match
                s_vis = self.calculate_visual_similarity(sighting, last_sighting)
                # 2. Road + Time Feasibility
                s_road, dist_km, dt_min, is_impossible = self.calculate_road_time_feasibility(last_sighting, sighting)

                # Check for Cloned Plate anomaly
                if is_impossible or (c_plate == raw_plate and s_vis < 0.25 and dist_km > 2.0):
                    self._create_cloned_plate_alert(last_sighting, sighting, dist_km, dt_min, s_vis)

                # Composite score: 45% plate confidence + 30% visual appearance + 25% road/time feasibility
                composite = round(0.45 * c_conf + 0.30 * s_vis + 0.25 * s_road, 2)

                reasoning_trail.append({
                    "candidate_plate": c_plate,
                    "plate_confidence": c_conf,
                    "visual_similarity": s_vis,
                    "road_feasibility": s_road,
                    "distance_km": dist_km,
                    "time_delta_min": dt_min,
                    "composite_score": composite
                })

                if composite > best_composite_score:
                    best_composite_score = composite
                    best_match_plate = c_plate

        # Check for RETROACTIVE WEAK-READ REPAIR (The Key Differentiator from Slide 2)
        # If this current sighting has strong confidence (e.g. >= 0.88), see if earlier ambiguous sightings
        # in other journeys should be merged and repaired!
        if plate_conf >= 0.85:
            repaired = self._try_retroactive_repair(sighting, raw_plate)

        # Determine Resolution Status
        if best_composite_score >= 0.80 or repaired:
            resolution_status = "CONFIRMED"
        elif best_composite_score >= 0.50:
            resolution_status = "PROBABLE"
        else:
            resolution_status = "UNKNOWN"

        sighting["resolved_plate"] = best_match_plate
        sighting["composite_score"] = round(best_composite_score, 2)
        sighting["resolution_status"] = resolution_status
        sighting["reasoning_trail"] = reasoning_trail
        sighting["retroactively_repaired"] = repaired

        # Append to master sightings
        self.sightings.append(sighting)

        # Update or create identity journey in graph
        target_plate = best_match_plate
        if target_plate not in self.identity_graph:
            self.identity_graph[target_plate] = {
                "plate": target_plate,
                "first_seen": sighting["timestamp"],
                "last_seen": sighting["timestamp"],
                "sightings": [sighting],
                "vehicle_class": sighting.get("class", "Car"),
                "color": sighting.get("color", "White"),
                "status": resolution_status,
                "confidence": best_composite_score,
                "trajectory": [sighting["camera_id"]]
            }
        else:
            j = self.identity_graph[target_plate]
            j["last_seen"] = sighting["timestamp"]
            j["sightings"].append(sighting)
            j["status"] = resolution_status
            j["confidence"] = max(j["confidence"], best_composite_score)
            if sighting["camera_id"] not in j["trajectory"]:
                j["trajectory"].append(sighting["camera_id"])

        return sighting

    def _try_retroactive_repair(self, current_sighting: Dict[str, Any], strong_plate: str) -> bool:
        """
        Implements Retroactive Weak-Read Repair:
        Earlier ambiguous sighting (e.g. KA03MN11B8 @ 46%) is re-evaluated using new strong evidence
        from Camera B (KA03MN1188 @ 94%). The earlier link is repaired to KA03MN1188 while preserving
        original raw camera reading in audit log.
        """
        repaired_any = False
        for s in self.sightings:
            # Look for an earlier ambiguous sighting within the last 45 minutes
            if s.get("resolved_plate") != strong_plate and s.get("camera_id") != current_sighting.get("camera_id"):
                dt = abs(current_sighting["timestamp"] - s["timestamp"])
                if dt < 2700: # Within 45 min
                    # Did this earlier sighting have strong_plate as a candidate hypothesis?
                    for h in s.get("hypotheses", []):
                        if h["plate"] == strong_plate:
                            # Verify visual & road feasibility
                            s_vis = self.calculate_visual_similarity(s, current_sighting)
                            s_road, dist_km, dt_min, is_impossible = self.calculate_road_time_feasibility(s, current_sighting)

                            if not is_impossible and s_vis >= 0.50:
                                old_resolved = s["resolved_plate"]
                                s["resolved_plate"] = strong_plate
                                s["resolution_status"] = "CONFIRMED"
                                s["retroactively_repaired"] = True
                                s["repair_reason"] = (
                                    f"Repaired from '{old_resolved}' to '{strong_plate}' using downstream evidence "
                                    f"at {current_sighting['camera_id']} ({int(current_sighting['hypotheses'][0]['confidence']*100)}%). "
                                    f"Appearance match: {int(s_vis*100)}%, Distance: {dist_km} km in {dt_min} min. "
                                    f"Original camera reading '{s['raw_reading']}' preserved for auditability."
                                )

                                # Log to repair audit trail
                                self.repair_audit_log.append({
                                    "sighting_id": s["sighting_id"],
                                    "camera_id": s["camera_id"],
                                    "original_raw_reading": s["raw_reading"],
                                    "original_resolved": old_resolved,
                                    "repaired_plate": strong_plate,
                                    "repaired_by_camera": current_sighting["camera_id"],
                                    "explanation": s["repair_reason"],
                                    "timestamp": time.time()
                                })
                                logger.info(f"RETROACTIVE REPAIR: {s['repair_reason']}")
                                repaired_any = True
                                break
        return repaired_any

    def _create_cloned_plate_alert(self, s1: Dict[str, Any], s2: Dict[str, Any], dist_km: float, dt_min: float, s_vis: float):
        """Creates high-priority alert for impossible travel time or visual mismatch on identical plate."""
        min_feasible_min = round(dist_km / 1.4, 1) # assuming 85 km/h max highway speed
        alert_id = f"ALT_CLONE_{int(time.time()*1000) % 100000}"
        
        # Check if already alerted recently
        for a in self.alerts:
            if a.get("type") == "CLONED_PLATE" and a.get("plate") == s1.get("resolved_plate"):
                return

        alert = {
            "alert_id": alert_id,
            "type": "CLONED_PLATE",
            "priority": "CRITICAL",
            "title": f"ALERT: PROBABLE CLONED PLATE ({s1.get('resolved_plate')})",
            "plate": s1.get("resolved_plate"),
            "timestamp": time.time(),
            "cam_a": {
                "id": s1["camera_id"],
                "vehicle": f"{s1.get('color')} {s1.get('class')}",
                "time": time.strftime("%H:%M:%S", time.localtime(s1["timestamp"]))
            },
            "cam_b": {
                "id": s2["camera_id"],
                "vehicle": f"{s2.get('color')} {s2.get('class')}",
                "time": time.strftime("%H:%M:%S", time.localtime(s2["timestamp"]))
            },
            "physics": {
                "distance_km": dist_km,
                "elapsed_time_min": dt_min,
                "minimum_feasible_time_min": min_feasible_min,
                "implied_speed_kmh": round((dist_km / max(0.01, dt_min / 60.0)), 1)
            },
            "visual_similarity": s_vis,
            "explanation": (
                f"Vehicle with plate '{s1.get('resolved_plate')}' observed at {s1['camera_id']} "
                f"and {s2['camera_id']} ({dist_km} km apart) within {dt_min} minutes. "
                f"Minimum feasible travel time is {min_feasible_min} minutes. "
                f"Visual check: Cam A ({s1.get('color')} {s1.get('class')}) vs Cam B ({s2.get('color')} {s2.get('class')})."
            ),
            "status": "UNRESOLVED_REQUIRES_HUMAN_ACTION"
        }
        self.alerts.insert(0, alert)
        logger.warning(f"CLONED PLATE DETECTED: {alert['explanation']}")

    def _check_watchlist_hit(self, sighting: Dict[str, Any], plate: str):
        """Checks if a plate matches the police wanted vehicle registry."""
        if plate in POLICE_WATCHLIST:
            w_info = POLICE_WATCHLIST[plate]
            alert_id = f"ALT_WATCH_{int(time.time()*1000) % 100000}"
            # Avoid duplicate alert within 2 minutes
            for a in self.alerts:
                if a.get("type") == "WATCHLIST_HIT" and a.get("plate") == plate and time.time() - a.get("timestamp", 0) < 120:
                    return

            alert = {
                "alert_id": alert_id,
                "type": "WATCHLIST_HIT",
                "priority": w_info["priority"],
                "title": f"POLICE WATCHLIST HIT: {plate}",
                "plate": plate,
                "timestamp": time.time(),
                "camera_id": sighting["camera_id"],
                "vehicle_details": f"{sighting.get('color', '')} {sighting.get('class', '')}",
                "reason": w_info["reason"],
                "flagged_by": w_info["flagged_by"],
                "action_required": w_info["action_required"],
                "status": "ACTION_PENDING"
            }
            self.alerts.insert(0, alert)
            logger.warning(f"WATCHLIST INTERCEPTION: {plate} at {sighting['camera_id']}")

    def get_journey(self, plate: str) -> Optional[Dict[str, Any]]:
        """Returns the full journey reconstruct with timeline, thumbnails, and reasoning inspector data."""
        clean = plate.replace(" ", "").upper()
        if clean in self.identity_graph:
            return self.identity_graph[clean]
        # Search partial or matching resolved plates
        for p, data in self.identity_graph.items():
            if clean in p:
                return data
        return None

    def get_all_journeys(self) -> List[Dict[str, Any]]:
        """Returns all reconstructed vehicle journeys sorted by latest sighting."""
        res = list(self.identity_graph.values())
        res.sort(key=lambda x: x["last_seen"], reverse=True)
        return res

    def get_alerts(self) -> List[Dict[str, Any]]:
        """Returns active alerts list."""
        return self.alerts

    def resolve_alert(self, alert_id: str, action: str, officer_notes: str = "") -> bool:
        """Applies human-in-the-loop action to an alert."""
        for a in self.alerts:
            if a["alert_id"] == alert_id:
                a["status"] = f"RESOLVED: {action.upper()}"
                a["resolution_action"] = action
                a["officer_notes"] = officer_notes
                a["resolved_at"] = time.time()
                return True
        return False
