import time
import random
from typing import Dict, Any, List

class AnalyticsEngine:
    def __init__(self):
        self.start_time = time.time()
        self.vehicle_counts = {
            "Car": 142,
            "SUV": 68,
            "Truck": 29,
            "Motorcycle": 44,
            "Bus": 17
        }
        self.hourly_history = [
            {"hour": "14:00", "count": 210, "avg_speed": 62.4},
            {"hour": "15:00", "count": 265, "avg_speed": 58.1},
            {"hour": "16:00", "count": 310, "avg_speed": 51.5},
            {"hour": "17:00", "count": 420, "avg_speed": 42.0},
            {"hour": "18:00", "count": 480, "avg_speed": 36.8},
            {"hour": "19:00", "count": 450, "avg_speed": 39.2},
            {"hour": "20:00", "count": 360, "avg_speed": 48.7}
        ]
        self.camera_stats = {
            "CAM_01": {"name": "NH-48 Entry Toll", "vehicles_passed": 184, "avg_speed": 74.2, "congestion_level": "LOW"},
            "CAM_02": {"name": "Kumar Resort Chowk", "vehicles_passed": 242, "avg_speed": 34.5, "congestion_level": "MEDIUM"},
            "CAM_03": {"name": "Khandala Interchange", "vehicles_passed": 169, "avg_speed": 82.1, "congestion_level": "LOW"},
            "CAM_04": {"name": "Maval Junction", "vehicles_passed": 135, "avg_speed": 46.8, "congestion_level": "LOW"},
            "CAM_OAKD": {"name": "OAK-D Pro Live Checkpoint", "vehicles_passed": 38, "avg_speed": 32.0, "congestion_level": "LOW"}
        }

    def record_detection(self, v_class: str, camera_id: str, speed_est: float = 55.0):
        """Records a vehicle pass for live analytics."""
        norm_class = "Car"
        if "SUV" in v_class or "Dark" in v_class:
            norm_class = "SUV"
        elif "Truck" in v_class:
            norm_class = "Truck"
        elif "Motorcycle" in v_class or "Bike" in v_class:
            norm_class = "Motorcycle"
        elif "Bus" in v_class:
            norm_class = "Bus"

        self.vehicle_counts[norm_class] = self.vehicle_counts.get(norm_class, 0) + 1

        if camera_id in self.camera_stats:
            stat = self.camera_stats[camera_id]
            stat["vehicles_passed"] += 1
            # Running average speed
            stat["avg_speed"] = round((stat["avg_speed"] * 0.95) + (speed_est * 0.05), 1)
            if stat["vehicles_passed"] > 220 or stat["avg_speed"] < 25.0:
                stat["congestion_level"] = "HIGH"
            elif stat["vehicles_passed"] > 120 or stat["avg_speed"] < 40.0:
                stat["congestion_level"] = "MEDIUM"
            else:
                stat["congestion_level"] = "LOW"

    def get_summary(self) -> Dict[str, Any]:
        """Returns comprehensive traffic intelligence dashboard metrics."""
        total_vehicles = sum(self.vehicle_counts.values())
        avg_speed = round(sum(s["avg_speed"] for s in self.camera_stats.values()) / len(self.camera_stats), 1)

        # Modal breakdown percentages
        breakdown = {}
        for k, v in self.vehicle_counts.items():
            breakdown[k] = round((v / max(1, total_vehicles)) * 100, 1)

        return {
            "total_vehicles_detected": total_vehicles,
            "corridor_avg_speed_kmh": avg_speed,
            "overall_density": "MODERATE (Level B)",
            "jurisdiction": "MH-14 (Lonavala - Maval - Pimpri)",
            "active_hotspots": ["Kumar Resort Chowk (Peak Inflow)"],
            "modal_split": breakdown,
            "camera_breakdown": self.camera_stats,
            "hourly_trends": self.hourly_history
        }
