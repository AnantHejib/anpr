import math
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from shapely.geometry import Point, LineString, Polygon, mapping
from shapely import ops

logger = logging.getLogger("TRINETRA.PostGIS")

# Standard WGS84 Spatial Reference
SRID_WGS84 = 4326

# Western Ghats Topographic Elevation Baseline (Digital Elevation Model interpolation)
# Known elevation anchor points (lat, lng, elevation_m, name)
ELEVATION_ANCHORS = [
    (18.7548, 73.4068, 625.0, "NH-48 Toll Plaza"),
    (18.7512, 73.4145, 630.0, "Kumar Resort Chowk"),
    (18.7595, 73.3762, 550.0, "Khandala Ghat Interchange"),
    (18.7320, 73.4420, 590.0, "Maval Industrial Junction"),
    (18.7530, 73.4100, 622.0, "OAK-D Mobile Command Post"),
    (18.7650, 73.3500, 240.0, "Khopoli Ghat Descent Base"),
    (18.7300, 73.3850, 780.0, "Tiger Leap / Duke's Nose Peak"),
    (18.7400, 73.4300, 645.0, "Bushi Dam Ridge")
]

class PostGISSpatialEngine:
    def __init__(self):
        # Current Command Post / Mobile Unit Location (Default: OAK-D Pro hardware position)
        self.current_location = {
            "lat": 18.7530,
            "lng": 73.4100,
            "elevation_m": 622.0,
            "name": "Luxonis OAK-D Pro Mobile Command Post",
            "source": "HARDWARE_GPS",
            "updated_at": time.time()
        }

        # Road Corridors in PostGIS format (LineStrings)
        self.corridors = self._init_corridors()
        
        # Camera Registry with PostGIS Points & Coverage Polygons
        self.cameras = self._init_cameras()

        # Spatial Query History
        self.query_history = []

        logger.info("PostGIS Spatial Intelligence Engine initialized (EPSG:4326 WGS84).")

    def _init_corridors(self) -> Dict[str, Dict[str, Any]]:
        """Initializes primary highway & arterial road corridors as PostGIS geometries."""
        return {
            "CORR_NH48": {
                "id": "CORR_NH48",
                "name": "NH-48 Pune-Mumbai Highway Corridor",
                "type": "National Highway",
                "speed_limit_kmh": 80,
                "coordinates": [
                    [73.3762, 18.7595], # Khandala
                    [73.3900, 18.7570],
                    [73.4068, 18.7548], # Toll Plaza
                    [73.4100, 18.7530], # Command Post
                    [73.4145, 18.7512], # Kumar Resort
                    [73.4280, 18.7420],
                    [73.4420, 18.7320]  # Maval Junction
                ]
            },
            "CORR_EXPRESSWAY": {
                "id": "CORR_EXPRESSWAY",
                "name": "Mumbai-Pune Yashwantrao Chavan Expressway (Khandala Ghat Sector)",
                "type": "Access-Controlled Expressway",
                "speed_limit_kmh": 100,
                "coordinates": [
                    [73.3500, 18.7650], # Bhor Ghat Descent
                    [73.3762, 18.7595], # Khandala Interchange
                    [73.4000, 18.7480],
                    [73.4250, 18.7350],
                    [73.4500, 18.7200]
                ]
            },
            "CORR_GHAT_PASS": {
                "id": "CORR_GHAT_PASS",
                "name": "Bhor Ghat Mountain Pass & Tunnel",
                "type": "Mountain Ghat Road",
                "speed_limit_kmh": 40,
                "coordinates": [
                    [73.3500, 18.7650],
                    [73.3620, 18.7630],
                    [73.3700, 18.7610],
                    [73.3762, 18.7595]
                ]
            }
        }

    def _init_cameras(self) -> Dict[str, Dict[str, Any]]:
        """Initializes camera nodes with spatial attributes, elevation, and FOV coverage azimuth."""
        cams = {
            "CAM_OAKD": {
                "id": "CAM_OAKD",
                "name": "Luxonis OAK-D Pro (3-Cam Perception & Odometry Engine)",
                "lat": 18.7530,
                "lng": 73.4100,
                "elevation_m": 622.0,
                "heading_deg": 135,
                "fov_deg": 75,
                "range_m": 120,
                "is_hardware": True
            },
            "CAM_01": {
                "id": "CAM_01",
                "name": "NH-48 Entry Toll Plaza",
                "lat": 18.7548,
                "lng": 73.4068,
                "elevation_m": 625.0,
                "heading_deg": 90,
                "fov_deg": 65,
                "range_m": 250,
                "is_hardware": False
            },
            "CAM_02": {
                "id": "CAM_02",
                "name": "Kumar Resort Chowk",
                "lat": 18.7512,
                "lng": 73.4145,
                "elevation_m": 630.0,
                "heading_deg": 220,
                "fov_deg": 90,
                "range_m": 180,
                "is_hardware": False
            },
            "CAM_03": {
                "id": "CAM_03",
                "name": "Expressway Interchange (Khandala)",
                "lat": 18.7595,
                "lng": 73.3762,
                "elevation_m": 550.0,
                "heading_deg": 310,
                "fov_deg": 70,
                "range_m": 300,
                "is_hardware": False
            },
            "CAM_04": {
                "id": "CAM_04",
                "name": "Maval Industrial Junction",
                "lat": 18.7320,
                "lng": 73.4420,
                "elevation_m": 590.0,
                "heading_deg": 45,
                "fov_deg": 80,
                "range_m": 200,
                "is_hardware": False
            }
        }
        return cams

    # =========================================================================
    # POSTGIS GEOMETRY GENERATORS & SPATIAL MATH
    # =========================================================================
    @staticmethod
    def ST_MakePoint(lng: float, lat: float) -> Point:
        """PostGIS ST_MakePoint(lng, lat) -> Point geometry."""
        return Point(lng, lat)

    @staticmethod
    def ST_DistanceSphere(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """
        PostGIS ST_DistanceSphere(geom1, geom2):
        Calculates Great-Circle Haversine distance in meters on the WGS-84 reference ellipsoid.
        """
        R = 6371000.0 # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lng2 - lng1)

        a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return round(R * c, 2)

    def ST_DWithin(self, lat1: float, lng1: float, lat2: float, lng2: float, distance_m: float) -> bool:
        """PostGIS ST_DWithin: Tests whether two geographic points are within distance_m."""
        dist = self.ST_DistanceSphere(lat1, lng1, lat2, lng2)
        return dist <= distance_m

    def ST_MakeLine(self, coords_list: List[List[float]]) -> LineString:
        """PostGIS ST_MakeLine: Creates a 2D or 3D LineString from coordinate array."""
        return LineString(coords_list)

    def ST_Buffer_CoverageCone(self, lat: float, lng: float, range_m: float, heading_deg: float, fov_deg: float) -> Polygon:
        """
        PostGIS ST_Buffer specialized: Generates a realistic camera field-of-view (FOV) coverage cone polygon
        pointing in the camera's azimuth heading direction.
        """
        # 1 degree latitude ~ 111,320 meters
        # 1 degree longitude ~ 111,320 * cos(lat) meters
        m_per_deg_lat = 111320.0
        m_per_deg_lng = 111320.0 * math.cos(math.radians(lat))

        points = [(lng, lat)]
        start_angle = heading_deg - (fov_deg / 2.0)
        end_angle = heading_deg + (fov_deg / 2.0)

        # Generate arc points
        num_steps = 16
        for step in range(num_steps + 1):
            theta = start_angle + (step / float(num_steps)) * (end_angle - start_angle)
            rad = math.radians(theta)
            # Standard compass: 0=North, 90=East, 180=South, 270=West
            dx_m = range_m * math.sin(rad)
            dy_m = range_m * math.cos(rad)
            arc_lng = lng + (dx_m / m_per_deg_lng)
            arc_lat = lat + (dy_m / m_per_deg_lat)
            points.append((arc_lng, arc_lat))

        points.append((lng, lat)) # Close polygon
        return Polygon(points)

    def get_terrain_elevation(self, lat: float, lng: float) -> Dict[str, Any]:
        """
        Topographic Terrain Elevation Engine:
        Computes realistic digital elevation model (DEM) altitude (in meters ASL) using
        inverse distance weighted (IDW) spatial interpolation across Western Ghats anchor points.
        """
        weights = []
        elevations = []
        for a_lat, a_lng, a_elev, name in ELEVATION_ANCHORS:
            d = self.ST_DistanceSphere(lat, lng, a_lat, a_lng)
            if d < 10.0:
                return {
                    "elevation_m": round(a_elev, 1),
                    "terrain_type": "Mountain Plateau / Ghat Pass",
                    "nearest_landmark": name,
                    "accuracy": "EXACT_ANCHOR"
                }
            w = 1.0 / (d ** 2)
            weights.append(w)
            elevations.append(a_elev)

        sum_w = sum(weights)
        interpolated = sum(w * e for w, e in zip(weights, elevations)) / sum_w if sum_w > 0 else 620.0
        elev = round(interpolated, 1)

        # Determine terrain category
        if elev >= 700:
            t_type = "High Sahyadri Peak / Cliff Ridge"
        elif elev >= 580:
            t_type = "Lonavala-Maval Mountain Plateau"
        elif elev >= 350:
            t_type = "Khandala Ghat Incline / Escarpment"
        else:
            t_type = "Konkan Valley Basin / Highway Foothills"

        return {
            "elevation_m": elev,
            "terrain_type": t_type,
            "contour_interval_m": 10,
            "slope_warning": elev < 500,
            "accuracy": "DEM_INTERPOLATED"
        }

    # =========================================================================
    # POSTGIS GEOJSON APIS
    # =========================================================================
    def get_cameras_geojson(self) -> Dict[str, Any]:
        """
        Returns all city camera nodes and their FOV coverage cone polygons
        as a standard PostGIS GeoJSON FeatureCollection (RFC 7946).
        """
        features = []
        for cam_id, cam in self.cameras.items():
            # 1. Point Feature
            pt_geom = self.ST_MakePoint(cam["lng"], cam["lat"])
            elev_info = self.get_terrain_elevation(cam["lat"], cam["lng"])
            dist_to_current = self.ST_DistanceSphere(
                self.current_location["lat"], self.current_location["lng"], cam["lat"], cam["lng"]
            )

            pt_feature = {
                "type": "Feature",
                "id": f"cam_node_{cam_id}",
                "geometry": mapping(pt_geom),
                "properties": {
                    "feature_type": "ANPR_CAMERA_NODE",
                    "camera_id": cam_id,
                    "name": cam["name"],
                    "is_hardware": cam["is_hardware"],
                    "elevation_m": elev_info["elevation_m"],
                    "terrain_type": elev_info["terrain_type"],
                    "heading_deg": cam["heading_deg"],
                    "range_m": cam["range_m"],
                    "distance_to_command_post_m": dist_to_current,
                    "postgis_geom_wkt": f"POINT({cam['lng']} {cam['lat']} {elev_info['elevation_m']})"
                }
            }
            features.append(pt_feature)

            # 2. Coverage Cone Polygon Feature
            cone_poly = self.ST_Buffer_CoverageCone(
                cam["lat"], cam["lng"], cam["range_m"], cam["heading_deg"], cam["fov_deg"]
            )
            cone_feature = {
                "type": "Feature",
                "id": f"cam_fov_{cam_id}",
                "geometry": mapping(cone_poly),
                "properties": {
                    "feature_type": "CAMERA_FOV_CONE",
                    "camera_id": cam_id,
                    "range_m": cam["range_m"],
                    "postgis_expr": f"ST_Buffer_Cone(POINT({cam['lng']} {cam['lat']}), {cam['range_m']}m, {cam['heading_deg']}deg)"
                }
            }
            features.append(cone_feature)

        return {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "features": features
        }

    def get_corridors_geojson(self) -> Dict[str, Any]:
        """Returns the highway and ghat road corridors as PostGIS LineString features."""
        features = []
        for c_id, c in self.corridors.items():
            line_geom = self.ST_MakeLine(c["coordinates"])
            length_m = 0.0
            coords = c["coordinates"]
            for i in range(len(coords) - 1):
                p1 = coords[i]
                p2 = coords[i+1]
                length_m += self.ST_DistanceSphere(p1[1], p1[0], p2[1], p2[0])

            f = {
                "type": "Feature",
                "id": c_id,
                "geometry": mapping(line_geom),
                "properties": {
                    "feature_type": "ROAD_CORRIDOR",
                    "corridor_id": c_id,
                    "name": c["name"],
                    "type": c["type"],
                    "speed_limit_kmh": c["speed_limit_kmh"],
                    "length_m": round(length_m, 1),
                    "length_km": round(length_m / 1000.0, 2),
                    "postgis_geom_wkt": line_geom.wkt
                }
            }
            features.append(f)

        return {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": features
        }

    def get_trajectories_geojson(self, sightings_by_plate: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Converts vehicle sightings into 4D Spatio-Temporal PostGIS Trajectories:
        (Longitude, Latitude, Elevation_Z, Timestamp_M).
        """
        features = []
        for plate, sightings in sightings_by_plate.items():
            if len(sightings) < 2:
                continue

            sorted_s = sorted(sightings, key=lambda x: x.get("timestamp", 0))
            coords_4d = []
            waypoints = []

            for s in sorted_s:
                cam_id = s.get("camera_id")
                cam = self.cameras.get(cam_id)
                if cam:
                    lat, lng = cam["lat"], cam["lng"]
                    elev = cam.get("elevation_m", 620.0)
                    ts = s.get("timestamp", 0.0)
                    coords_4d.append([lng, lat])
                    waypoints.append({
                        "camera_id": cam_id,
                        "name": cam["name"],
                        "lat": lat,
                        "lng": lng,
                        "elevation_m": elev,
                        "timestamp": ts,
                        "plate": plate,
                        "confidence": s.get("hypotheses", [{}])[0].get("confidence", 0.9)
                    })

            if len(coords_4d) >= 2:
                line_geom = LineString(coords_4d)
                # Compute total distance and speed
                total_dist_m = 0.0
                for i in range(len(coords_4d) - 1):
                    p1 = coords_4d[i]
                    p2 = coords_4d[i+1]
                    total_dist_m += self.ST_DistanceSphere(p1[1], p1[0], p2[1], p2[0])

                t_start = waypoints[0]["timestamp"]
                t_end = waypoints[-1]["timestamp"]
                dt_sec = max(1.0, t_end - t_start)
                speed_kmh = round((total_dist_m / dt_sec) * 3.6, 1)

                elev_start = waypoints[0]["elevation_m"]
                elev_end = waypoints[-1]["elevation_m"]
                elev_gain = round(elev_end - elev_start, 1)

                wkt_pts = [f"{w['lng']} {w['lat']} {w['elevation_m']} {int(w['timestamp'])}" for w in waypoints]
                f = {
                    "type": "Feature",
                    "id": f"traj_{plate}",
                    "geometry": mapping(line_geom),
                    "properties": {
                        "feature_type": "VEHICLE_TRAJECTORY_LINE",
                        "plate": plate,
                        "waypoints_count": len(waypoints),
                        "total_distance_m": round(total_dist_m, 1),
                        "total_distance_km": round(total_dist_m / 1000.0, 2),
                        "travel_time_sec": round(dt_sec, 1),
                        "average_speed_kmh": speed_kmh,
                        "elevation_gain_m": elev_gain,
                        "waypoints": waypoints,
                        "postgis_wkt": f"LINESTRING ZM ({', '.join(wkt_pts)})"
                    }
                }
                features.append(f)

        return {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": features
        }

    def execute_spatial_proximity_query(self, center_lat: float, center_lng: float, radius_m: float) -> Dict[str, Any]:
        """
        Executes a PostGIS ST_DWithin proximity search around (center_lat, center_lng):
        Finds all cameras, corridor segments, and active checkpoints within radius_m.
        """
        t0 = time.time()
        matched_cameras = []
        for cam_id, cam in self.cameras.items():
            dist = self.ST_DistanceSphere(center_lat, center_lng, cam["lat"], cam["lng"])
            if dist <= radius_m:
                matched_cameras.append({
                    "camera_id": cam_id,
                    "name": cam["name"],
                    "lat": cam["lat"],
                    "lng": cam["lng"],
                    "distance_meters": dist,
                    "is_hardware": cam["is_hardware"],
                    "elevation_m": cam["elevation_m"]
                })

        matched_cameras.sort(key=lambda x: x["distance_meters"])

        # Create query buffer polygon for visualization
        # In meters on sphere
        m_per_deg_lat = 111320.0
        m_per_deg_lng = 111320.0 * math.cos(math.radians(center_lat))
        circle_points = []
        for step in range(37):
            rad = math.radians(step * 10)
            dx = radius_m * math.sin(rad)
            dy = radius_m * math.cos(rad)
            circle_points.append((center_lng + dx / m_per_deg_lng, center_lat + dy / m_per_deg_lat))

        buffer_poly = Polygon(circle_points)
        t_exec = round((time.time() - t0) * 1000, 2)

        sql_expr = f"""
SELECT id, name, ST_DistanceSphere(geom, ST_MakePoint({center_lng}, {center_lat})) AS dist_m
FROM anpr_cameras
WHERE ST_DWithin(geom, ST_MakePoint({center_lng}, {center_lat}), {radius_m})
ORDER BY dist_m ASC;
        """.strip()

        result = {
            "query_status": "SUCCESS",
            "postgis_sql": sql_expr,
            "execution_time_ms": t_exec,
            "center": {"lat": center_lat, "lng": center_lng},
            "radius_meters": radius_m,
            "count_matches": len(matched_cameras),
            "matches": matched_cameras,
            "buffer_polygon_geojson": mapping(buffer_poly)
        }
        self.query_history.append(result)
        return result

    def set_current_location(self, lat: float, lng: float, name: str = "Live Command Unit / GPS", source: str = "CLIENT_GPS") -> Dict[str, Any]:
        """
        Dynamically sets current command post location (from laptop GPS, user click, or OAK-D rig).
        Recomputes spatial vectors, distance, and topographic elevation profile.
        """
        elev_info = self.get_terrain_elevation(lat, lng)
        self.current_location = {
            "lat": lat,
            "lng": lng,
            "elevation_m": elev_info["elevation_m"],
            "terrain_type": elev_info["terrain_type"],
            "name": name,
            "source": source,
            "updated_at": time.time()
        }

        # Update OAK-D hardware node coordinates if hardware
        self.cameras["CAM_OAKD"]["lat"] = lat
        self.cameras["CAM_OAKD"]["lng"] = lng
        self.cameras["CAM_OAKD"]["elevation_m"] = elev_info["elevation_m"]

        logger.info(f"Current PostGIS Location updated: ({lat:.5f}, {lng:.5f}) - Elev: {elev_info['elevation_m']}m ({name})")
        return self.get_current_location_summary()

    def get_current_location_summary(self) -> Dict[str, Any]:
        """Returns the active Command Post location with spatial vectors to all cameras."""
        cur_lat = self.current_location["lat"]
        cur_lng = self.current_location["lng"]
        cur_elev = self.current_location["elevation_m"]

        vectors = []
        for cam_id, cam in self.cameras.items():
            if cam_id == "CAM_OAKD":
                continue
            dist_m = self.ST_DistanceSphere(cur_lat, cur_lng, cam["lat"], cam["lng"])
            elev_diff = cam["elevation_m"] - cur_elev
            
            # Compass bearing from current location to camera
            y = math.sin(math.radians(cam["lng"] - cur_lng)) * math.cos(math.radians(cam["lat"]))
            x = math.cos(math.radians(cur_lat)) * math.sin(math.radians(cam["lat"])) - \
                math.sin(math.radians(cur_lat)) * math.cos(math.radians(cam["lat"])) * math.cos(math.radians(cam["lng"] - cur_lng))
            bearing_deg = (math.degrees(math.atan2(y, x)) + 360) % 360

            vectors.append({
                "camera_id": cam_id,
                "name": cam["name"],
                "distance_meters": dist_m,
                "distance_km": round(dist_m / 1000.0, 2),
                "elevation_delta_m": round(elev_diff, 1),
                "bearing_degrees": round(bearing_deg, 1),
                "bearing_cardinal": self._bearing_to_cardinal(bearing_deg),
                "estimated_transit_min": round((dist_m / 1000.0) / 45.0 * 60, 1) # assuming 45 km/h urban speed
            })

        vectors.sort(key=lambda v: v["distance_meters"])

        return {
            "current_location": self.current_location,
            "postgis_point_wkt": f"POINT({cur_lng} {cur_lat} {cur_elev})",
            "srid": SRID_WGS84,
            "spatial_vectors_to_cameras": vectors,
            "nearest_camera": vectors[0] if vectors else None
        }

    def get_terrain_elevation_profile(self, corridor_id: str = "CORR_NH48") -> Dict[str, Any]:
        """
        Returns topographic elevation slice and gradient along a corridor or trajectory.
        Highlights mountain climbs (e.g. Khandala Ghat ascent).
        """
        corridor = self.corridors.get(corridor_id, self.corridors["CORR_NH48"])
        coords = corridor["coordinates"]
        profile = []
        cumulative_dist_m = 0.0

        for i, (lng, lat) in enumerate(coords):
            if i > 0:
                prev_lng, prev_lat = coords[i-1]
                cumulative_dist_m += self.ST_DistanceSphere(prev_lat, prev_lng, lat, lng)

            elev_data = self.get_terrain_elevation(lat, lng)
            profile.append({
                "index": i,
                "lat": lat,
                "lng": lng,
                "distance_km": round(cumulative_dist_m / 1000.0, 2),
                "elevation_m": elev_data["elevation_m"],
                "terrain_type": elev_data["terrain_type"],
                "slope_status": "ASCENDING" if (i > 0 and elev_data["elevation_m"] > profile[i-1]["elevation_m"]) else "DESCENDING"
            })

        return {
            "corridor_id": corridor_id,
            "corridor_name": corridor["name"],
            "total_distance_km": profile[-1]["distance_km"] if profile else 0.0,
            "elevation_min_m": min(p["elevation_m"] for p in profile) if profile else 0.0,
            "elevation_max_m": max(p["elevation_m"] for p in profile) if profile else 0.0,
            "elevation_profile_points": profile
        }

    @staticmethod
    def _bearing_to_cardinal(deg: float) -> str:
        """Converts azimuth degrees to compass cardinal direction."""
        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        ix = round(deg / (360.0 / len(dirs))) % len(dirs)
        return dirs[ix]
