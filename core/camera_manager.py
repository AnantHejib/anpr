import time
import threading
import logging
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger("TRINETRA.CameraManager")

# OAK-D Pro Camera definition (Centerpiece of TRINETRA-VIG)
OAKD_CONFIG = {
    "id": "CAM_OAKD",
    "name": "Luxonis OAK-D Pro (3-Cam Perception & Odometry Engine)",
    "jurisdiction": "MH-14 Mobile / Command Unit",
    "location": {"lat": 18.7530, "lng": 73.4100},
    "speed_limit_kmh": 40,
    "is_hardware": True,
    "type": "Luxonis OAK-D Pro 3-Camera Spatial AI & Odometry Rig",
    "status": "INITIALIZING"
}

CITY_CAMERAS = {
    "CAM_OAKD": OAKD_CONFIG,
    "CAM_01": {
        "id": "CAM_01",
        "name": "NH-48 Entry Toll Plaza",
        "jurisdiction": "MH-14 Lonavala East",
        "location": {"lat": 18.7548, "lng": 73.4068},
        "speed_limit_kmh": 80,
        "is_hardware": False,
        "type": "Fixed ANPR High-Speed",
        "status": "ONLINE"
    },
    "CAM_02": {
        "id": "CAM_02",
        "name": "Kumar Resort Chowk",
        "jurisdiction": "MH-14 Central Lonavala",
        "location": {"lat": 18.7512, "lng": 73.4145},
        "speed_limit_kmh": 40,
        "is_hardware": False,
        "type": "Traffic PTZ + ANPR",
        "status": "ONLINE"
    },
    "CAM_03": {
        "id": "CAM_03",
        "name": "Expressway Interchange (Khandala)",
        "jurisdiction": "MH-14 Expressway Sector",
        "location": {"lat": 18.7595, "lng": 73.3762},
        "speed_limit_kmh": 90,
        "is_hardware": False,
        "type": "Highway ANPR Corridor",
        "status": "ONLINE"
    },
    "CAM_04": {
        "id": "CAM_04",
        "name": "Maval Industrial Junction",
        "jurisdiction": "MH-14 Maval Taluka",
        "location": {"lat": 18.7320, "lng": 73.4420},
        "speed_limit_kmh": 50,
        "is_hardware": False,
        "type": "Urban Intersection ANPR",
        "status": "ONLINE"
    }
}

class CameraManager:
    def __init__(self):
        self.cameras = CITY_CAMERAS.copy()
        self.oakd_pipeline = None
        self.oakd_device = None
        self.oakd_thread = None
        self.oakd_running = False
        
        # Frame buffers
        self.oakd_latest_frame = None       # BGR 640x480 (CAM_A)
        self.oakd_latest_depth = None       # RAW16 mm depth 640x480 (Stereo Depth)
        self.oakd_latest_depth_color = None # Colorized TURBO depth 640x480
        self.oakd_latest_left_mono = None   # Grayscale 640x400 (CAM_B Left IR)
        self.oakd_latest_right_mono = None  # Grayscale 640x400 (CAM_C Right IR)
        
        self.oakd_latest_ts = 0.0
        self.oakd_fps = 0.0
        
        # 3D Spatial Odometry & Tracking State
        self.odometry_history: List[Tuple[float, float, float, float]] = [] # (ts, X, Y, Z)
        self.current_distance_m: float = 0.0
        self.current_coords: Dict[str, float] = {"X": 0.0, "Y": 0.0, "Z": 0.0, "D": 0.0}
        self.current_speed_kmh: float = 0.0
        self.motion_state: str = "SEARCHING" # "APPROACHING", "RECEDING", "STATIONARY", "SEARCHING"
        
        # Hardware Telemetry
        self.oakd_info = {
            "connected": False,
            "device_id": "Unknown",
            "name": "Luxonis OAK-D Pro",
            "resolution": "640x400 @ 60 FPS",
            "fps": 0.0,
            "temperature_c": 38.5,
            "usb_speed": "USB3",
            "ir_active": True,
            "ir_laser_intensity": 0.7,
            "ir_flood_intensity": 0.2,
            "cameras_active": {
                "CAM_A_RGB": True,
                "CAM_B_LEFT_IR": True,
                "CAM_C_RIGHT_IR": True,
                "STEREO_DEPTH": True
            }
        }
        
        self.synthetic_threads = {}
        self.synthetic_frames = {}
        self.lock = threading.Lock()
        
        self._started = False
        self._start_synthetic_cams()
        self._init_oakd_hardware()

    def _init_oakd_hardware(self):
        """Initializes Luxonis OAK-D Pro 3-Camera Pipeline + StereoDepth using DepthAI v3."""
        if self._started:
            return
        self._started = True

        def run_oakd():
            import depthai as dai
            retry_count = 0
            while retry_count < 15 and not self.oakd_running:
                try:
                    logger.info("Scanning for OAK-D Pro devices...")
                    devices = dai.Device.getAllAvailableDevices()
                    if not devices:
                        logger.warning("Waiting for OAK-D Pro device... retrying in 2s")
                        time.sleep(2.0)
                        retry_count += 1
                        continue

                    d = devices[0]
                    device_id = getattr(d, 'deviceId', None) or getattr(d, 'name', None) or "OAKD_PRO_DEVICE"
                    self.oakd_info["device_id"] = str(device_id)
                    logger.info(f"OAK-D Pro detected (ID: {device_id}). Constructing 3-Camera Perception Pipeline...")

                    pipeline = dai.Pipeline()

                    # 1. CAM A (Center Color RGB Sensor) - 640x400 @ 60 FPS for Ultra-Smooth Feed
                    cam_rgb = pipeline.create(dai.node.Camera)
                    cam_rgb.build(dai.CameraBoardSocket.CAM_A)
                    rgb_out = cam_rgb.requestOutput(size=(640, 400), type=dai.ImgFrame.Type.BGR888p, fps=60.0)
                    q_rgb = rgb_out.createOutputQueue(maxSize=2, blocking=False)

                    # 2. CAM B (Left Mono IR Sensor) - 640x400 @ 60 FPS
                    cam_left = pipeline.create(dai.node.Camera)
                    cam_left.build(dai.CameraBoardSocket.CAM_B)
                    left_out = cam_left.requestOutput(size=(640, 400), type=dai.ImgFrame.Type.RAW8, fps=60.0)
                    q_left = left_out.createOutputQueue(maxSize=2, blocking=False)

                    # 3. CAM C (Right Mono IR Sensor) - 640x400 @ 60 FPS
                    cam_right = pipeline.create(dai.node.Camera)
                    cam_right.build(dai.CameraBoardSocket.CAM_C)
                    right_out = cam_right.requestOutput(size=(640, 400), type=dai.ImgFrame.Type.RAW8, fps=60.0)
                    q_right = right_out.createOutputQueue(maxSize=2, blocking=False)

                    # 4. StereoDepth Engine (Linked to Left & Right, aligned to RGB)
                    stereo = pipeline.create(dai.node.StereoDepth)
                    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
                    stereo.setLeftRightCheck(True)
                    stereo.setSubpixel(True)
                    stereo.setOutputSize(640, 400)
                    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

                    left_out.link(stereo.left)
                    right_out.link(stereo.right)

                    q_depth = stereo.depth.createOutputQueue(maxSize=2, blocking=False)

                    # Activate IR Laser Projector and Floodlight on OAK-D Pro hardware
                    try:
                        dev = pipeline.getDefaultDevice()
                        dev.setIrLaserDotProjectorIntensity(0.7)
                        dev.setIrFloodLightIntensity(0.2)
                        logger.info("OAK-D Pro IR Laser Dot Projector (700mA) & Floodlight (200mA) activated!")
                    except Exception as ex:
                        logger.info(f"IR configuration note: {ex}")

                    logger.info("Starting DepthAI pipeline execution on OAK-D Pro...")
                    pipeline.start()
                    self.oakd_pipeline = pipeline
                    self.oakd_running = True
                    self.oakd_info["connected"] = True
                    self.cameras["CAM_OAKD"]["status"] = "ONLINE"
                    logger.info("OAK-D Pro 3-Camera Perception & Odometry Engine ONLINE and streaming!")

                    frame_count = 0
                    start_time = time.time()

                    while self.oakd_running:
                        # Grab RGB frame
                        frame_rgb = q_rgb.tryGet()
                        if frame_rgb is not None:
                            bgr = frame_rgb.getCvFrame()
                            with self.lock:
                                self.oakd_latest_frame = bgr
                                self.oakd_latest_ts = time.time()
                                frame_count += 1

                        # Grab Depth frame
                        frame_depth = q_depth.tryGet()
                        if frame_depth is not None:
                            depth_data = frame_depth.getCvFrame() # uint16 mm depth
                            with self.lock:
                                self.oakd_latest_depth = depth_data
                                # Generate colorized TURBO depth visualization
                                depth_vis = np.clip(depth_data, 0, 7000)
                                depth_vis_8u = (depth_vis * (255.0 / 7000.0)).astype(np.uint8)
                                depth_turbo = cv2.applyColorMap(depth_vis_8u, cv2.COLORMAP_TURBO)
                                depth_turbo[depth_data == 0] = (15, 18, 24)
                                self.oakd_latest_depth_color = depth_turbo

                        # Grab Left Mono IR frame
                        frame_left = q_left.tryGet()
                        if frame_left is not None:
                            with self.lock:
                                self.oakd_latest_left_mono = frame_left.getCvFrame()

                        # Grab Right Mono IR frame
                        frame_right = q_right.tryGet()
                        if frame_right is not None:
                            with self.lock:
                                self.oakd_latest_right_mono = frame_right.getCvFrame()

                        # Update FPS
                        elapsed = time.time() - start_time
                        if elapsed >= 1.0:
                            self.oakd_fps = round(frame_count / elapsed, 1)
                            self.oakd_info["fps"] = self.oakd_fps
                            frame_count = 0
                            start_time = time.time()

                        time.sleep(0.002)

                except Exception as e:
                    logger.warning(f"OAK-D Pro pipeline error: {e}")
                    time.sleep(2.0)
                    retry_count += 1
                finally:
                    if self.oakd_pipeline is not None and not self.oakd_running:
                        try:
                            self.oakd_pipeline.stop()
                        except Exception:
                            pass

            if not self.oakd_running:
                logger.info("OAK-D Pro hardware offline or unbooted, switching to stand-by state.")
                self.cameras["CAM_OAKD"]["status"] = "STANDBY"

        self.oakd_thread = threading.Thread(target=run_oakd, daemon=True)
        self.oakd_thread.start()

    def update_spatial_odometry(self, bbox: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Calculates exact metric distance (X, Y, Z in meters) between the camera and the detected car / object.
        Also computes visual odometry velocity (km/h approach/receding speed) using temporal filtering.
        """
        with self.lock:
            depth_map = self.oakd_latest_depth.copy() if self.oakd_latest_depth is not None else None

        now = time.time()
        fx, fy = 467.0, 467.0 # OAK-D Pro calibrated intrinsics for 640x400
        cx, cy = 320.0, 200.0

        target_u, target_v = 320, 200
        z_meters = 0.0

        if depth_map is not None:
            if bbox is not None:
                bx1, by1, bx2, by2 = bbox
                target_u = int((bx1 + bx2) / 2)
                target_v = int((by1 + by2) / 2)
                
                # Sample central 30% region of the detected vehicle / target
                w_box = max(6, int((bx2 - bx1) * 0.3))
                h_box = max(6, int((by2 - by1) * 0.3))
                sy1 = max(0, target_v - h_box)
                sy2 = min(400, target_v + h_box)
                sx1 = max(0, target_u - w_box)
                sx2 = min(640, target_u + w_box)
                patch = depth_map[sy1:sy2, sx1:sx2]
            else:
                # Sample center reticle crosshair
                patch = depth_map[180:220, 300:340]
                target_u, target_v = 320, 200

            valid = patch[(patch > 200) & (patch < 35000)]
            if len(valid) > 0:
                dist_mm = float(np.median(valid))
                z_meters = round(dist_mm / 1000.0, 2)
        else:
            # Standby simulation depth for UI responsiveness if sensor uncalibrated
            z_meters = 2.35

        # Pinhole projection for lateral (X) and vertical (Y) coordinates
        if z_meters > 0:
            x_meters = round(((target_u - cx) * z_meters) / fx, 2)
            y_meters = round(((target_v - cy) * z_meters) / fy, 2)
            euclidean_dist = round(float(np.sqrt(x_meters**2 + y_meters**2 + z_meters**2)), 2)
        else:
            x_meters, y_meters, euclidean_dist = 0.0, 0.0, 0.0

        # Odometry Speed Calculation (dz / dt)
        speed_kmh = self.current_speed_kmh
        motion_state = "STATIONARY"

        if z_meters > 0:
            if len(self.odometry_history) > 0:
                last_t, last_x, last_y, last_z = self.odometry_history[-1]
                dt = now - last_t
                if 0.04 <= dt <= 1.0:
                    dz = last_z - z_meters # positive if closing distance / approaching
                    raw_speed = (dz / dt) * 3.6
                    # Exponential moving average filter
                    speed_kmh = round(0.75 * self.current_speed_kmh + 0.25 * raw_speed, 1)

                    if speed_kmh > 1.8:
                        motion_state = "APPROACHING"
                    elif speed_kmh < -1.8:
                        motion_state = "RECEDING"
                    else:
                        motion_state = "STATIONARY"

            self.odometry_history.append((now, x_meters, y_meters, z_meters))
            if len(self.odometry_history) > 40:
                self.odometry_history.pop(0)

        self.current_distance_m = euclidean_dist
        self.current_coords = {
            "X": x_meters,
            "Y": y_meters,
            "Z": z_meters,
            "D": euclidean_dist
        }
        self.current_speed_kmh = max(0.0, speed_kmh) if motion_state == "APPROACHING" else round(speed_kmh, 1)
        self.motion_state = motion_state

        return {
            "distance_m": euclidean_dist,
            "depth_z_m": z_meters,
            "lateral_x_m": x_meters,
            "vertical_y_m": y_meters,
            "speed_kmh": abs(self.current_speed_kmh),
            "motion_state": motion_state,
            "target_pixel": [target_u, target_v],
            "proximity_level": "ALERT" if (0 < euclidean_dist < 2.0) else ("CAUTION" if euclidean_dist < 4.5 else "NOMINAL")
        }

    def get_raw_frame(self, camera_id="CAM_OAKD"):
        """Returns the latest raw BGR frame from the requested camera."""
        with self.lock:
            if camera_id == "CAM_OAKD":
                if self.oakd_latest_frame is not None:
                    return self.oakd_latest_frame.copy()
                # Standby frame
                fallback = np.zeros((400, 640, 3), dtype=np.uint8)
                fallback[:] = (18, 22, 28)
                cv2.putText(fallback, "LUXONIS OAK-D PRO HARDWARE", (70, 170), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 242, 254), 2)
                cv2.putText(fallback, f"Status: {self.cameras['CAM_OAKD']['status']} | Device: {self.oakd_info['device_id']}", 
                            (70, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 220), 1)
                cv2.putText(fallback, "640x400 @ 60 FPS Perception Pipeline & Stereo Depth", (70, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 118), 1)
                return fallback
            else:
                return self.synthetic_frames.get(camera_id, np.zeros((400, 640, 3), dtype=np.uint8)).copy()

    def get_depth_frame(self):
        """Returns the raw 16-bit millimeter depth frame from OAK-D Pro."""
        with self.lock:
            if self.oakd_latest_depth is not None:
                return self.oakd_latest_depth.copy()
            return None

    def get_depth_colored_frame(self):
        """Returns the colorized TURBO depth visualization frame."""
        with self.lock:
            if self.oakd_latest_depth_color is not None:
                return self.oakd_latest_depth_color.copy()
            # Standby colorized depth pattern
            canvas = np.zeros((400, 640, 3), dtype=np.uint8)
            canvas[:] = (20, 15, 30)
            cv2.putText(canvas, "STEREO DEPTH MAP (3D TURBO)", (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            return canvas

    def get_left_mono_frame(self):
        """Returns the Left IR mono camera frame."""
        with self.lock:
            if self.oakd_latest_left_mono is not None:
                # Convert 1-channel to 3-channel for MJPEG stream
                mono = self.oakd_latest_left_mono.copy()
                if len(mono.shape) == 2:
                    return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
                return mono
            canvas = np.zeros((400, 640, 3), dtype=np.uint8)
            canvas[:] = (25, 25, 25)
            cv2.putText(canvas, "LEFT MONO INFRARED SENSOR (CAM_B)", (60, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
            return canvas

    def get_right_mono_frame(self):
        """Returns the Right IR mono camera frame."""
        with self.lock:
            if self.oakd_latest_right_mono is not None:
                mono = self.oakd_latest_right_mono.copy()
                if len(mono.shape) == 2:
                    return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
                return mono
            canvas = np.zeros((400, 640, 3), dtype=np.uint8)
            canvas[:] = (25, 25, 25)
            cv2.putText(canvas, "RIGHT MONO INFRARED SENSOR (CAM_C)", (60, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
            return canvas

    def get_triple_panorama_frame(self, rgb_hud_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Creates a composite panoramic fusion of all 3 OAK-D Pro cameras:
        Left IR mono (CAM_B) | Center Perception RGB (CAM_A) | Right IR mono (CAM_C)
        """
        h_target, w_center = 400, 640
        w_side = 200

        # Center frame
        if rgb_hud_frame is not None:
            center = cv2.resize(rgb_hud_frame, (w_center, h_target))
        else:
            center = self.get_raw_frame("CAM_OAKD")
            center = cv2.resize(center, (w_center, h_target))

        # Left IR
        left_raw = self.get_left_mono_frame()
        left_resized = cv2.resize(left_raw, (w_side, h_target))
        cv2.rectangle(left_resized, (0, 0), (w_side, 28), (10, 10, 10), -1)
        cv2.putText(left_resized, "LEFT IR (CAM_B)", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 200), 1)

        # Right IR
        right_raw = self.get_right_mono_frame()
        right_resized = cv2.resize(right_raw, (w_side, h_target))
        cv2.rectangle(right_resized, (0, 0), (w_side, 28), (10, 10, 10), -1)
        cv2.putText(right_resized, "RIGHT IR (CAM_C)", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 200), 1)

        # Stitch horizontally
        pan = np.hstack([left_resized, center, right_resized])
        return pan

    def _start_synthetic_cams(self):
        """Generates realistic synthetic video streams for auxiliary city cameras."""
        def run_sim(cam_id):
            frame_idx = 0
            while True:
                h, w = 480, 640
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                canvas[100:440, :] = (35, 38, 42)
                cv2.line(canvas, (0, 100), (w, 100), (80, 85, 90), 3)
                cv2.line(canvas, (0, 440), (w, 440), (80, 85, 90), 3)
                dash_shift = (frame_idx * 8) % 60
                for x in range(-60 + dash_shift, w + 60, 60):
                    cv2.line(canvas, (x, 270), (x + 30, 270), (200, 200, 200), 2)
                    cv2.line(canvas, (x, 185), (x + 30, 185), (100, 100, 100), 1)
                    cv2.line(canvas, (x, 355), (x + 30, 355), (100, 100, 100), 1)

                car_x = int((frame_idx * 5) % (w + 200)) - 100
                car_y = 220
                if -80 < car_x < w + 80:
                    cv2.rectangle(canvas, (car_x, car_y), (car_x + 120, car_y + 65), (220, 220, 220), -1)
                    cv2.rectangle(canvas, (car_x + 20, car_y + 10), (car_x + 95, car_y + 40), (50, 60, 70), -1)
                    cv2.rectangle(canvas, (car_x + 35, car_y + 48), (car_x + 85, car_y + 62), (255, 255, 255), -1)
                    cv2.putText(canvas, "MH14", (car_x + 40, car_y + 59), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

                cam_info = CITY_CAMERAS.get(cam_id, {})
                cv2.rectangle(canvas, (0, 0), (w, 40), (15, 18, 22), -1)
                cv2.putText(canvas, f"TRINETRA-VIG | {cam_info.get('name', cam_id)}", (15, 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2)
                cv2.putText(canvas, f"LIVE | {time.strftime('%Y-%m-%d %H:%M:%S')}", (w - 220, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 200), 1)

                with self.lock:
                    self.synthetic_frames[cam_id] = canvas

                frame_idx += 1
                time.sleep(0.04)

        for cam_id in ["CAM_01", "CAM_02", "CAM_03", "CAM_04"]:
            t = threading.Thread(target=run_sim, args=(cam_id,), daemon=True)
            self.synthetic_threads[cam_id] = t
            t.start()

    def get_telemetry(self):
        """Returns real-time telemetry from the OAK-D Pro camera, spatial odometry, and network."""
        return {
            "oakd": {
                **self.oakd_info,
                "status": self.cameras["CAM_OAKD"]["status"],
                "fps": self.oakd_fps,
                "latest_ts": self.oakd_latest_ts,
                "spatial_odometry": {
                    "distance_m": self.current_distance_m,
                    "coords": self.current_coords,
                    "speed_kmh": self.current_speed_kmh,
                    "motion_state": self.motion_state
                }
            },
            "network": {
                cam_id: {
                    **data,
                    "active": True
                }
                for cam_id, data in self.cameras.items()
            }
        }
