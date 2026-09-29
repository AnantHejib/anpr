// =========================================================
// TRINETRA-VIG URBAN VEHICLE INTELLIGENCE PLATFORM (app.js)
// SIH 2026 Problem Statement 26127 · Team Tech Innovation Crew
// =========================================================

let map;
let cameraMarkers = {};
let activeTrajectoryLine = null;
let currentSelectedPlate = "KA03MN1188";
let currentCamera = "CAM_OAKD";

// PostGIS & Topographic Terrain Layer State
let tileLayers = {};
let currentLayerKey = "topo";
let commandPostMarker = null;
let postgisBufferLayer = null;
let postgisCorridorsLayer = null;
let postgisConesLayer = null;
let currentLocationData = null;

// Initialize Application on Load
document.addEventListener("DOMContentLoaded", () => {
    initClock();
    initMap();
    fetchCameras();
    fetchCurrentLocation();
    fetchPostgisLayers();
    fetchSystemStatus();
    fetchSightings();
    fetchJourneys();
    fetchAlerts();
    fetchAnalytics();

    // Periodic updates every 2 seconds
    setInterval(() => {
        fetchSystemStatus();
        fetchSightings();
        fetchJourneys();
        fetchAlerts();
        fetchAnalytics();
        fetchCurrentLocation();
    }, 2000);

    // Fast 600ms polling for live detection radar (Detecting or Not)
    pollLiveDetectionRadar();
    setInterval(pollLiveDetectionRadar, 600);

    // Fast 900ms polling for zoomed plate & backend OCR output (/output)
    pollOutputInspection();
    fetchOutputHistory();
    setInterval(pollOutputInspection, 900);

    // Setup Audit Modal Button
    document.getElementById("btn-audit-toggle").addEventListener("click", toggleAuditModal);
});

// Real-Time Clock
function initClock() {
    const clockEl = document.getElementById("live-clock");
    function update() {
        const now = new Date();
        clockEl.textContent = now.toLocaleTimeString("en-IN", { hour12: false }) + " IST";
    }
    update();
    setInterval(update, 1000);
}

let currentStreamMode = "rgb_hud";
let radarSweepAngle = 0;

// Toggle Widescreen Theater Mode for OAK-D Camera
function toggleTheaterMode() {
    const grid = document.querySelector(".main-dashboard-grid");
    const btn = document.getElementById("btn-theater-mode");
    const icon = document.getElementById("theater-icon");
    const text = document.getElementById("theater-text");

    if (!grid) return;

    grid.classList.toggle("theater-mode");
    const isTheater = grid.classList.contains("theater-mode");

    if (btn && icon && text) {
        if (isTheater) {
            btn.classList.add("active");
            icon.textContent = "↙";
            text.textContent = "COLLAPSE";
        } else {
            btn.classList.remove("active");
            icon.textContent = "⛶";
            text.textContent = "EXPAND";
        }
    }

    // Invalidate map size so Leaflet resizes smoothly
    setTimeout(() => {
        if (typeof map !== 'undefined' && map) {
            map.invalidateSize();
        }
    }, 400);
}

// Multi-View Stream Mode Switcher
function switchStreamMode(mode) {
    currentStreamMode = mode;
    
    // Update button states
    const tabs = ["rgb", "depth", "left", "right", "triple"];
    tabs.forEach(t => {
        const btn = document.getElementById(`tab-mode-${t}`);
        if (btn) btn.classList.remove("active");
    });
    
    const activeBtnMap = {
        "rgb_hud": "tab-mode-rgb",
        "depth_color": "tab-mode-depth",
        "left_ir": "tab-mode-left",
        "right_ir": "tab-mode-right",
        "triple_cam": "tab-mode-triple"
    };
    const activeBtn = document.getElementById(activeBtnMap[mode]);
    if (activeBtn) activeBtn.classList.add("active");

    // Update video stream src
    const img = document.getElementById("live-stream-img");
    if (img) {
        img.src = `/api/stream/oakd?mode=${mode}&t=${Date.now()}`;
    }

    // Update badge text
    const badge = document.getElementById("video-cam-badge");
    if (badge) {
        const titles = {
            "rgb_hud": "OAK-D PRO · RGB + STEREO PERCEPTION",
            "depth_color": "OAK-D PRO · STEREO DEPTH 3D TURBO",
            "left_ir": "OAK-D PRO · LEFT MONO IR (CAM_B)",
            "right_ir": "OAK-D PRO · RIGHT MONO IR (CAM_C)",
            "triple_cam": "OAK-D PRO · 3-CAMERA FUSION PANORAMA"
        };
        badge.textContent = titles[mode] || "OAK-D PRO";
    }
}

// 3D Polar Spatial Radar Canvas Renderer
function drawSpatialRadar(targetX, targetZ, proximity) {
    const canvas = document.getElementById("spatial-radar-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    const originX = w / 2;
    const originY = h - 6;
    const maxRadius = h - 14;

    // Draw concentric distance rings (1m, 2m, 3m, 4m, 5m, 6m)
    const ringSteps = [1.5, 3.0, 4.5, 6.0];
    ringSteps.forEach(rDist => {
        const r = (rDist / 6.0) * maxRadius;
        ctx.beginPath();
        ctx.arc(originX, originY, r, Math.PI, 0, false);
        ctx.strokeStyle = "rgba(0, 242, 254, 0.15)";
        ctx.lineWidth = 1;
        ctx.stroke();

        // Label
        ctx.fillStyle = "rgba(180, 200, 220, 0.4)";
        ctx.font = "8px JetBrains Mono, monospace";
        ctx.fillText(`${rDist.toFixed(0)}m`, originX + r - 10, originY - 2);
    });

    // Angle guideline rays (-35°, 0°, +35°)
    const angles = [-0.6, 0, 0.6];
    angles.forEach(ang => {
        const x2 = originX + Math.sin(ang) * maxRadius;
        const y2 = originY - Math.cos(ang) * maxRadius;
        ctx.beginPath();
        ctx.moveTo(originX, originY);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
        ctx.lineWidth = 1;
        ctx.stroke();
    });

    // Radar scanning sweep ray
    radarSweepAngle += 0.08;
    const sweepX = originX + Math.sin(Math.sin(radarSweepAngle) * 0.65) * maxRadius;
    const sweepY = originY - Math.cos(Math.sin(radarSweepAngle) * 0.65) * maxRadius;
    ctx.beginPath();
    ctx.moveTo(originX, originY);
    ctx.lineTo(sweepX, sweepY);
    ctx.strokeStyle = "rgba(0, 242, 254, 0.5)";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Camera origin icon
    ctx.beginPath();
    ctx.arc(originX, originY, 4, 0, Math.PI * 2);
    ctx.fillStyle = "#00f2fe";
    ctx.fill();

    // Plot target blip if distance valid
    if (targetZ > 0 && targetZ < 8.0) {
        // Map (X, Z) in meters to canvas
        const clampedZ = Math.min(6.0, Math.max(0.4, targetZ));
        const blipR = (clampedZ / 6.0) * maxRadius;
        const blipAngle = Math.atan2(targetX, targetZ);
        const blipX = originX + Math.sin(blipAngle) * blipR;
        const blipY = originY - Math.cos(blipAngle) * blipR;

        // Target color based on proximity
        let col = "#00e676"; // green
        if (proximity === "ALERT") col = "#ff3366"; // red
        else if (proximity === "CAUTION") col = "#ffb300"; // amber

        // Blip outer pulse
        ctx.beginPath();
        ctx.arc(blipX, blipY, 7, 0, Math.PI * 2);
        ctx.fillStyle = col + "44";
        ctx.fill();

        // Blip inner dot
        ctx.beginPath();
        ctx.arc(blipX, blipY, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = col;
        ctx.shadowColor = col;
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;

        // Target distance text tag
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 9px JetBrains Mono, monospace";
        ctx.fillText(`${targetZ.toFixed(2)}m`, blipX + 6, blipY - 2);
    }
}

// Live Target Detection & 3D Spatial Odometry Polling
function pollLiveDetectionRadar() {
    fetch("/api/oakd/live_detection")
        .then(res => res.json())
        .then(data => {
            const vDetected = data.vehicle_detected;
            const pDetected = data.plate_detected;
            const odo = data.spatial_odometry || {};

            // 1. Update 3D Metric Distance Readout
            const distEl = document.getElementById("odo-distance-val");
            const xEl = document.getElementById("odo-x-val");
            const yEl = document.getElementById("odo-y-val");
            const zEl = document.getElementById("odo-z-val");
            const speedEl = document.getElementById("odo-speed-val");
            const motionEl = document.getElementById("odo-motion-state");
            const motionIcon = document.getElementById("odo-motion-icon");
            const proxPill = document.getElementById("odo-proximity-pill");

            const distM = odo.distance_m || odo.depth_z_m || 0.0;
            const latX = odo.lateral_x_m || 0.0;
            const vertY = odo.vertical_y_m || 0.0;
            const depthZ = odo.depth_z_m || 0.0;
            const speedKmh = odo.speed_kmh || 0.0;
            const motion = odo.motion_state || "SEARCHING";
            const prox = odo.proximity_level || "NOMINAL";

            if (distEl) {
                distEl.textContent = distM > 0 ? distM.toFixed(2) : "--";
            }
            if (xEl) xEl.textContent = (latX >= 0 ? "+" : "") + latX.toFixed(2);
            if (yEl) yEl.textContent = (vertY >= 0 ? "+" : "") + vertY.toFixed(2);
            if (zEl) zEl.textContent = depthZ.toFixed(2);
            if (speedEl) speedEl.textContent = speedKmh.toFixed(1);
            
            if (motionEl) {
                motionEl.textContent = motion;
                motionEl.style.color = motion === "APPROACHING" ? "var(--accent-green)" : (motion === "RECEDING" ? "var(--accent-amber)" : "var(--accent-cyan)");
            }
            if (motionIcon) {
                motionIcon.textContent = motion === "APPROACHING" ? "⚡" : (motion === "RECEDING" ? "💨" : "🚗");
            }

            if (proxPill) {
                if (prox === "ALERT") {
                    proxPill.className = "proximity-pill alert";
                    proxPill.textContent = "PROXIMITY ALERT";
                } else if (prox === "CAUTION") {
                    proxPill.className = "proximity-pill caution";
                    proxPill.textContent = "CAUTION RANGE";
                } else {
                    proxPill.className = "proximity-pill nominal";
                    proxPill.textContent = "LOCKED ZONE";
                }
            }

            // 2. Draw 3D Spatial Radar
            drawSpatialRadar(latX, depthZ, prox);

            // 3. Update Vehicle detection card
            const vCard = document.getElementById("radar-vehicle-card");
            const vInd = document.getElementById("rc-v-indicator");
            const vStatus = document.getElementById("rc-v-status");
            const vDetail = document.getElementById("rc-v-detail");

            if (vCard && vInd && vStatus && vDetail) {
                if (vDetected) {
                    vCard.className = "radar-card detected";
                    vInd.className = "rc-indicator green";
                    vStatus.textContent = "VEHICLE DETECTED";
                    vDetail.textContent = `${data.primary_vehicle || "Car"} · Dist: ${distM.toFixed(2)}m`;
                } else {
                    vCard.className = "radar-card";
                    vInd.className = "rc-indicator red";
                    vStatus.textContent = "NO VEHICLE IN VIEW";
                    vDetail.textContent = "Searching frame for vehicles...";
                }
            }

            // 4. Update Plate detection card
            const pCard = document.getElementById("radar-plate-card");
            const pInd = document.getElementById("rc-p-indicator");
            const pStatus = document.getElementById("rc-p-status");
            const pDetail = document.getElementById("rc-p-detail");

            if (pCard && pInd && pStatus && pDetail) {
                if (pDetected) {
                    pCard.className = "radar-card plate-locked";
                    pInd.className = "rc-indicator cyan";
                    pStatus.textContent = `LOCKED: ${data.primary_plate}`;
                    const topConf = data.plates?.[0]?.confidence ? Math.round(data.plates[0].confidence * 100) : 94;
                    pDetail.textContent = `Confidence: ${topConf}% · Optical Lock`;
                } else {
                    pCard.className = "radar-card";
                    pInd.className = "rc-indicator amber";
                    pStatus.textContent = "NO PLATE IN VIEW";
                    pDetail.textContent = "Scanning for HSRP license plate...";
                }
            }

            // 5. Update RGB Sensor FPS badge
            const fpsEl = document.getElementById("sensor-rgb-fps");
            if (fpsEl) {
                const camFps = data.fps ? data.fps.toFixed(0) : "60";
                const aiFps = data.ai_fps ? ` · AI: ${data.ai_fps.toFixed(0)} FPS` : "";
                fpsEl.textContent = `640x400 @ ${camFps} FPS${aiFps}`;
            }
        })
        .catch(err => console.debug("Radar fetch error:", err));
}

// =========================================================
// POSTGIS SPATIAL & TOPOGRAPHIC TERRAIN ENGINE (Leaflet)
// =========================================================
function initMap() {
    // 1. Initialize Topographic Terrain, Satellite Hybrid, and Dark Vector Layers
    tileLayers.topo = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}", {
        maxZoom: 19,
        attribution: "Esri World Topographic DEM"
    });

    tileLayers.sat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
        maxZoom: 19,
        attribution: "Esri World Imagery"
    });

    tileLayers.dark = L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
        maxZoom: 19,
        subdomains: "abcd",
        attribution: "CartoDB Dark Matter"
    });

    // Center on Lonavala / Khandala Western Ghats corridor (MH-14 Jurisdiction)
    // DEFAULT LAYER: Topographic Mountain Terrain as explicitly requested!
    map = L.map("city-map", {
        zoomControl: true,
        attributionControl: false,
        layers: [tileLayers.topo]
    }).setView([18.7530, 73.4100], 13);
}

function switchMapLayer(layerKey) {
    if (!tileLayers[layerKey]) return;
    Object.values(tileLayers).forEach(layer => {
        if (map.hasLayer(layer)) map.removeLayer(layer);
    });
    map.addLayer(tileLayers[layerKey]);
    currentLayerKey = layerKey;

    // Update active button state
    document.querySelectorAll(".map-layer-selector .btn").forEach(btn => btn.classList.remove("active-layer"));
    const activeBtn = document.getElementById(`btn-layer-${layerKey}`);
    if (activeBtn) activeBtn.classList.add("active-layer");
}

function fetchCurrentLocation() {
    fetch("/api/postgis/location/current")
        .then(res => res.json())
        .then(data => {
            currentLocationData = data;
            const cur = data.current_location;
            if (!cur) return;

            // Update PostGIS Telemetry Bar
            const coordEl = document.getElementById("pg-cur-coords");
            const elevEl = document.getElementById("pg-cur-elev");
            const terrainEl = document.getElementById("pg-cur-terrain");
            const nearEl = document.getElementById("pg-nearest-cam");

            if (coordEl) coordEl.textContent = `${cur.lat.toFixed(4)}°N, ${cur.lng.toFixed(4)}°E`;
            if (elevEl) elevEl.textContent = `${cur.elevation_m.toFixed(0)} m ASL`;
            if (terrainEl) terrainEl.textContent = `(${cur.terrain_type || "Mountain Plateau"})`;
            if (nearEl && data.nearest_camera) {
                nearEl.textContent = `${data.nearest_camera.name} (${data.nearest_camera.distance_km} km)`;
            }

            // Update or create Command Post Glowing Marker
            if (commandPostMarker) {
                commandPostMarker.setLatLng([cur.lat, cur.lng]);
            } else {
                const cpHtml = `
                    <div class="custom-cam-pin command-post" title="${cur.name}">
                        📍
                    </div>
                `;
                const cpIcon = L.divIcon({
                    html: cpHtml,
                    className: 'cam-icon-wrapper',
                    iconSize: [36, 36],
                    iconAnchor: [18, 18]
                });
                commandPostMarker = L.marker([cur.lat, cur.lng], { icon: cpIcon, zIndexOffset: 1000 }).addTo(map);
                commandPostMarker.bindPopup(`
                    <div style="font-family: Inter, sans-serif; font-size: 12px; color: #111;">
                        <strong style="color: #0284c7; font-size: 13px;">📍 ${cur.name}</strong><br/>
                        <span style="color: #64748b;">PostGIS: ST_MakePoint(${cur.lng.toFixed(4)}, ${cur.lat.toFixed(4)})</span><br/>
                        <span style="color: #0f172a; font-weight: 700;">Altitude: ${cur.elevation_m} m ASL (Western Ghats)</span><br/>
                        <span style="color: #10b981; font-weight: bold;">Status: ACTIVE SENSOR RIG</span><br/>
                        <span style="color: #f59e0b;">Source: ${cur.source}</span>
                    </div>
                `);
            }
        })
        .catch(err => console.debug("Error fetching current location:", err));
}

function locateUserPosition() {
    const btn = document.getElementById("btn-my-location");
    if (btn) btn.textContent = "📍 Locating...";

    if ("geolocation" in navigator) {
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                const lat = pos.coords.latitude;
                const lng = pos.coords.longitude;
                fetch("/api/postgis/location/set", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        lat: lat,
                        lng: lng,
                        name: "Operator Live Device GPS",
                        source: "BROWSER_GEOLOCATION"
                    })
                })
                .then(res => res.json())
                .then(updated => {
                    fetchCurrentLocation();
                    map.flyTo([lat, lng], 14, { duration: 1.5 });
                    if (btn) btn.textContent = "📍 GPS Synced!";
                    setTimeout(() => { if (btn) btn.textContent = "📍 My Location"; }, 3000);
                });
            },
            (err) => {
                console.warn("Geolocation fallback to OAK-D position:", err);
                if (currentLocationData && currentLocationData.current_location) {
                    const c = currentLocationData.current_location;
                    map.flyTo([c.lat, c.lng], 14, { duration: 1.5 });
                }
                if (btn) btn.textContent = "📍 Command Post Centered";
                setTimeout(() => { if (btn) btn.textContent = "📍 My Location"; }, 2500);
            },
            { timeout: 6000, enableHighAccuracy: true }
        );
    } else {
        if (currentLocationData && currentLocationData.current_location) {
            const c = currentLocationData.current_location;
            map.flyTo([c.lat, c.lng], 14, { duration: 1.5 });
        }
        if (btn) btn.textContent = "📍 My Location";
    }
}

function togglePostgisBuffer() {
    if (postgisBufferLayer) {
        map.removeLayer(postgisBufferLayer);
        postgisBufferLayer = null;
        const btn = document.getElementById("btn-postgis-buffer");
        if (btn) btn.textContent = "🎯 ST_DWithin";
        return;
    }

    const cur = currentLocationData?.current_location || { lat: 18.7530, lng: 73.4100 };
    fetch("/api/postgis/spatial_query/within_radius", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lat: cur.lat, lng: cur.lng, radius_m: 3500.0 })
    })
    .then(res => res.json())
    .then(data => {
        const radius = data.radius_meters || 3500;
        postgisBufferLayer = L.circle([cur.lat, cur.lng], {
            radius: radius,
            color: "#00f2fe",
            weight: 2,
            dashArray: "6, 6",
            fillColor: "#00f2fe",
            fillOpacity: 0.12
        }).addTo(map);

        postgisBufferLayer.bindPopup(`
            <div style="font-family: Inter, sans-serif; font-size: 12px; color: #111;">
                <strong style="color: #0284c7;">PostGIS ST_DWithin Buffer</strong><br/>
                <span>Radius: ${radius / 1000} km (${data.count_matches} Cameras Inside)</span><br/>
                <span style="color: #64748b;">Exec Time: ${data.execution_time_ms} ms</span>
            </div>
        `).openPopup();

        map.fitBounds(postgisBufferLayer.getBounds(), { padding: [40, 40] });
        const btn = document.getElementById("btn-postgis-buffer");
        if (btn) btn.textContent = "🎯 Buffer Active";
    })
    .catch(err => console.error("Buffer error:", err));
}

function fetchPostgisLayers() {
    // 1. Fetch Highway & Ghat Road Corridors
    fetch("/api/postgis/corridors/geojson")
        .then(res => res.json())
        .then(geo => {
            if (postgisCorridorsLayer) map.removeLayer(postgisCorridorsLayer);
            postgisCorridorsLayer = L.geoJSON(geo, {
                style: (feature) => {
                    const type = feature.properties.type;
                    if (type === "Access-Controlled Expressway") {
                        return { color: "#ffb300", weight: 4, opacity: 0.85 };
                    } else if (type === "Mountain Ghat Road") {
                        return { color: "#ff3366", weight: 3, opacity: 0.9, dashArray: "4, 4" };
                    }
                    return { color: "#00f2fe", weight: 3, opacity: 0.75 };
                },
                onEachFeature: (feature, layer) => {
                    layer.bindPopup(`
                        <div style="font-family: Inter, sans-serif; font-size: 12px; color: #111;">
                            <strong style="color: #0284c7;">${feature.properties.name}</strong><br/>
                            <span>Type: ${feature.properties.type}</span><br/>
                            <span>Speed Limit: ${feature.properties.speed_limit_kmh} km/h</span><br/>
                            <span>Length: ${feature.properties.length_km} km</span>
                        </div>
                    `);
                }
            }).addTo(map);
        })
        .catch(err => console.debug("Corridors error:", err));

    // 2. Fetch Camera FOV Cones
    fetch("/api/postgis/cameras/geojson")
        .then(res => res.json())
        .then(geo => {
            const conesOnly = {
                type: "FeatureCollection",
                features: geo.features.filter(f => f.properties.feature_type === "CAMERA_FOV_CONE")
            };
            if (postgisConesLayer) map.removeLayer(postgisConesLayer);
            postgisConesLayer = L.geoJSON(conesOnly, {
                style: {
                    color: "#00e676",
                    weight: 1,
                    fillColor: "#00e676",
                    fillOpacity: 0.15
                }
            }).addTo(map);
        })
        .catch(err => console.debug("Cones error:", err));
}

function toggleElevationModal() {
    const modal = document.getElementById("elevation-modal");
    if (!modal) return;
    const isHidden = modal.style.display === "none";
    modal.style.display = isHidden ? "flex" : "none";
    if (isHidden) {
        renderElevationProfile();
    }
}

function renderElevationProfile() {
    fetch("/api/postgis/terrain/elevation_profile")
        .then(res => res.json())
        .then(data => {
            document.getElementById("elev-stat-min").textContent = `${data.elevation_min_m} m`;
            document.getElementById("elev-stat-max").textContent = `${data.elevation_max_m} m`;
            const gain = Math.round(data.elevation_max_m - data.elevation_min_m);
            document.getElementById("elev-stat-gain").textContent = `+${gain} m`;
            document.getElementById("elev-stat-dist").textContent = `${data.total_distance_km} km`;

            const pts = data.elevation_profile_points || [];
            if (pts.length < 2) return;

            const svg = document.getElementById("elev-svg-chart");
            const w = 700, h = 180, padX = 50, padY = 30;
            const minE = 150, maxE = 750;
            const maxD = data.total_distance_km || 15;

            const svgPoints = pts.map(p => {
                const x = padX + (p.distance_km / maxD) * (w - 2 * padX);
                const y = h - padY - ((p.elevation_m - minE) / (maxE - minE)) * (h - 2 * padY);
                return { x, y, p };
            });

            let pathD = `M ${svgPoints[0].x} ${svgPoints[0].y}`;
            for (let i = 1; i < svgPoints.length; i++) {
                pathD += ` L ${svgPoints[i].x} ${svgPoints[i].y}`;
            }

            const areaD = `${pathD} L ${svgPoints[svgPoints.length - 1].x} ${h - padY} L ${svgPoints[0].x} ${h - padY} Z`;

            let circlesHtml = svgPoints.map(pt => `
                <circle cx="${pt.x}" cy="${pt.y}" r="4" fill="#00f2fe" stroke="#fff" stroke-width="1.5" />
                <text x="${pt.x}" y="${pt.y - 8}" fill="#00f2fe" font-size="9" font-family="monospace" text-anchor="middle">${pt.p.elevation_m}m</text>
            `).join("");

            svg.innerHTML = `
                <defs>
                    <linearGradient id="elevGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stop-color="#00f2fe" stop-opacity="0.45" />
                        <stop offset="100%" stop-color="#00f2fe" stop-opacity="0.02" />
                    </linearGradient>
                </defs>
                <line x1="${padX}" y1="${h - padY}" x2="${w - padX}" y2="${h - padY}" stroke="#223344" stroke-width="1" />
                <path d="${areaD}" fill="url(#elevGrad)" />
                <path d="${pathD}" fill="none" stroke="#00f2fe" stroke-width="2.5" />
                ${circlesHtml}
            `;
        });
}

function toggleSqlModal() {
    const modal = document.getElementById("sql-modal");
    if (!modal) return;
    modal.style.display = modal.style.display === "none" ? "flex" : "none";
}

function setSqlPreset(presetKey) {
    const input = document.getElementById("sql-query-input");
    if (!input) return;
    if (presetKey === "within_5k") {
        input.value = "SELECT id, name, ST_DistanceSphere(geom, ST_MakePoint(73.4100, 18.7530)) AS distance_meters, elevation_m FROM anpr_cameras WHERE ST_DWithin(geom, ST_MakePoint(73.4100, 18.7530), 5000) ORDER BY distance_meters ASC;";
    } else if (presetKey === "corridors_length") {
        input.value = "SELECT corridor_id, name, type, speed_limit_kmh, ST_Length(geom) / 1000 AS length_km FROM road_corridors ORDER BY length_km DESC;";
    } else {
        input.value = "SELECT camera_id, name, ST_AsText(geom) AS postgis_wkt, elevation_m FROM anpr_cameras;";
    }
}

function executePostgisSql() {
    const input = document.getElementById("sql-query-input");
    const statusEl = document.getElementById("sql-exec-status");
    const timeEl = document.getElementById("sql-exec-time");
    const thead = document.getElementById("sql-results-thead");
    const tbody = document.getElementById("sql-results-tbody");
    if (!input || !tbody) return;

    statusEl.textContent = "Executing PostGIS spatial query...";
    fetch("/api/postgis/sql_console", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: input.value })
    })
    .then(res => res.json())
    .then(data => {
        statusEl.textContent = `Returned ${data.row_count} rows in ${data.execution_time_ms} ms`;
        timeEl.textContent = `Execution Time: ${data.execution_time_ms} ms`;

        thead.innerHTML = `<tr>${data.columns.map(c => `<th>${c}</th>`).join("")}</tr>`;
        if (data.rows.length === 0) {
            tbody.innerHTML = `<tr><td colspan="${data.columns.length}" style="text-align: center; color: var(--text-muted);">No records matched query.</td></tr>`;
        } else {
            tbody.innerHTML = data.rows.map(r => `
                <tr>${data.columns.map(c => `<td>${r[c] !== undefined ? r[c] : ""}</td>`).join("")}</tr>
            `).join("");
        }
    })
    .catch(err => {
        statusEl.textContent = "Query execution error: " + err;
    });
}

function fetchCameras() {
    fetch("/api/cameras")
        .then(res => res.json())
        .then(cameras => {
            cameras.forEach(cam => {
                const lat = cam.location.lat;
                const lng = cam.location.lng;
                const isHw = cam.is_hardware;

                const pinHtml = `
                    <div class="custom-cam-pin ${isHw ? 'hw' : ''}" title="${cam.name}">
                        ${isHw ? '📷' : '📹'}
                    </div>
                `;

                const customIcon = L.divIcon({
                    html: pinHtml,
                    className: 'cam-icon-wrapper',
                    iconSize: [32, 32],
                    iconAnchor: [16, 16]
                });

                const marker = L.marker([lat, lng], { icon: customIcon }).addTo(map);
                marker.bindPopup(`
                    <div style="font-family: Inter, sans-serif; font-size: 12px; color: #111;">
                        <strong style="color: #0f172a; font-size: 13px;">${cam.name}</strong><br/>
                        <span style="color: #64748b;">ID: ${cam.id} | ${cam.type}</span><br/>
                        <span style="color: #0284c7;">Jurisdiction: ${cam.jurisdiction}</span><br/>
                        <span style="color: #10b981; font-weight: bold;">Status: ${cam.status}</span><br/>
                        <button onclick="switchCamera('${cam.id}')" style="margin-top: 6px; padding: 3px 8px; background: #0284c7; color: #fff; border: none; border-radius: 4px; cursor: pointer;">
                            View Live Stream
                        </button>
                    </div>
                `);

                cameraMarkers[cam.id] = { marker, data: cam };
            });

            // Draw default trajectory for KA03MN1188
            drawVehicleTrajectory(["CAM_01", "CAM_02"]);
        })
        .catch(err => console.error("Error fetching cameras:", err));
}

function resetMapView() {
    if (map) map.setView([18.7530, 73.4100], 13);
}

function fitAllCameras() {
    const latLngs = Object.values(cameraMarkers).map(c => [c.data.location.lat, c.data.location.lng]);
    if (latLngs.length > 0) {
        map.fitBounds(L.latLngBounds(latLngs), { padding: [40, 40] });
    }
}

function drawVehicleTrajectory(camIdList, isAlert = false) {
    if (activeTrajectoryLine) {
        map.removeLayer(activeTrajectoryLine);
        activeTrajectoryLine = null;
    }

    const points = [];
    camIdList.forEach(id => {
        if (cameraMarkers[id]) {
            points.push([cameraMarkers[id].data.location.lat, cameraMarkers[id].data.location.lng]);
        }
    });

    if (points.length >= 2) {
        activeTrajectoryLine = L.polyline(points, {
            color: isAlert ? "#ff3366" : "#00f2fe",
            weight: 4,
            opacity: 0.85,
            dashArray: "8, 8"
        }).addTo(map);

        map.fitBounds(activeTrajectoryLine.getBounds(), { padding: [50, 50] });
    }
}

// =========================================================
// CAMERA STREAM SWITCHING
// =========================================================
function switchCamera(camId) {
    currentCamera = camId;
    const select = document.getElementById("camera-select");
    if (select) select.value = camId;

    const streamImg = document.getElementById("live-stream-img");
    const badge = document.getElementById("video-cam-badge");

    if (camId === "CAM_OAKD") {
        streamImg.src = `/api/stream/oakd?mode=${currentStreamMode || "rgb_hud"}&t=${Date.now()}`;
        badge.textContent = "CAM_OAKD · LUXONIS OAK-D PRO (RGB + HUD)";
    } else {
        streamImg.src = `/api/stream/${camId}?t=${Date.now()}`;
        const name = cameraMarkers[camId]?.data?.name || camId;
        badge.textContent = `${camId} · ${name.toUpperCase()}`;
    }
}

function handleStreamError() {
    console.warn("Video stream connection re-establishing...");
}

// =========================================================
// SYSTEM STATUS & TELEMETRY
// =========================================================
function fetchSystemStatus() {
    fetch("/api/status")
        .then(res => res.json())
        .then(data => {
            document.getElementById("engine-status").textContent = data.status;
            
            const oakd = data.oakd_pro;
            if (oakd) {
                const isConn = oakd.connected;
                document.getElementById("oakd-status").textContent = isConn ? "OAK-D PRO ACTIVE" : "OAK-D STANDBY";
                document.getElementById("oakd-fps-badge").textContent = `${oakd.fps || 25.0} FPS`;
                if (oakd.device_id && oakd.device_id !== "Unknown") {
                    document.getElementById("tel-device-id").textContent = oakd.device_id;
                }
            }
        })
        .catch(err => console.debug("Status fetch:", err));
}

// =========================================================
// SIGHTINGS FEED
// =========================================================
function fetchSightings() {
    fetch("/api/sightings")
        .then(res => res.json())
        .then(sightings => {
            const list = document.getElementById("sightings-list");
            if (!list) return;

            list.innerHTML = "";
            const recent = sightings.slice(-8).reverse();

            recent.forEach(s => {
                const primaryPlate = s.resolved_plate || s.raw_reading || "MH14";
                const isRepaired = s.retroactively_repaired;
                const conf = Math.round((s.composite_score || 0.88) * 100);

                const item = document.createElement("div");
                item.className = "sighting-chip";
                item.onclick = () => selectVehicleJourney(primaryPlate);

                item.innerHTML = `
                    <div>
                        <span class="chip-plate">${primaryPlate}</span>
                        ${isRepaired ? '<span title="Retroactively Repaired" style="font-size:10px; color:#00f2fe;"> 🔄</span>' : ''}
                        <div class="chip-meta">${s.camera_id} · ${s.color || ''} ${s.class || 'Car'}</div>
                    </div>
                    <span class="chip-conf">${conf}%</span>
                `;
                list.appendChild(item);
            });
        })
        .catch(err => console.debug("Sightings fetch:", err));
}

// =========================================================
// VEHICLE JOURNEYS FEED
// =========================================================
function fetchJourneys() {
    fetch("/api/journeys")
        .then(res => res.json())
        .then(journeys => {
            const container = document.getElementById("journeys-container");
            const countBadge = document.getElementById("journeys-count");
            if (!container) return;

            countBadge.textContent = `${journeys.length} Journeys`;
            container.innerHTML = "";

            journeys.forEach(j => {
                const plate = j.plate;
                const status = j.status || "CONFIRMED";
                const isCloned = (plate === "DL01AB1234");
                const isSelected = (plate === currentSelectedPlate);

                const card = document.createElement("div");
                card.className = `journey-card ${isSelected ? 'selected' : ''}`;
                card.onclick = () => selectVehicleJourney(plate);

                const routeStr = (j.trajectory || ["CAM_01"]).join(' <span class="j-arrow">➔</span> ');

                card.innerHTML = `
                    <div class="j-top">
                        <span class="j-plate">${plate}</span>
                        <span class="j-status-pill ${isCloned ? 'cloned' : 'confirmed'}">
                            ${isCloned ? '🚨 CLONED' : status}
                        </span>
                    </div>
                    <div style="font-size: 0.72rem; color: #94a3b8;">
                        ${j.color || 'White'} ${j.vehicle_class || 'Sedan'} · ${j.sightings?.length || 1} Camera Sighting(s)
                    </div>
                    <div class="j-route">
                        Route: ${routeStr}
                    </div>
                `;
                container.appendChild(card);
            });
        })
        .catch(err => console.debug("Journeys fetch:", err));
}

// =========================================================
// EXPLAINABLE AI REASONING INSPECTOR (Slide 2 & 6 Feature)
// =========================================================
function selectVehicleJourney(plate) {
    currentSelectedPlate = plate;

    fetch(`/api/journey/${plate}`)
        .then(res => res.json())
        .then(j => {
            updateReasoningInspector(j);
            if (j.trajectory) {
                drawVehicleTrajectory(j.trajectory, plate === "DL01AB1234");
            }
        })
        .catch(err => console.debug("Journey detail fetch:", err));
}

function updateReasoningInspector(j) {
    const plate = j.plate;
    const isCloned = (plate === "DL01AB1234");

    document.getElementById("insp-plate").textContent = plate;
    const badge = document.getElementById("insp-status-badge");

    if (isCloned) {
        badge.className = "v-status-badge cloned";
        badge.textContent = "CLONED DETECTED";
    } else {
        badge.className = "v-status-badge confirmed";
        badge.textContent = `CONFIRMED (${Math.round((j.confidence || 0.94)*100)}%)`;
    }

    // 1. Candidate Hypotheses
    const candContainer = document.getElementById("insp-candidates");
    candContainer.innerHTML = "";
    const lastSgt = j.sightings?.[j.sightings.length - 1];
    const hypotheses = lastSgt?.hypotheses || [
        { plate: plate, confidence: 0.94 }
    ];

    hypotheses.slice(0, 3).forEach(h => {
        const pct = Math.round(h.confidence * 100);
        const row = document.createElement("div");
        row.className = "cand-row";
        row.innerHTML = `
            <span class="cand-plate mono">${h.plate}</span>
            <div class="cand-bar-wrap">
                <div class="cand-bar" style="width: ${pct}%;"></div>
            </div>
            <span class="cand-conf">${pct}%</span>
        `;
        candContainer.appendChild(row);
    });

    // 2. Appearance Match
    document.getElementById("insp-vclass").textContent = j.vehicle_class || "Car (Sedan)";
    document.getElementById("insp-vcolor").textContent = j.color || "White";
    const reidEl = document.getElementById("insp-vreid");
    if (isCloned) {
        reidEl.className = "app-v match-high";
        reidEl.style.color = "#ff3366";
        reidEl.textContent = "MISMATCH (White Sedan vs Dark SUV)";
    } else {
        reidEl.className = "app-v match-high";
        reidEl.style.color = "#00e676";
        reidEl.textContent = "96.4% MATCH";
    }

    // 3. Road & Time Feasibility
    const distEl = document.getElementById("insp-dist");
    const dtEl = document.getElementById("insp-dt");
    const spdEl = document.getElementById("insp-speed");
    const tagEl = document.getElementById("insp-feasibility-tag");

    if (isCloned) {
        distEl.textContent = "27.2 km (Cam 1 ➔ Cam 3)";
        dtEl.textContent = "0.8 minutes (50 sec)";
        spdEl.textContent = "2040 km/h (Required)";
        tagEl.className = "f-feasibility-tag impossible";
        tagEl.textContent = "⚠️ IMPOSSIBLE TRAVEL TIME ➔ CLONED VEHICLE";
    } else {
        distEl.textContent = "6.2 km (Cam 1 ➔ Cam 2)";
        dtEl.textContent = "11.0 minutes";
        spdEl.textContent = "33.8 km/h (Within Limit: 80)";
        tagEl.className = "f-feasibility-tag";
        tagEl.textContent = "✓ PHYSICALLY CONSISTENT & FEASIBLE";
    }

    // Repair Banner
    const banner = document.getElementById("insp-repair-banner");
    const bannerTxt = document.getElementById("insp-repair-text");
    if (plate === "KA03MN1188") {
        banner.style.display = "flex";
        bannerTxt.innerHTML = `
            Camera B high-confidence read (94%) + Appearance match (White Sedan) + Road-time feasibility (11 mins for 6.2 km) 
            repaired Camera A ambiguous read (46% KA03MN11B8 ➔ KA03MN1188). 
            <em>Original raw sensor reading preserved in audit log.</em>
        `;
    } else {
        banner.style.display = "none";
    }

    // 4. RTO / VAHAN Official Lookup (Slide 6)
    fetchVahanDetails(plate);
}

function fetchVahanDetails(plate) {
    fetch(`/api/vahan/${plate}`)
        .then(res => res.json())
        .then(res => {
            if (res.data) {
                const d = res.data;
                document.getElementById("vahan-model").textContent = d.make_model || "Passenger Car";
                document.getElementById("vahan-rto").textContent = d.rto_jurisdiction || "MH-14";
                document.getElementById("vahan-owner").textContent = d.owner_masked || "CITIZEN REGISTRY";
                const statusEl = document.getElementById("vahan-status");
                if (d.blacklist_status && d.blacklist_status !== "CLEAR") {
                    statusEl.style.color = "#ff3366";
                    statusEl.textContent = d.blacklist_status;
                } else {
                    statusEl.style.color = "#00e676";
                    statusEl.textContent = `${d.puc_status} PUC · CLEAR`;
                }
            }
        })
        .catch(err => console.debug("VAHAN fetch:", err));
}

// =========================================================
// ALERTS PANEL
// =========================================================
function fetchAlerts() {
    fetch("/api/alerts")
        .then(res => res.json())
        .then(alerts => {
            const container = document.getElementById("alerts-container");
            const badge = document.getElementById("alerts-count-badge");
            if (!container) return;

            badge.textContent = `${alerts.length} Active`;
            container.innerHTML = "";

            alerts.forEach(al => {
                const isCrit = al.priority === "CRITICAL";
                const card = document.createElement("div");
                card.className = `alert-card ${isCrit ? 'critical' : 'high'}`;

                card.innerHTML = `
                    <div class="al-title">${al.title}</div>
                    <div class="al-desc">${al.explanation || al.reason || ''}</div>
                    <div class="al-actions">
                        <button class="btn-al dispatch" onclick="handleAlertAction('${al.alert_id}', 'DISPATCH_PATROL')">Dispatch Patrol</button>
                        <button class="btn-al verify" onclick="handleAlertAction('${al.alert_id}', 'VERIFY')">Verify</button>
                        <button class="btn-al dismiss" onclick="handleAlertAction('${al.alert_id}', 'DISMISS')">Dismiss</button>
                    </div>
                `;
                container.appendChild(card);
            });
        })
        .catch(err => console.debug("Alerts fetch:", err));
}

function handleAlertAction(alertId, action) {
    fetch(`/api/alerts/${alertId}/action`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: action })
    })
    .then(res => res.json())
    .then(() => fetchAlerts())
    .catch(err => console.error("Error handling alert:", err));
}

// =========================================================
// TRAFFIC ANALYTICS
// =========================================================
function fetchAnalytics() {
    fetch("/api/analytics")
        .then(res => res.json())
        .then(data => {
            document.getElementById("stat-total-sightings").textContent = data.total_vehicles_detected;
            document.getElementById("stat-avg-speed").textContent = `${data.corridor_avg_speed_kmh} km/h`;
            document.getElementById("stat-density").textContent = data.overall_density;
        })
        .catch(err => console.debug("Analytics fetch:", err));
}

// =========================================================
// INTERACTIVE PPT SCENARIO TRIGGERS
// =========================================================
function triggerScenario(scKey) {
    // Visual button state
    document.querySelectorAll(".btn-scenario").forEach(b => b.classList.remove("active"));
    const activeBtn = document.getElementById(
        scKey === "weak_read_repair" ? "btn-sc-repair" :
        scKey === "cloned_plate" ? "btn-sc-clone" :
        scKey === "watchlist_hit" ? "btn-sc-watchlist" : "btn-sc-oakd"
    );
    if (activeBtn) activeBtn.classList.add("active");

    fetch("/api/scenarios/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario: scKey })
    })
    .then(res => res.json())
    .then(res => {
        if (scKey === "weak_read_repair") {
            selectVehicleJourney("KA03MN1188");
            switchCamera("CAM_02");
        } else if (scKey === "cloned_plate") {
            selectVehicleJourney("DL01AB1234");
            switchCamera("CAM_03");
        } else if (scKey === "watchlist_hit") {
            selectVehicleJourney("MH14DX9921");
            switchCamera("CAM_04");
        } else if (scKey === "oakd_live_capture") {
            switchCamera("CAM_OAKD");
            selectVehicleJourney(res.resolved_plate || "MH14EX9900");
        }

        fetchSightings();
        fetchJourneys();
        fetchAlerts();
        fetchAnalytics();
    })
    .catch(err => console.error("Error triggering scenario:", err));
}

// =========================================================
// AUDIT LOG MODAL
// =========================================================
function toggleAuditModal() {
    const modal = document.getElementById("audit-modal");
    if (modal.style.display === "none") {
        modal.style.display = "flex";
        fetchAuditLog();
    } else {
        modal.style.display = "none";
    }
}

function fetchAuditLog() {
    fetch("/api/audit/repairs")
        .then(res => res.json())
        .then(logs => {
            const tbody = document.getElementById("audit-table-body");
            tbody.innerHTML = "";

            if (logs.length === 0) {
                tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:#94a3b8; padding:20px;">No retroactive repairs recorded yet. Run Scenario 1 to trigger.</td></tr>`;
                return;
            }

            logs.forEach(l => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td class="mono">${new Date(l.timestamp * 1000).toLocaleTimeString("en-IN")}</td>
                    <td><strong>${l.camera_id}</strong></td>
                    <td class="mono" style="color: #ffb300;">${l.original_raw_reading}</td>
                    <td class="mono" style="color: #00e676; font-weight: bold;">${l.repaired_plate}</td>
                    <td>${l.repaired_by_camera}</td>
                    <td style="font-size: 11px; color: #cbd5e1;">${l.explanation}</td>
                `;
                tbody.appendChild(tr);
            });
        })
        .catch(err => console.debug("Audit fetch:", err));
}

// =========================================================
// SEARCH & VEHICLE LOOKUP (Slide 3 & 5)
// =========================================================
function handleSearch(e) {
    if (e.key === "Enter") {
        executeSearch();
    }
}

function executeSearch() {
    const input = document.getElementById("vehicle-search-input");
    const q = input.value.trim().toUpperCase();
    if (!q) return;

    fetch(`/api/search?q=${encodeURIComponent(q)}`)
        .then(res => res.json())
        .then(res => {
            if (res.results && res.results.length > 0) {
                const target = res.results[0];
                selectVehicleJourney(target.plate);
            } else {
                alert(`No vehicle records found matching query: "${q}". Try KA03, DL01, or MH14.`);
            }
        })
        .catch(err => console.error("Search error:", err));
}

// =========================================================
// CONGESTION HEATMAP OVERLAY (Slide 5)
// =========================================================
let heatmapLayers = [];
let heatmapActive = false;

function toggleHeatmap() {
    heatmapActive = !heatmapActive;
    const btn = document.getElementById("btn-heatmap");
    if (btn) {
        btn.textContent = heatmapActive ? "🔥 Heatmap: ON" : "🔥 Heatmap: OFF";
        btn.classList.toggle("active", heatmapActive);
    }

    if (!heatmapActive) {
        heatmapLayers.forEach(l => map.removeLayer(l));
        heatmapLayers = [];
        return;
    }

    // Draw congestion heat circles around cameras
    const heatData = [
        { cam: "CAM_01", color: "#00e676", radius: 450, density: "LOW" },
        { cam: "CAM_02", color: "#ffb300", radius: 750, density: "MEDIUM (KUMAR RESORT INFLOW)" },
        { cam: "CAM_03", color: "#00e676", radius: 400, density: "LOW" },
        { cam: "CAM_04", color: "#00e676", radius: 420, density: "LOW" },
        { cam: "CAM_OAKD", color: "#00e676", radius: 350, density: "LOW" }
    ];

    heatData.forEach(h => {
        if (cameraMarkers[h.cam]) {
            const loc = cameraMarkers[h.cam].data.location;
            const circle = L.circle([loc.lat, loc.lng], {
                color: h.color,
                fillColor: h.color,
                fillOpacity: 0.35,
                radius: h.radius
            }).addTo(map);

            circle.bindTooltip(`<strong>${cameraMarkers[h.cam].data.name}</strong><br/>Density: ${h.density}`, { sticky: true });
            heatmapLayers.push(circle);
        }
    });
}

// =========================================================
// MOBILE / PATROL UNIT MODAL (Slide 3)
// =========================================================
function togglePatrolModal() {
    const modal = document.getElementById("patrol-modal");
    if (modal.style.display === "none") {
        modal.style.display = "flex";
    } else {
        modal.style.display = "none";
    }
}

function submitPatrolUpload() {
    const plate = document.getElementById("patrol-plate-input").value.trim().toUpperCase();
    const vclass = document.getElementById("patrol-class-input").value;
    const color = document.getElementById("patrol-color-input").value;
    const loc = document.getElementById("patrol-loc-input").value;

    fetch("/api/mobile/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            plate: plate,
            vehicle_class: vclass,
            color: color,
            location_desc: loc
        })
    })
    .then(res => res.json())
    .then(res => {
        togglePatrolModal();
        selectVehicleJourney(res.resolved_plate || plate);
        switchCamera("CAM_OAKD");
        fetchSightings();
        fetchJourneys();
        fetchAnalytics();
    })
    .catch(err => console.error("Patrol upload error:", err));
}

// =========================================================
// ZOOMED PLATE & BACKEND OCR INSPECTION (/output)
// =========================================================
let lastOutputTs = 0;

function pollOutputInspection() {
    fetch("/api/output/latest_detection")
        .then(res => res.json())
        .then(data => {
            if (!data) return;

            const isNew = data.timestamp && data.timestamp !== lastOutputTs;
            if (isNew) {
                lastOutputTs = data.timestamp;
            }

            const plateVal = document.getElementById("output-plate-val");
            const confVal = document.getElementById("output-conf-val");
            const tsVal = document.getElementById("output-ts-val");
            const zoomTag = document.getElementById("output-zoom-tag");
            const pathTag = document.getElementById("output-zoomed-path");
            const statusPill = document.getElementById("output-status-pill");
            const zoomedImg = document.getElementById("output-zoomed-plate-img");
            const ocrImg = document.getElementById("output-ocr-card-img");

            if (plateVal) plateVal.textContent = data.plate_text || "--";
            if (confVal) confVal.textContent = (data.confidence && data.confidence > 0) ? Math.round(data.confidence * 100) + "%" : "--";
            if (tsVal) tsVal.textContent = data.timestamp_str || "LIVE";
            if (zoomTag) zoomTag.textContent = (data.zoom_scale || 3.5) + "x OPTICAL ZOOM";
            if (pathTag) pathTag.textContent = data.zoomed_image || "output/latest_plate.png";

            if (statusPill) {
                if (data.status === "SAVED_TO_BACKEND_OUTPUT" || (data.plate_text && data.plate_text !== "SEARCHING")) {
                    statusPill.className = "proximity-pill nominal";
                    statusPill.textContent = "● SAVED TO /output";
                } else {
                    statusPill.className = "proximity-pill caution";
                    statusPill.textContent = "● SCANNING ACTIVE";
                }
            }

            // Only refresh image src when timestamp changes to avoid flicker
            if (isNew && zoomedImg && ocrImg) {
                zoomedImg.src = `/output/latest_plate.png?t=${data.timestamp}`;
                ocrImg.src = `/output/latest_ocr.png?t=${data.timestamp}`;
                fetchOutputHistory();
            }
        })
        .catch(err => console.debug("Output polling error:", err));
}

function refreshOutputInspection() {
    const zoomedImg = document.getElementById("output-zoomed-plate-img");
    const ocrImg = document.getElementById("output-ocr-card-img");
    const now = Date.now();
    if (zoomedImg) zoomedImg.src = `/output/latest_plate.png?t=${now}`;
    if (ocrImg) ocrImg.src = `/output/latest_ocr.png?t=${now}`;
    fetchOutputHistory();
    pollOutputInspection();
}

function fetchOutputHistory() {
    fetch("/api/output/history")
        .then(res => res.json())
        .then(history => {
            const gallery = document.getElementById("saved-plates-gallery");
            if (!gallery || !Array.isArray(history)) return;

            gallery.innerHTML = "";
            if (history.length === 0) {
                gallery.innerHTML = `<span style="font-size:0.7rem; color:var(--text-muted); padding:6px;">No archived plate images yet.</span>`;
                return;
            }

            history.slice(0, 12).forEach(item => {
                const thumb = document.createElement("div");
                thumb.className = "saved-plate-thumb";
                const imgUrl = item.zoomed_image || item.url || `/output/plates/${item.filename}`;
                thumb.title = `${item.plate_text || item.filename || "Plate"} (${item.timestamp_str || ""})`;
                thumb.innerHTML = `<img src="/${imgUrl.replace(/^\/+/, '')}" alt="Saved Plate" onerror="this.style.display='none'" />`;
                thumb.onclick = () => {
                    const zoomedImg = document.getElementById("output-zoomed-plate-img");
                    if (zoomedImg) zoomedImg.src = `/${imgUrl.replace(/^\/+/, '')}`;
                };
                gallery.appendChild(thumb);
            });
        })
        .catch(err => console.debug("Output history fetch error:", err));
}
