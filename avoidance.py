#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
avoidance.py — Skid-Steer Pivot Kaçınma Kontrolcüsü + Video Kayıt

Skid-steer pivot mantığı:
  x = 0                        (ileri bileşen YOK — yerinde döner)
  r = +AVOIDANCE_STEERING      (sol pivot)
  r = -AVOIDANCE_STEERING      (sağ pivot)

ArduRover MANUAL_CONTROL r convention:
  Pozitif r (+) → sola dön (CCW)
  Negatif r (-) → sağa dön (CW)
"""

import time
import math
import threading
import cv2
import numpy as np
from datetime import datetime

from conf import (
    AVOIDANCE_THROTTLE, AVOIDANCE_STEERING,
    AVOIDANCE_DURATION, AVOIDANCE_COOLDOWN,
    GUIDED_AVOID_SPEED, GUIDED_AVOID_YAW_RATE,
    VIDEO_FPS, VIDEO_FRAME_W, VIDEO_FRAME_H, VIDEO_OUTPUT_DIR,
)

FRAME_W = VIDEO_FRAME_W
FRAME_H = VIDEO_FRAME_H

START_TIME = time.time()

def get_boot_time_ms() -> int:
    return int((time.time() - START_TIME) * 1000)

def send_guided_velocity(master, speed_ms: float, yaw_rate_rad: float):
    """
    GUIDED modda ileri hız + dönüş hızı (yaw rate) komutu gönderir.
    """
    from pymavlink import mavutil
    mask = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    )
    master.mav.set_position_target_local_ned_send(
        get_boot_time_ms(),
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        mask,
        0, 0, 0,                 # x, y, z — yok sayılır
        float(speed_ms), 0, 0,   # vx, vy(yok sayılır), vz(yok sayılır)
        0, 0, 0,                 # ax, ay, az — yok sayılır
        0,                       # yaw (mutlak) — yok sayılır
        float(yaw_rate_rad)      # yaw_rate — kullanılan alan
    )


class AvoidanceController:
    """
    Skid-steer USV için engel kaçınma kontrolcüsü.

    Kaçınma adımları:
      1. DUR  — vx=0 yaw_rate=0, 0.5s (atalet kesilir)
      2. PIVOT — vx=0, yaw_rate=±GUIDED_AVOID_YAW_RATE, DURATION saniye (yerinde dön)
      3. DUR  — vx=0 yaw_rate=0, 0.2s (dönüş sonrası sabitlen)

    Cooldown: kaçınma sonrası AVOIDANCE_COOLDOWN saniye boyunca
    tekrar tetiklenmez.
    """

    def __init__(self):
        self._last_time = 0.0
        self._avoiding  = False
        self._lock      = threading.Lock()

    # ------------------------------------------------------------------
    # Durum sorgu API'si
    # ------------------------------------------------------------------

    def should_avoid(self) -> bool:
        """Cooldown geçtiyse True döner."""
        return (time.time() - self._last_time) > AVOIDANCE_COOLDOWN

    def is_avoiding(self) -> bool:
        with self._lock:
            return self._avoiding

    def cooldown_remaining(self) -> float:
        """Cooldown'dan kalan saniye (0 ise hazır)."""
        return max(0.0, AVOIDANCE_COOLDOWN - (time.time() - self._last_time))

    # ------------------------------------------------------------------
    # Komutlar
    # ------------------------------------------------------------------

    def stop_vehicle(self, master):
        """Aracı durdurur (vx=0 yaw_rate=0)."""
        send_guided_velocity(master, 0.0, 0.0)

    def send_avoid_command(self, master, direction: str):
        """Non-blocking saf pivot dönüş komutu gönderir (vx=0, yaw_rate=±GUIDED_AVOID_YAW_RATE)."""
        # ArduPilot/NED: + = sağa (saat yönü), - = sola (saat yönünün tersi).
        yaw_rate = -GUIDED_AVOID_YAW_RATE if direction == 'left' else GUIDED_AVOID_YAW_RATE
        send_guided_velocity(master, 0.0, yaw_rate)

    def start_cooldown(self):
        """Manevra bitiminde cooldown süresini başlatır."""
        self._last_time = time.time()

    def avoid(self, master, direction: str):
        """
        Skid-steer pivot kaçınma uygular. (Bloklayıcı çağrı — yedek kullanım için)
        """
        if not self.should_avoid():
            return

        with self._lock:
            self._avoiding = True

        if direction == 'left':
            yaw_rate = -GUIDED_AVOID_YAW_RATE
            arrow = '← Sol (pivot)'
        else:
            yaw_rate = +GUIDED_AVOID_YAW_RATE
            arrow = '→ Sağ (pivot)'

        # --- ADIM 1: Dur ---
        print(f"\n  🛑 [STOP] Engel algılandı — duruyorum...")
        t_end = time.time() + 0.5
        while time.time() < t_end:
            send_guided_velocity(master, 0.0, 0.0)
            time.sleep(0.08)

        # --- ADIM 2: Dön ---
        print(f"  🔄 [PIVOT] Kaçınma yönü: {arrow}")
        t_end = time.time() + AVOIDANCE_DURATION
        while time.time() < t_end:
            send_guided_velocity(master, GUIDED_AVOID_SPEED, yaw_rate)
            time.sleep(0.08)

        # --- ADIM 3: Sabitle ---
        for _ in range(3):
            send_guided_velocity(master, 0.0, 0.0)
            time.sleep(0.08)

        self.start_cooldown()

        with self._lock:
            self._avoiding = False

        print(f"  ✅ Kaçınma tamamlandı. {AVOIDANCE_COOLDOWN}s cooldown başladı.")


# ======================================================================
# Video Kaydedici
# ======================================================================

class ScreenRecorder:
    """
    RGB + Derinlik haritasını yan yana koyarak MP4'e kaydeder.
    Çıktı: recordings/usv_YYYYMMDD_HHMMSS.mp4
    HUD üzerinde mesafe, engel durumu, duba tespitleri ve GPS izi gösterilir.
    """

    _TRACK_LEN = 300   # Saklanacak maksimum GPS noktası

    def __init__(self, output_dir: str = VIDEO_OUTPUT_DIR):
        import os
        os.makedirs(output_dir, exist_ok=True)
        ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path     = f"{output_dir}/usv_{ts}.mp4"
        fourcc        = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer  = cv2.VideoWriter(self.path, fourcc, VIDEO_FPS, (FRAME_W, FRAME_H))
        self._lock       = threading.Lock()
        self._frame      = None
        self._frame_seq  = 0   # Her yeni frame'de artar — loop tekrar yazmayı önler
        self._last_seq   = -1  # Loop'un son yazdığı seq numarası
        self._running    = False
        self._thread     = None
        self._frames_written = 0
        # Duba tespitleri ve GPS izi (thread-safe değil ama sadece ana thread yazar)
        self._buoy_dets   = None
        self._track       = []   # [(lat, lon), ...] GPS izi

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()
        print(f"✓ Video kaydı başlatıldı → {self.path}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        # Son bekleyen frame'i garantiyle yaz
        with self._lock:
            frame = self._frame
            seq   = self._frame_seq
        if frame is not None and seq != self._last_seq:
            self._writer.write(frame)
            self._frames_written += 1
        self._writer.release()
        duration_s = self._frames_written / max(1, VIDEO_FPS)
        print(f"✓ Video kaydedildi → {self.path}  "
              f"({self._frames_written} frame, ~{duration_s:.1f}s)")

    def update(self, rgb_bgra: np.ndarray, depth: np.ndarray,
               info: dict = None,
               buoy_detections: dict = None,
               vehicle_pos: tuple = None,
               path_lateral_m: float = None,
               path_lookahead_m: float = None,
               grid_map = None):
        """
        Ana thread'den çağrılır — yeni frame gönderir.

        Args:
            buoy_detections : detect_colored_buoys() çıktısı (isteğe bağlı)
            vehicle_pos     : (lat, lon) GPS konumu (isteğe bağlı)
            path_lateral_m  : Yanal kaçma/ortalama hedefi (metre)
            path_lookahead_m: Hedef bakış mesafesi (metre)
            grid_map        : 2D Grid harita nesnesi (isteğe bağlı)
        """
        if buoy_detections:
            self._buoy_dets = buoy_detections
        if vehicle_pos and vehicle_pos[0] != 0.0 and vehicle_pos[1] != 0.0:
            self._track.append(vehicle_pos)
            if len(self._track) > self._TRACK_LEN:
                self._track.pop(0)
        frame = _build_frame(rgb_bgra, depth, info, self._buoy_dets, self._track, path_lateral_m, path_lookahead_m, grid_map)
        with self._lock:
            self._frame     = frame
            self._frame_seq += 1   # Yeni frame geldiğini işaretle

    def _write_loop(self):
        while self._running:
            with self._lock:
                frame = self._frame
                seq   = self._frame_seq
            # Yalnızca YENİ bir frame varsa yaz (aynı frame'i tekrar yazma)
            if frame is not None and seq != self._last_seq:
                self._writer.write(frame)
                self._last_seq = seq
                self._frames_written += 1
            time.sleep(1.0 / VIDEO_FPS)


def _build_frame(rgb_bgra: np.ndarray, depth: np.ndarray,
                 info: dict = None,
                 buoy_detections: dict = None,
                 vehicle_track: list = None,
                 path_lateral_m: float = None,
                 path_lookahead_m: float = None,
                 grid_map = None) -> np.ndarray:
    """RGB + Depth görselini 1280x360 HUD'lu kare olarak döner.

    Overlaylar:
      • RGB panel — turuncu/sarı duba daireleri + yol çizgisi + mesafe etiketi
      • Depth panel (sağ alt) — GPS izli mini-harita
    """
    # ---- RGB panel ----
    if rgb_bgra.shape[2] == 4:
        rgb_bgr = cv2.cvtColor(rgb_bgra, cv2.COLOR_BGRA2BGR)
    else:
        rgb_bgr = rgb_bgra
    rgb_small = cv2.resize(rgb_bgr, (640, 360))
    scale = 0.5   # 1280→0640, 720→360

    # ---- Yol ve Carrot Overlay (RGB panel) ----
    if path_lateral_m is not None and path_lookahead_m is not None:
        h = 0.45  # Kameranın su seviyesinden yaklaşık yüksekliği (metre)
        z_start = 0.8  # Yol çiziminin başlayacağı ön mesafe (metre)
        denom = max(0.1, path_lookahead_m - z_start)
        
        pts = []
        for z in np.linspace(z_start, path_lookahead_m, 12):
            # Yol eğrisi: z_start'ta 0 sapmadan başlar, path_lookahead_m'de tam sapmaya ulaşır
            x = (z - z_start) * (path_lateral_m / denom)
            u = int((x / z) * 320 + 320)
            v = int((h / z) * 320 + 180)
            if 0 <= u < 640 and 0 <= v < 360:
                pts.append((u, v))
        
        # Yol çizgisini sarı renkte kesintisiz çiz
        for i in range(len(pts) - 1):
            cv2.line(rgb_small, pts[i], pts[i+1], (0, 255, 255), 2, cv2.LINE_AA)
            
        # Carrot hedefini kırmızı halka ve etiketle göster
        if pts:
            u_c, v_c = pts[-1]
            cv2.circle(rgb_small, (u_c, v_c), 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(rgb_small, (u_c, v_c), 9, (255, 255, 255), 1, cv2.LINE_AA)
            lbl = f"Carrot: {path_lateral_m:+.1f}m"
            cv2.putText(rgb_small, lbl, (u_c + 12, v_c + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    # ---- Duba Overlay (RGB panel) ----
    if buoy_detections:
        _COLORS = {
            'orange': (0,  140, 255),   # turuncu (BGR)
            'yellow': (0,  210, 210),   # sarı (BGR)
            'red':    (0,  0,   255),   # kırmızı (BGR)
            'green':  (0,  255, 0),     # yeşil (BGR)
            'black':  (255, 255, 255),  # siyah (beyaz / BGR)
        }
        for color_key, bgr in _COLORS.items():
            for cx, cy, dist in buoy_detections.get(color_key, []):
                px = int(cx * scale)
                py = int(cy * scale)
                # Dünya dubası çemberi
                cv2.circle(rgb_small, (px, py), 14, bgr, 2)
                cv2.circle(rgb_small, (px, py),  3, bgr, -1)
                # Mesafe etiketi
                label = f"{dist:.1f}m"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.rectangle(rgb_small,
                              (px + 8, py - th - 6),
                              (px + 8 + tw + 4, py - 2),
                              (20, 20, 20), -1)
                cv2.putText(rgb_small, label,
                            (px + 10, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr, 1, cv2.LINE_AA)


    # ---- Depth panel ----
    depth_clip  = np.clip(depth, 0, 10)
    finite_mask = np.isfinite(depth)
    depth_clip  = np.nan_to_num(depth_clip, nan=10.0, posinf=10.0, neginf=10.0)
    depth_uint8 = (depth_clip / 10.0 * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    depth_color[~finite_mask] = 0  # Filtrelenen/su yüzeyi alanları siyah göster
    depth_small = cv2.resize(depth_color, (640, 360))

    # ---- Mini-harita (GPS izi, depth panelinin sağ alt köşesi) ----
    if vehicle_track and len(vehicle_track) >= 2:
        MAP_SZ  = 170   # piksel
        PAD     = 8     # kenar boşluğu (depth panel içinde)

        # GPS noktalarını metre offsetine çevir (ilk nokta orijin)
        origin_lat, origin_lon = vehicle_track[0]
        cos_lat = math.cos(math.radians(origin_lat))

        pts_m = []
        for lat, lon in vehicle_track:
            e = (lon - origin_lon) * 111320.0 * cos_lat   # Doğu (m)
            n = (lat - origin_lat) * 111320.0              # Kuzey (m)
            pts_m.append((e, n))

        xs = [p[0] for p in pts_m]
        ys = [p[1] for p in pts_m]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 5.0)   # en az 5m span
        margin = span * 0.15
        x_min, x_max = min(xs) - margin, max(xs) + margin
        y_min, y_max = min(ys) - margin, max(ys) + margin
        span_p = max(x_max - x_min, y_max - y_min)

        def gps_to_px(e, n):
            px_ = int((e - x_min) / span_p * MAP_SZ)
            py_ = int(MAP_SZ - 1 - (n - y_min) / span_p * MAP_SZ)  # Kuzey yukarı
            return (max(0, min(MAP_SZ - 1, px_)),
                    max(0, min(MAP_SZ - 1, py_)))

        # Mini-harita canvas
        mmap = np.zeros((MAP_SZ, MAP_SZ, 3), dtype=np.uint8)
        mmap[:] = (30, 30, 30)

        # Haritadaki engelleri birikimli olarak çiz
        if grid_map is not None and grid_map.origin_lat is not None:
            from map2d import _CENTER, MAP2D_RESOLUTION
            grid_array = grid_map._grid
            rows, cols = np.where(grid_array > 50)
            cos_lat_grid = math.cos(math.radians(grid_map.origin_lat))
            for r, c in zip(rows, cols):
                obs_dy = (_CENTER - r) * MAP2D_RESOLUTION
                obs_dx = (c - _CENTER) * MAP2D_RESOLUTION
                
                # Orijine göre GPS konumunu bul
                obs_lat = grid_map.origin_lat + obs_dy / 111320.0
                obs_lon = grid_map.origin_lon + obs_dx / (111320.0 * cos_lat_grid)
                
                # Mini-harita offsetine çevir
                e_obs = (obs_lon - origin_lon) * 111320.0 * cos_lat
                n_obs = (obs_lat - origin_lat) * 111320.0
                
                if x_min <= e_obs <= x_max and y_min <= n_obs <= y_max:
                    px, py = gps_to_px(e_obs, n_obs)
                    cv2.circle(mmap, (px, py), 1, (0, 0, 160), -1)

        # İz çizgisi (cyan)
        track_px = [gps_to_px(e, n) for e, n in pts_m]
        for i in range(1, len(track_px)):
            cv2.line(mmap, track_px[i - 1], track_px[i], (200, 180, 0), 1)

        # Araç son konumu (yeşil dolu daire)
        cx_m, cy_m = track_px[-1]
        cv2.circle(mmap, (cx_m, cy_m), 5, (0, 230, 0), -1)
        cv2.circle(mmap, (cx_m, cy_m), 5, (255, 255, 255), 1)

        # Kuzey oku
        cv2.arrowedLine(mmap, (MAP_SZ - 16, 22), (MAP_SZ - 16, 6),
                        (220, 220, 220), 1, tipLength=0.45)
        cv2.putText(mmap, "N", (MAP_SZ - 21, 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220, 220, 220), 1)

        # Ölçek: 1m veya 5m (span'e göre)
        grid_m = 1.0 if span_p < 20 else 5.0
        sc_px  = int(grid_m / span_p * MAP_SZ)
        cv2.line(mmap, (4, MAP_SZ - 6), (4 + sc_px, MAP_SZ - 6), (160, 160, 160), 1)
        cv2.putText(mmap, f"{grid_m:.0f}m",
                    (4, MAP_SZ - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (160, 160, 160), 1)

        # Çerçeve
        cv2.rectangle(mmap, (0, 0), (MAP_SZ - 1, MAP_SZ - 1), (120, 120, 120), 1)

        # Depth paneline yapıştır (sağ alt)
        ox = 640 - MAP_SZ - PAD
        oy = 360 - MAP_SZ - PAD
        # Yarı-saydam birleştirme
        roi = depth_small[oy: oy + MAP_SZ, ox: ox + MAP_SZ]
        blended = cv2.addWeighted(roi, 0.25, mmap, 0.75, 0)
        depth_small[oy: oy + MAP_SZ, ox: ox + MAP_SZ] = blended

    # ---- Canvas birleştir ----
    canvas = np.hstack([rgb_small, depth_small])

    # ---- HUD (mesafe + engel) ----
    if info:
        col    = (0, 0, 255) if info.get('obstacle') else (0, 220, 0)
        status = "!! ENGEL !!" if info.get('obstacle') else "Temiz"
        cv2.putText(canvas, f"SOL   : {info['dist_left']:.1f} m",   (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(canvas, f"MERKEZ: {info['dist_center']:.1f} m", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
        cv2.putText(canvas, f"SAG   : {info['dist_right']:.1f} m",  (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(canvas, status, (520, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)

    # ---- Duba sayı özeti (HUD altı) ----
    if buoy_detections:
        n_orange = len(buoy_detections.get('orange', []))
        n_yellow = len(buoy_detections.get('yellow', []))
        cv2.putText(canvas, f"🟠 Turuncu:{n_orange}  🟡 Sarı:{n_yellow}",
                    (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # ---- Zaman damgası ----
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    cv2.putText(canvas, ts, (890, 348),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    return canvas

