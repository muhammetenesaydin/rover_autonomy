#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
map2d.py — Persistent Landmark Map + Occupancy Grid Renderer

Landmark listesi kalıcı veri kaynağıdır. Grid her karede sıfırdan üretilir.
"""

import math, os, time
import numpy as np
import cv2
from datetime import datetime

from conf import (
    MAP2D_WORLD_SIZE_M, MAP2D_MAX_DEPTH_M, MAP2D_RESOLUTION,
    MAP2D_COL_STEP,
    MAP2D_CAMERA_FOV_H, MAP2D_INFLATE_M,
    MAP2D_MERGE_DIST, MAP2D_EMA_ALPHA, MAP2D_CONFIRM_FRAMES,
    ROI_Y_RATIO, VIDEO_OUTPUT_DIR, IMAGE_WIDTH,
)

_GRID_SIZE  = int(MAP2D_WORLD_SIZE_M / MAP2D_RESOLUTION)
_CENTER     = _GRID_SIZE // 2
_FOV_H_RAD  = math.radians(MAP2D_CAMERA_FOV_H)
_INFLATE_PX = max(0, int(round(MAP2D_INFLATE_M / MAP2D_RESOLUTION)))


class Map2D:
    """Persistent Landmark Map — NED dünya-sabit harita."""

    def __init__(self, output_dir: str = VIDEO_OUTPUT_DIR):
        self._grid      = np.zeros((_GRID_SIZE, _GRID_SIZE), dtype=np.float32)
        self._out_dir   = output_dir
        self.origin_lat = None
        self.origin_lon = None
        self.veh_row    = _CENTER
        self.veh_col    = _CENTER
        self.vehicle_track = []

        # Landmark sistemi
        self._landmarks  = []   # kalıcı dubalar
        self._candidates = []   # aday dubalar (onay bekleyen)

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def update(self, depth: np.ndarray,
               curr_lat=None, curr_lon=None,
               heading_deg=None, buoy_detections=None):

        if curr_lat is None or curr_lon is None or curr_lat == 0.0 or curr_lon == 0.0:
            return

        if self.origin_lat is None:
            self.origin_lat = curr_lat
            self.origin_lon = curr_lon
            print(f"📍 [Map2D] Origin kilitlendi: ({curr_lat:.7f}, {curr_lon:.7f})")

        cos_lat = math.cos(math.radians(self.origin_lat))

        # Araç grid konumu
        dy_v = (curr_lat - self.origin_lat) * 111320.0
        dx_v = (curr_lon - self.origin_lon) * 111320.0 * cos_lat
        self.veh_row = max(0, min(_GRID_SIZE-1, _CENTER - int(round(dy_v / MAP2D_RESOLUTION))))
        self.veh_col = max(0, min(_GRID_SIZE-1, _CENTER + int(round(dx_v / MAP2D_RESOLUTION))))

        if not self.vehicle_track or self.vehicle_track[-1] != (self.veh_row, self.veh_col):
            self.vehicle_track.append((self.veh_row, self.veh_col))

        # Duba tespitlerini landmark'a çevir
        if heading_deg is not None and buoy_detections is not None:
            psi = math.radians(heading_deg)
            now = time.monotonic()

            for color in ('orange', 'yellow'):
                for cx_px, _, dist_m in buoy_detections.get(color, []):
                    if dist_m < 0.1 or dist_m > MAP2D_MAX_DEPTH_M:
                        continue

                    angle  = ((cx_px / IMAGE_WIDTH) - 0.5) * _FOV_H_RAD
                    x_body = dist_m * math.sin(angle)
                    z_body = dist_m * math.cos(angle)

                    north = z_body * math.cos(psi) - x_body * math.sin(psi)
                    east  = z_body * math.sin(psi) + x_body * math.cos(psi)

                    # Araç konumuna göre mutlak NED
                    abs_n = dy_v + north
                    abs_e = dx_v + east

                    self._process_detection(color, abs_n, abs_e, now)

        # Grid'i sıfırdan üret
        self._rebuild_grid()

    # ------------------------------------------------------------------
    def _process_detection(self, color, north, east, now):
        """Yeni tespiti mevcut landmark veya aday ile eşleştir."""
        alpha = MAP2D_EMA_ALPHA
        merge = MAP2D_MERGE_DIST

        # 1) Mevcut landmark'larla eşleştir (aynı renk)
        for lm in self._landmarks:
            if lm['color'] != color:
                continue
            d = math.hypot(north - lm['north'], east - lm['east'])
            if d < merge:
                lm['north'] = (1 - alpha) * lm['north'] + alpha * north
                lm['east']  = (1 - alpha) * lm['east']  + alpha * east
                lm['confidence'] += 1
                lm['last_seen'] = now
                return

        # 2) Mevcut adaylarla eşleştir (aynı renk)
        for cand in self._candidates:
            if cand['color'] != color:
                continue
            d = math.hypot(north - cand['north'], east - cand['east'])
            if d < merge:
                cand['north'] = (1 - alpha) * cand['north'] + alpha * north
                cand['east']  = (1 - alpha) * cand['east']  + alpha * east
                cand['hits'] += 1
                cand['last_seen'] = now

                # Yeterli onay → kalıcı landmark'a terfi
                if cand['hits'] >= MAP2D_CONFIRM_FRAMES:
                    self._landmarks.append({
                        'color':      cand['color'],
                        'north':      cand['north'],
                        'east':       cand['east'],
                        'confidence': cand['hits'],
                        'last_seen':  now,
                    })
                    self._candidates.remove(cand)
                return

        # 3) Yeni aday oluştur
        self._candidates.append({
            'color': color, 'north': north, 'east': east,
            'hits': 1, 'last_seen': now,
        })

        # Eski adayları temizle (2 saniye görülmeyeni sil)
        self._candidates = [c for c in self._candidates if now - c['last_seen'] < 2.0]

    # ------------------------------------------------------------------
    def _rebuild_grid(self):
        """Grid'i sıfırla ve tüm landmark'ları yeniden çiz."""
        self._grid.fill(0)
        for lm in self._landmarks:
            row = _CENTER - int(round(lm['north'] / MAP2D_RESOLUTION))
            col = _CENTER + int(round(lm['east']  / MAP2D_RESOLUTION))
            if 0 <= row < _GRID_SIZE and 0 <= col < _GRID_SIZE:
                if _INFLATE_PX <= 1:
                    self._grid[row, col] = 255.0
                else:
                    cv2.circle(self._grid, (col, row), _INFLATE_PX, 255.0, thickness=-1)

    # ------------------------------------------------------------------
    def get_grid(self):
        return self._grid, self.origin_lat, self.origin_lon, self.veh_row, self.veh_col

    def get_landmarks(self):
        """Planlayıcılar için landmark listesini döner."""
        return list(self._landmarks)

    # ------------------------------------------------------------------
    def save_final_png(self, output_dir=None):
        if self.origin_lat is None:
            print("⚠️ [Map2D] Origin kilitlenmedi, harita kaydedilemedi.")
            return None
        d    = output_dir or self._out_dir
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{d}/map2d_world_final_{ts}.png"
        cv2.imwrite(path, self._render())
        print(f"✓ Map2D kaydedildi → {path}")
        return path

    def _render(self):
        grid = np.clip(self._grid, 0, 255).astype(np.uint8)
        img  = np.full((_GRID_SIZE, _GRID_SIZE, 3), 240, dtype=np.uint8)

        # Izgara
        step = int(1.0 / MAP2D_RESOLUTION)
        for i in range(0, _GRID_SIZE, step):
            c = (200,200,200) if (i % (step*5) == 0) else (225,225,225)
            t = 2 if (i % (step*5) == 0) else 1
            cv2.line(img, (0,i), (_GRID_SIZE-1,i), c, t)
            cv2.line(img, (i,0), (i,_GRID_SIZE-1), c, t)

        # Engeller
        occ = grid > 0
        dark = (grid.astype(np.float32) / 255.0) * 240.0
        pv   = np.clip(240.0 - dark, 0, 240).astype(np.uint8)
        for ch in range(3):
            img[:,:,ch] = np.where(occ, pv, img[:,:,ch])

        # Araç izi
        if len(self.vehicle_track) > 1:
            for i in range(1, len(self.vehicle_track)):
                p1 = (self.vehicle_track[i-1][1], self.vehicle_track[i-1][0])
                p2 = (self.vehicle_track[i][1],   self.vehicle_track[i][0])
                cv2.line(img, p1, p2, (200,50,0), 2)

        # Araç konumu
        cv2.circle(img, (self.veh_col, self.veh_row), 6, (0,0,220), -1)
        cv2.circle(img, (self.veh_col, self.veh_row), 6, (0,0,0), 1)

        # Bilgi
        cv2.putText(img, "NED HARITASI (Kuzey Yukari)", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50,50,50), 1, cv2.LINE_AA)
        n_lm = len(self._landmarks)
        n_cd = len(self._candidates)
        cv2.putText(img, f"Landmark:{n_lm}  Aday:{n_cd}", (10,38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1, cv2.LINE_AA)
        cv2.putText(img, f"Hucre:{MAP2D_RESOLUTION*100:.0f}cm", (10,54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1, cv2.LINE_AA)

        # Kuzey oku
        nx, ny = _GRID_SIZE-30, 40
        cv2.arrowedLine(img, (nx,ny+20), (nx,ny-20), (40,40,40), 2, tipLength=0.3)
        cv2.putText(img, "K", (nx-6,ny-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40,40,40), 2)

        # Ölçek
        sc_px = int(5.0 / MAP2D_RESOLUTION)
        mx, my = 20, _GRID_SIZE-20
        cv2.line(img, (mx,my), (mx+sc_px,my), (40,40,40), 3)
        cv2.line(img, (mx,my-5), (mx,my+5), (40,40,40), 2)
        cv2.line(img, (mx+sc_px,my-5), (mx+sc_px,my+5), (40,40,40), 2)
        cv2.putText(img, "5m", (mx+sc_px//2-10,my-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40,40,40), 1)

        return img


# ══════════════════════════════════════════════════════════════════════
# MissionMap — Görev Sonu Tek PNG
# ══════════════════════════════════════════════════════════════════════

class MissionMap:
    _IMG_SIZE  = 1200
    _PAD_RATIO = 0.15

    def __init__(self, output_dir=VIDEO_OUTPUT_DIR, waypoints=None):
        self._out_dir   = output_dir
        self._waypoints = list(waypoints) if waypoints else []
        self._track     = []
        self._obstacles = []
        os.makedirs(output_dir, exist_ok=True)

    def log(self, lat, lon, heading_deg=None, obs_info=None):
        if lat == 0.0 and lon == 0.0:
            return
        self._track.append((lat, lon))

        if obs_info and obs_info.get('obstacle') and heading_deg is not None:
            dc = obs_info.get('dist_center', 1.5)
            dl = obs_info.get('dist_left',   1.5)
            dr = obs_info.get('dist_right',  1.5)
            direction = obs_info.get('direction', 'none')

            if direction == 'left':
                d_obs, side = min(dl, dc), -35.0
            elif direction == 'right':
                d_obs, side = min(dr, dc), +35.0
            else:
                d_obs, side = dc, 0.0

            d_obs = max(0.3, min(d_obs, MAP2D_MAX_DEPTH_M))
            hdg_rad = math.radians(heading_deg + side)
            lat_r   = math.radians(lat)
            obs_lat = lat + (d_obs * math.cos(hdg_rad)) / 111320.0
            obs_lon = lon + (d_obs * math.sin(hdg_rad)) / (111320.0 * math.cos(lat_r))
            self._obstacles.append((obs_lat, obs_lon, direction))

    def save_final_png(self, path=None):
        if len(self._track) < 2:
            print("  ⚠️ MissionMap: Yeterli GPS verisi yok.")
            return None
        if path is None:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{self._out_dir}/mission_map_{ts}.png"
        cv2.imwrite(path, self._render(), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        print(f"✓ Görev iz haritası → {path}")
        return path

    def _render(self):
        SIZE = self._IMG_SIZE
        all_lats = [p[0] for p in self._track] + [p[0] for p in self._obstacles] + [w[0] for w in self._waypoints]
        all_lons = [p[1] for p in self._track] + [p[1] for p in self._obstacles] + [w[1] for w in self._waypoints]

        origin_lat = (min(all_lats)+max(all_lats))/2.0
        origin_lon = (min(all_lons)+max(all_lons))/2.0
        cos_lat    = math.cos(math.radians(origin_lat))

        def to_m(la, lo):
            return (lo-origin_lon)*111320.0*cos_lat, (la-origin_lat)*111320.0

        all_m = [to_m(la,lo) for la,lo in zip(all_lats, all_lons)]
        xs = [p[0] for p in all_m]; ys = [p[1] for p in all_m]
        span = max(max(xs)-min(xs), max(ys)-min(ys), 10.0)
        pad  = span * self._PAD_RATIO
        x_min = min(xs)-pad; x_max = max(xs)+pad
        y_min = min(ys)-pad; y_max = max(ys)+pad
        span  = max(x_max-x_min, y_max-y_min)
        scale = SIZE / span

        def to_px(la, lo):
            x_m, y_m = to_m(la, lo)
            px = int((x_m-x_min)/span*SIZE)
            py = int(SIZE-1-(y_m-y_min)/span*SIZE)
            return (max(0,min(SIZE-1,px)), max(0,min(SIZE-1,py)))

        img = np.full((SIZE,SIZE,3), 248, dtype=np.uint8)

        grid_m = 1.0 if scale >= 15 else (5.0 if scale >= 3 else 10.0)
        x_g = math.floor(x_min/grid_m)*grid_m
        while x_g <= x_max:
            gx = int((x_g-x_min)/span*SIZE)
            if 0 <= gx < SIZE:
                cv2.line(img, (gx,0), (gx,SIZE-1), (218,218,218), 1)
                cv2.putText(img, f"{x_g:.0f}m", (gx+2,SIZE-5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170,170,170), 1)
            x_g += grid_m
        y_g = math.floor(y_min/grid_m)*grid_m
        while y_g <= y_max:
            gy = int(SIZE-1-(y_g-y_min)/span*SIZE)
            if 0 <= gy < SIZE:
                cv2.line(img, (0,gy), (SIZE-1,gy), (218,218,218), 1)
                cv2.putText(img, f"{y_g:.0f}m", (2,gy-3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170,170,170), 1)
            y_g += grid_m

        for i, (wlat,wlon) in enumerate(self._waypoints):
            wp = to_px(wlat,wlon)
            cv2.drawMarker(img, wp, (0,165,255), cv2.MARKER_STAR, 28, 3)
            cv2.putText(img, f"WP{i+1}", (wp[0]+14,wp[1]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,120,200), 2)

        pts = [to_px(la,lo) for la,lo in self._track]
        for i in range(1, len(pts)):
            cv2.line(img, pts[i-1], pts[i], (180,60,0), 2)

        for obs_lat, obs_lon, _ in self._obstacles:
            op = to_px(obs_lat, obs_lon)
            cv2.circle(img, op, 6, (0,0,210), -1)
            cv2.circle(img, op, 6, (0,0,0), 1)

        sp = to_px(*self._track[0])
        cv2.circle(img, sp, 11, (0,180,0), -1)
        cv2.circle(img, sp, 11, (0,0,0), 2)
        cv2.putText(img, "BASLANGIC", (sp[0]+14,sp[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,120,0), 2)

        ep = to_px(*self._track[-1])
        cv2.circle(img, ep, 11, (170,0,170), -1)
        cv2.circle(img, ep, 11, (0,0,0), 2)
        cv2.putText(img, "BITIS", (ep[0]+14,ep[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120,0,120), 2)

        sc_px = int(grid_m*scale)
        mx, my = 30, SIZE-40
        cv2.line(img, (mx,my), (mx+sc_px,my), (40,40,40), 3)
        cv2.line(img, (mx,my-7), (mx,my+7), (40,40,40), 2)
        cv2.line(img, (mx+sc_px,my-7), (mx+sc_px,my+7), (40,40,40), 2)
        cv2.putText(img, f"{grid_m:.0f}m", (mx,my-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40,40,40), 2)

        nx, ny = SIZE-60, 80
        cv2.arrowedLine(img, (nx,ny+35), (nx,ny-35), (40,40,40), 3, tipLength=0.35)
        cv2.putText(img, "K", (nx-9,ny-42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40,40,40), 2)

        n_obs = len(self._obstacles)
        title = f"USV Gorev Haritasi  |  {len(self._track)} konum  |  {n_obs} engel"
        cv2.putText(img, title, (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30,30,30), 2)

        lx, ly = 12, SIZE-130
        items = [
            ((0,180,0),   "Baslangic", "circle"),
            ((170,0,170), "Bitis",     "circle"),
            ((180,60,0),  "Arac izi",  "line"),
            ((0,0,210),   "Engel",     "circle"),
            ((0,165,255), "Waypoint",  "star"),
        ]
        for color, label, shape in items:
            if shape == "circle":
                cv2.circle(img, (lx+8,ly), 7, color, -1)
            elif shape == "line":
                cv2.line(img, (lx,ly), (lx+16,ly), color, 2)
            elif shape == "star":
                cv2.drawMarker(img, (lx+8,ly), color, cv2.MARKER_STAR, 16, 2)
            cv2.putText(img, label, (lx+22,ly+5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30,30,30), 1)
            ly += 22

        return img
