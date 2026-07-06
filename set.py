#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
set.py — USV Otonom Navigasyon + ZED Engel Kaçınma (Durum Makinesi)

Durum Makinesi (düşük seviye, MAVLink):
  NAVIGATE  → GUIDED mod, setpoint yenile, engel yokken ilerle
  STOP      → Engel algılandı! GUIDED'da kal, vx=0 yaw_rate=0, 0.5s dur
  TURN      → GUIDED'da kal, pivot dön (vx=0, yaw_rate=±GUIDED_AVOID_YAW_RATE)
  STABILIZE → GUIDED'da kal: dönüş sonu sabitle
  COOLDOWN  → bekle, hâlâ engel var mı kontrol et

  NOT (v4): Araç artık kaçınma sırasında MANUAL moduna HİÇ geçmiyor.
  Tüm STOP/TURN/STABILIZE hareketleri GUIDED modda
  SET_POSITION_TARGET_LOCAL_NED (vx + yaw_rate, body-relative) ile
  yapılıyor — bkz. send_guided_velocity(). set_mode(master,"MANUAL")
  çağrıları kaçınma akışından tamamen kaldırıldı.

Skid-steer için kilit kural:
  Kaçınma sırasında vx=0 — araç yerinde döner, ileri gitmez.

Düzeltmeler (v3):
  - goal_local_coords: gerçek NED→body-frame dönüşümü (v2'den)
  - Sarı ok overlay: _build_wp_overlay() ile doğru açı (v2'den)
  - global_planner.check_waypoint_reached(dist) düzeltildi (v2'den)
  - LocalPlanner.atan2 sırası düzeltildi (v2'den)
  - fuse_obstacles: sağ/sol yan mesafeler artık yalnızca merkez
    mesafesinden de yakınsa tehlikeli sayılıyor — koridor kenarı
    false-positive'lerini engeller (SAG:0.8m iken MERKEZ:2.4m senaryosu)
  - STATE_NAVIGATE near katmanı: ObstacleDetector emergency bayrağı da
    tetikleyici olarak eklendi
  - COOLDOWN: cooldown bitmeden de yakın engel varsa hemen STOP
  - STATE_TURN çıkış koşulu: tüm engel yerine yalnızca MERKEZ engeli
    temizlenince çıkıyor (yan/koridor gürültüsü false-exit'i engeller)
  - turn_dir karar mantığı: 'left'/'right'/'center' tutarlı hâle getirildi
"""

import time
import math
import cv2
import numpy as np
from enum import Enum, auto
from pymavlink import mavutil

from conf import (
    CONNECTION_STRING, GPS_WAYPOINTS,
    SAFETY_DISTANCE, SETPOINT_INTERVAL,
    WAYPOINT_ACCEPTANCE_RADIUS, DOCKING_WAYPOINT_NUM,
    CORRIDOR_MIN_DIST, CORRIDOR_MAX_DIST, CORRIDOR_CORRECT_M,
    CORRIDOR_INTERVAL, CORRIDOR_MIN_WIDTH, IMAGE_WIDTH,
    OBSTACLE_NEAR_LIMIT, OBSTACLE_MED_LIMIT,
    ORANGE_PANIC_DIST, ORANGE_ESCAPE_DIST, ORANGE_ALIGN_DIST,
    GUIDED_AVOID_SPEED, GUIDED_AVOID_YAW_RATE,
)
from zed_obstacle import ZEDCamera, ObstacleDetector
from avoidance    import AvoidanceController, ScreenRecorder
from map2d        import Map2D, MissionMap

START_TIME = time.time()

# ============================================================
# Durum sabitleri (düşük seviye MAVLink döngüsü)
# ============================================================
STATE_NAVIGATE  = 'NAVIGATE'
STATE_STOP      = 'STOP'
STATE_TURN      = 'TURN'
STATE_STABILIZE = 'STABILIZE'
STATE_COOLDOWN  = 'COOLDOWN'


# ============================================================
# MissionState + MissionController
# ============================================================

class MissionState(Enum):
    IDLE               = auto()
    NAVIGATING         = auto()
    AVOIDING_OBSTACLE  = auto()
    DOCKING            = auto()
    COMPLETED          = auto()
    TURNING            = auto()


class MissionController:
    """Görev durum makinesini yönetir."""

    def __init__(self):
        self.state = MissionState.IDLE

    def update(self, waypoint_reached, obstacle_detected,
               target_detected, heading_aligned=False):
        if self.state == MissionState.IDLE:
            self.state = MissionState.NAVIGATING

        elif self.state == MissionState.NAVIGATING:
            if waypoint_reached:
                self.state = MissionState.TURNING
            elif target_detected:
                self.state = MissionState.DOCKING
            elif obstacle_detected:
                self.state = MissionState.AVOIDING_OBSTACLE

        elif self.state == MissionState.TURNING:
            if heading_aligned:
                self.state = MissionState.NAVIGATING

        elif self.state == MissionState.AVOIDING_OBSTACLE:
            if waypoint_reached:
                self.state = MissionState.TURNING
            elif not obstacle_detected:
                self.state = MissionState.NAVIGATING

        elif self.state == MissionState.DOCKING:
            pass   # Yanaşma mantığı ileride eklenecek

        return self.state


# ============================================================
# GlobalPlanner
# ============================================================

class GlobalPlanner:
    """Yüksek seviyeli waypoint rotalarını yönetir."""

    def __init__(self):
        self.waypoints         = GPS_WAYPOINTS
        self.current_wp_index  = 0
        self.acceptance_radius = WAYPOINT_ACCEPTANCE_RADIUS

    def get_current_waypoint(self):
        if self.current_wp_index < len(self.waypoints):
            return self.waypoints[self.current_wp_index]
        return None

    def check_waypoint_reached(self, dist_m: float) -> bool:
        """
        dist_m: hedefe olan düz mesafe (metre).
        DÜZELTME: Eski çağrı check_waypoint_reached(dist, 0) yanlıştı —
        fonksiyon x,y bileşen alıyordu ama dist skaler veriliyordu.
        Artık doğrudan mesafe alıyor.
        """
        if self.current_wp_index >= len(self.waypoints):
            return True
        if dist_m < self.acceptance_radius:
            self.current_wp_index += 1
            return True
        return False


# ============================================================
# LocalPlanner
# ============================================================

class LocalPlanner:
    """
    Occupancy haritası + duba tespitleri + hedef konumu → hız/açı kararı.

    DÜZELTME: goal_local_coords artık gerçek NED→body-frame dönüşümüyle
    hesaplanıyor (set.py ana döngüsünde). Bu fonksiyon doğru girdi alınca
    doğru çalışır.

    atan2 sırası: math.atan2(goal_x, goal_y) — goal_y=ileri, goal_x=sağ,
    pozitif açı = sola dön (CCW). Eski kod atan2(goal_y, goal_x) diyordu
    ve yön tersine dönüyordu.
    """

    def __init__(self):
        pass

    def compute_velocity_command(self, occupancy_grid,
                                 goal_local_coords,
                                 buoy_detections=None,
                                 image_width=IMAGE_WIDTH):
        """
        Returns:
            (linear_vel m/s, angular_vel rad/s)
            linear_vel  : ileri hız (0..5 m/s)
            angular_vel : dönüş hızı (negatif=sağ, pozitif=sol, rad/s)
        """
        if buoy_detections is None:
            buoy_detections = {'orange': [], 'yellow': []}

        orange_buoys   = buoy_detections.get('orange', [])
        yellow_buoys   = buoy_detections.get('yellow', [])
        image_center_x = image_width / 2.0

        goal_x, goal_y     = goal_local_coords          # sağ, ileri (metre)
        waypoint_distance  = math.sqrt(goal_x**2 + goal_y**2)

        # Hedefe olan gerçek açı (body frame):
        # goal_y=ileri ekseni → atan2(goal_x, goal_y): sola pozitif
        # DÜZELTME: eskiden atan2(goal_y, goal_x) yazıyordu → yön tersiydi
        heading_error = math.atan2(goal_x, goal_y)   # sola = pozitif

        # ── FAZ 1: Hedefe yakın (<3m) ────────────────────────────────
        if waypoint_distance < 3.0:
            if abs(heading_error) > 0.52:   # >30° → yerinde dön
                return 0.5, math.copysign(0.8, heading_error)
            else:
                return 2.0, heading_error * 0.5

        # ── Şerit merkezi hesapla (turuncu dubalar) ──────────────────
        lane_center_x = None
        lane_offset   = 0.0
        nearby_orange = [b for b in orange_buoys if b[2] < 10.0]

        if len(nearby_orange) >= 2:
            xs = [b[0] for b in nearby_orange]
            lx, rx = min(xs), max(xs)
            lane_w = rx - lx
            min_w  = image_width * 0.1
            max_w  = image_width * 0.8
            if min_w < lane_w < max_w:
                cx = (lx + rx) / 2.0
                # Waypoint yönü ile koridor yönü aynı tarafta mı?
                if (heading_error * (cx - image_center_x) >= 0) or abs(heading_error) < 0.1:
                    lane_center_x = cx
                    lane_offset   = float(np.clip(
                        (image_center_x - cx) / (lane_w / 2.0), -1.0, 1.0
                    ))

        # ── ÖNCELİK 0: Turuncu dubaya çok yakın (acil kaçış) ─────────
        close_orange = [b for b in orange_buoys
                        if b[2] < 5.0 and abs(b[0] - image_center_x) < image_width * 0.8]
        if close_orange:
            closest = min(close_orange, key=lambda b: b[2])
            return (0.0, -1.8) if closest[0] < image_center_x else (0.0, 1.8)

        # ── ÖNCELİK 1: Sarı duba (engel) kaçınma ─────────────────────
        close_yellow = [b for b in yellow_buoys
                        if b[2] < 12.0 and abs(b[0] - image_center_x) < image_width * 0.7]
        if close_yellow:
            closest             = min(close_yellow, key=lambda b: b[2])
            obs_x, _, obs_dist  = closest
            dynamic_speed = max(0.0, (obs_dist - 5.0) * 0.3)
            turn_power    = 3.5 if obs_dist <= 5.0 else (2.5 if obs_dist <= 8.0 else 1.5)
            if obs_x < image_center_x:
                return (dynamic_speed, -turn_power) if lane_offset < 0.8  else (0.5, -0.2)
            else:
                return (dynamic_speed,  turn_power) if lane_offset > -0.8 else (0.5,  0.2)

        # ── ÖNCELİK 2: Şerit koruma ───────────────────────────────────
        if lane_center_x is not None and abs(lane_offset) > 0.4:
            return 1.2, float(np.clip(lane_offset * 1.6, -1.5, 1.5))

        # ── ÖNCELİK 3: Occupancy grid navigasyonu ─────────────────────
        grid_h, grid_w = occupancy_grid.shape
        cc = grid_w // 2
        front_view = occupancy_grid[max(0, grid_h - 100): grid_h, cc - 10: cc + 10]

        if np.any(front_view > 128):
            left_sum  = np.sum(occupancy_grid[grid_h - 100: grid_h, 0: cc])
            right_sum = np.sum(occupancy_grid[grid_h - 100: grid_h, cc:])
            bias      = (grid_h * grid_w * 255) * 0.2
            ls = left_sum  + (bias if heading_error <= 0 else 0)
            rs = right_sum + (bias if heading_error >  0 else 0)
            if ls < rs:
                return (2.0, 0.5)  if lane_offset > -0.7 else (1.5, 0.2)
            else:
                return (2.0, -0.5) if lane_offset <  0.7 else (1.5, -0.2)

        # ── ÖNCELİK 4: Düz hedef takibi ──────────────────────────────
        k_p         = 0.8
        angular_vel = float(np.clip(k_p * heading_error, -1.0, 1.0))
        if lane_center_x is not None:
            angular_vel = float(np.clip(angular_vel + lane_offset * 0.4, -1.0, 1.0))
            if abs(lane_offset) > 0.3:
                return 2.5, angular_vel
        return 5.0, angular_vel


# ============================================================
# Yardımcı: waypoint yön oku overlay
# ============================================================

def _build_wp_overlay(rgb_bgra: np.ndarray,
                      goal_x: float, goal_y: float,
                      dist_m: float) -> np.ndarray:
    """
    RGB görüntüsü üzerine waypoint yön okunu çizer.

    DÜZELTME: Eski kod bu overlay'i hiç çizmiyordu; görüntüdeki sarı çizgi
    test kodundan kalıyordu ve heading bilgisi kullanılmıyordu.
    Artık gerçek body-frame (goal_x, goal_y) ile doğru yön hesaplanır.

    Args:
        rgb_bgra : ZED'den gelen BGRA veya BGR görüntü
        goal_x   : Hedefe body-frame sağ ekseni (metre, + sağ)
        goal_y   : Hedefe body-frame ileri ekseni (metre, + ileri)
        dist_m   : Hedefe olan düz mesafe (etiket için)

    Returns:
        BGR kopyası — çizgiler eklenmiş
    """
    if rgb_bgra is None:
        return rgb_bgra

    if rgb_bgra.shape[2] == 4:
        img = cv2.cvtColor(rgb_bgra, cv2.COLOR_BGRA2BGR)
    else:
        img = rgb_bgra.copy()

    h, w = img.shape[:2]
    # Çizginin başlangıç noktası: görüntü alt-ortası (araç konumu)
    cx, cy = w // 2, int(h * 0.82)

    # Yön açısı: goal_x=sağ(+), goal_y=ileri(+)
    # Görüntü koordinatlarında: sağ=+x, yukarı=-y
    # atan2(goal_x, goal_y): 0=düz ileri, sağa pozitif
    angle_rad = math.atan2(goal_x, goal_y)   # body-frame açısı

    arrow_len = min(h, w) * 0.30   # ok uzunluğu piksel
    tip_x = int(cx + arrow_len * math.sin(angle_rad))
    tip_y = int(cy - arrow_len * math.cos(angle_rad))   # y ekranı ters

    # Araç başlangıç noktası (kırmızı dolu daire)
    cv2.circle(img, (cx, cy), 6, (0, 0, 220), -1)
    cv2.circle(img, (cx, cy), 6, (255, 255, 255), 1)

    # Yön oku (sarı)
    cv2.arrowedLine(img, (cx, cy), (tip_x, tip_y),
                    (0, 220, 220), 2, tipLength=0.18, line_type=cv2.LINE_AA)

    # Mesafe etiketi
    label = f"WP {dist_m:.1f}m"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    lx = max(0, min(tip_x - tw // 2, w - tw - 4))
    ly = max(th + 4, tip_y - 8)
    cv2.rectangle(img, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), (20, 20, 20), -1)
    cv2.putText(img, label, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 220), 1, cv2.LINE_AA)

    return img


# ============================================================
# Yardımcı: goal_local_coords hesabı
# ============================================================

def compute_goal_body_frame(curr_lat: float, curr_lon: float,
                             sp_lat: float, sp_lon: float,
                             heading_deg: float) -> tuple:
    """
    Hedef waypoint'in araç body-frame koordinatlarını döner.

    DÜZELTME: Eski kod (0.0, dist) kullanıyordu → hedef her zaman
    düz ileride varsayılıyordu. Araç yan açıda olunca hem LocalPlanner
    hem de overlay yanlış yön veriyordu.

    NED → body-frame dönüşümü:
      psi = heading_deg (0=Kuzey, CW pozitif)
      goal_x (sağ)  = -d_north * sin(psi) + d_east * cos(psi)
      goal_y (ileri) =  d_north * cos(psi) + d_east * sin(psi)

    Returns:
        (goal_x, goal_y) metre — body-frame (sağ, ileri)
    """
    if curr_lat == 0.0 or curr_lon == 0.0:
        return 0.0, 1.0   # GPS yoksa düz ileri

    cos_lat  = math.cos(math.radians(curr_lat))
    d_north  = (sp_lat - curr_lat) * 111320.0
    d_east   = (sp_lon - curr_lon) * 111320.0 * cos_lat

    psi      = math.radians(heading_deg)
    goal_x   = -d_north * math.sin(psi) + d_east * math.cos(psi)   # sağ
    goal_y   =  d_north * math.cos(psi) + d_east * math.sin(psi)   # ileri

    return goal_x, goal_y


# ============================================================
# MAVLink yardımcıları
# ============================================================

def get_boot_time_ms() -> int:
    return int((time.time() - START_TIME) * 1000)


script_set_mode = None
last_mode_change_time = 0.0

def set_mode(master, mode_name: str) -> bool:
    global script_set_mode, last_mode_change_time
    mode_map = master.mode_mapping()
    if mode_name not in mode_map:
        print(f"  ✗ [{mode_name}] modu bulunamadı!")
        return False
    master.set_mode(mode_map[mode_name])
    time.sleep(0.3)
    print(f"  ✓ Mod: {mode_name}")
    script_set_mode = mode_name
    last_mode_change_time = time.time()
    return True


def arm_vehicle(master, arm: bool):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1 if arm else 0, 0, 0, 0, 0, 0, 0
    )
    print(f"  → {'ARM' if arm else 'DISARM'} komutu gönderildi.")


def send_global_setpoint(master, lat: float, lon: float):
    """GUIDED modda GPS pozisyon hedefi gönder."""
    mask = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
    )
    master.mav.set_position_target_global_int_send(
        get_boot_time_ms(),
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_INT,
        mask,
        int(lat * 1e7), int(lon * 1e7), 0,
        0, 0, 0, 0, 0, 0, 0, 0
    )


def send_stop(master):
    """GUIDED modda dur (vx=0, yaw_rate=0)."""
    send_guided_velocity(master, 0.0, 0.0)


def send_guided_velocity(master, speed_ms: float, yaw_rate_rad: float):
    """
    GUIDED modda ileri hız + dönüş hızı (yaw rate) komutu gönderir.
    MOD DEĞİŞİMİ YAPMAZ — araç GUIDED'dan hiç çıkmadan pivot/dur/kaçınma uygular.

    SET_POSITION_TARGET_LOCAL_NED, body-relative çerçevede sadece
    vx (ileri hız) ve yaw_rate alanlarını kullanır; konum/ivme/vy/vz/yaw
    (mutlak) yok sayılır. ArduPilot Rover GUIDED bunu
    "set_desired_speed_and_turn_rate" olarak yorumlar.

    Args:
        speed_ms     : ileri hız (m/s). Pivot dönüş için 0.0.
        yaw_rate_rad : dönüş hızı (rad/s). ArduPilot/NED konvansiyonunda
                       POZİTİF = SAAT YÖNÜNDE (sağa), NEGATİF = sola.
                       ⚠️ Bu işaret kuralını suya çıkmadan önce sahada/
                       masa üstünde (motorsuz, sadece log/RC ile) DOĞRULA —
                       yanlışsa araç ters yöne pivot döner.
    """
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


def _yaw_rate_for_direction(direction: str) -> float:
    """turn_dir ('left'/'right') → GUIDED yaw_rate (rad/s).
    ArduPilot NED konvansiyonu: + = sağa (saat yönü), - = sola.
    (avoidance.py'deki MANUAL_CONTROL r işareti bunun TERSİYDİ — dikkat.)
    """
    return -GUIDED_AVOID_YAW_RATE if direction == 'left' else GUIDED_AVOID_YAW_RATE


def send_guided_speed(master, speed_ms: float):
    """
    GUIDED modda araç hız limitini değiştirir (DO_CHANGE_SPEED).
    Proaktif fren: engel yaklaştıkça hızı kıs, STOP ulaşmadan önce
    araç zaten yavaşlamış olsun.

    speed_ms: 0=dur, -1=varsayılana dön (conf.py WP_SPEED)
    """
    print(f"\n  [HIZ] Hedef hız limiti güncelleniyor: {speed_ms:.2f} m/s")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
        0,
        1,          # speed_type=1: ground speed
        float(max(0.0, speed_ms)),
        -1,         # throttle: -1=değiştirme
        0, 0, 0, 0
    )


def get_distance_meters(clat: float, clon: float,
                        tlat: float, tlon: float) -> float:
    return math.sqrt(
        ((clat - tlat) * 111320.0) ** 2 +
        ((clon - tlon) *  85390.0) ** 2
    )


# ============================================================
# Navigasyon setpoint fonksiyonları
# ============================================================

def corridor_corrected_setpoint(sp_lat, sp_lon, orange_buoys, heading_deg):
    """
    Turuncu duba çifti tespit edilince GPS setpoint'ini koridor merkezine
    doğru yanal olarak kaydırır. (≥2 duba gerekli)
    """
    if len(orange_buoys) < 2:
        return sp_lat, sp_lon, 0.0

    xs            = [b[0] for b in orange_buoys]
    corridor_w_px = max(xs) - min(xs)

    if corridor_w_px < IMAGE_WIDTH * CORRIDOR_MIN_WIDTH:
        return sp_lat, sp_lon, 0.0

    corridor_cx = (min(xs) + max(xs)) / 2.0
    image_cx    = IMAGE_WIDTH / 2.0
    error_px    = corridor_cx - image_cx   # + = koridor sağda

    lane_offset = float(np.clip(error_px / (corridor_w_px / 2.0), -1.0, 1.0))

    if abs(lane_offset) < 0.08:
        return sp_lat, sp_lon, lane_offset

    lateral_m = lane_offset * CORRIDOR_CORRECT_M
    psi       = math.radians(heading_deg)
    north     = lateral_m * (-math.sin(psi))
    east      = lateral_m *   math.cos(psi)

    cos_lat       = math.cos(math.radians(sp_lat))
    corrected_lat = sp_lat + north / 111320.0
    corrected_lon = sp_lon + east  / (111320.0 * cos_lat)
    return corrected_lat, corrected_lon, lane_offset


def corridor_carrot_setpoint(curr_lat, curr_lon, orange_buoys, heading_deg,
                              lookahead_m=6.0):
    """
    Turuncu dubalara göre koridor merkezini bulur ve lookahead_m ileri
    yanal kaydırılmış carrot setpoint döner.
    """
    if not orange_buoys:
        return None, None, 0.0

    image_cx    = IMAGE_WIDTH / 2.0
    buoys_with_y = []
    for cx_px, _, dist_m in orange_buoys:
        angle_deg = (cx_px - image_cx) / image_cx * 45.0   # ZED ~90° efektif FOV
        y_m       = dist_m * math.sin(math.radians(angle_deg))
        buoys_with_y.append((cx_px, dist_m, y_m))

    left_buoys  = [b for b in buoys_with_y if b[2] < 0]
    right_buoys = [b for b in buoys_with_y if b[2] >= 0]

    safe_wall = 2.2   # metre: dubadan ideal mesafe

    if left_buoys and right_buoys:
        y_l = min(left_buoys,  key=lambda b: b[1])[2]
        y_r = min(right_buoys, key=lambda b: b[1])[2]
        lateral_m = (y_l + y_r) / 2.0
    elif left_buoys:
        y_l       = min(left_buoys,  key=lambda b: b[1])[2]
        lateral_m = safe_wall + y_l    # y_l negatif → sağa çek
    elif right_buoys:
        y_r       = min(right_buoys, key=lambda b: b[1])[2]
        lateral_m = -safe_wall + y_r   # y_r pozitif → sola çek
    else:
        return None, None, 0.0

    lateral_m   = float(np.clip(lateral_m, -2.5, 2.5))
    lane_offset = float(np.clip(lateral_m / safe_wall, -1.0, 1.0))

    psi   = math.radians(heading_deg)
    north = lookahead_m * math.cos(psi) + lateral_m * (-math.sin(psi))
    east  = lookahead_m * math.sin(psi) + lateral_m *   math.cos(psi)

    cos_lat       = math.cos(math.radians(curr_lat))
    corrected_lat = curr_lat + north / 111320.0
    corrected_lon = curr_lon + east  / (111320.0 * cos_lat)
    return corrected_lat, corrected_lon, lane_offset


def obstacle_avoidance_carrot_setpoint(curr_lat, curr_lon, obs_direction, heading_deg,
                                        lookahead_m=6.0):
    """Orta mesafeli engelden kaçmak için carrot setpoint üretir."""
    lateral_m = 2.0 if obs_direction in ('left', 'center') else -2.0

    psi   = math.radians(heading_deg)
    north = lookahead_m * math.cos(psi) + lateral_m * (-math.sin(psi))
    east  = lookahead_m * math.sin(psi) + lateral_m *   math.cos(psi)

    cos_lat       = math.cos(math.radians(curr_lat))
    corrected_lat = curr_lat + north / 111320.0
    corrected_lon = curr_lon + east  / (111320.0 * cos_lat)
    return corrected_lat, corrected_lon


def obstacle_avoidance_setpoint(sp_lat, sp_lon, obs_direction, heading_deg):
    """GPS'siz fallback: waypoint'i yanal kaydırır."""
    lateral_m = 1.5 if obs_direction in ('left', 'center') else -1.5

    psi   = math.radians(heading_deg)
    north = lateral_m * (-math.sin(psi))
    east  = lateral_m *   math.cos(psi)

    cos_lat       = math.cos(math.radians(sp_lat))
    corrected_lat = sp_lat + north / 111320.0
    corrected_lon = sp_lon + east  / (111320.0 * cos_lat)
    return corrected_lat, corrected_lon


def fuse_obstacles(depth_info, yellow_buoys):
    """
    Derinlik kamerası + sarı dubaları birleştirir.

    Karar mantığı:
      - Merkez mesafesi her zaman dahil edilir (düz çarpma riski).
      - Yan mesafeler yalnızca OBSTACLE_NEAR_LIMIT'in yarısından küçükse
        tehlikeli sayılır (koridor kenarı false-positive'lerini engeller).
      - Sarı dubalar piksel X konumuna göre sola/sağa/merkeze atanır.
      - En yakın tehlike kaynağı ve yönü döndürülür.

    Returns: (min_dist_m, direction)  direction ∈ {'left','center','right','none'}
    """
    dists = []

    if depth_info:
        dc = depth_info['dist_center']
        dl = depth_info['dist_left']
        dr = depth_info['dist_right']

        # Merkez her zaman tehlike kaynağı
        dists.append((dc, 'center'))

        # Yan mesafeler: sadece merkez mesafesinden de yakınsa veya
        # OBSTACLE_NEAR_LIMIT yarısından küçükse tehlikeli say.
        # Bu sayede koridor kenarı (duba) köpük false-positive'leri engellenir.
        near_half = OBSTACLE_NEAR_LIMIT * 0.6
        if dl < near_half or dl < dc:
            dists.append((dl, 'left'))
        if dr < near_half or dr < dc:
            dists.append((dr, 'right'))

    if yellow_buoys:
        for cx_px, _, dist_m in yellow_buoys:
            if   cx_px < IMAGE_WIDTH * 0.33: dists.append((dist_m, 'left'))
            elif cx_px > IMAGE_WIDTH * 0.67: dists.append((dist_m, 'right'))
            else:                             dists.append((dist_m, 'center'))

    if not dists:
        return float('inf'), 'none'

    return min(dists, key=lambda x: x[0])


# ============================================================
# Ana Program
# ============================================================

print("=" * 55)
print("  USV Navigasyon + ZED Engel Kaçınma (Pivot / State Machine)")
print(f"  Bağlantı: {CONNECTION_STRING}")
print("=" * 55)

camera   = None
recorder = None

try:
    # ---- MAVLink bağlantısı ----
    master = mavutil.mavlink_connection(CONNECTION_STRING)
    print("Heartbeat bekleniyor...")
    master.wait_heartbeat()
    print(f"✓ Pixhawk bağlandı! Sistem ID: {master.target_system}")

    # ---- Modülleri başlat ----
    print("\n[INIT] ZED Kamera ve modüller başlatılıyor...")
    camera         = ZEDCamera()
    detector       = ObstacleDetector(threshold=OBSTACLE_NEAR_LIMIT)
    avoider        = AvoidanceController()
    local_planner  = LocalPlanner()
    global_planner = GlobalPlanner()

    camera.start()
    time.sleep(1.0)

    # ---- Video kaydını program başında tek seferlik başlat ----
    recorder_start_time = None
    try:
        recorder = ScreenRecorder()
        recorder.start()
        recorder_start_time = time.time()
    except Exception as e:
        print(f"  ⚠️ Video kaydı başlatılamadı: {e}")
        recorder = None

    # Görev durum değişkenleri
    mission_active = False
    grid_map       = None
    mission_map    = None
    mission_ctrl   = None

    curr_lat    = curr_lon = 0.0
    dist        = 999.0
    speed       = 0.0
    heading_deg = 0.0
    info        = None
    lin_vel     = 0.0
    ang_vel     = 0.0
    goal_x      = 0.0
    goal_y      = 1.0
    mc_state    = MissionState.IDLE
    buoy_detections = {'orange': [], 'yellow': [], 'black': [], 'red': [], 'green': []}

    gps_fix         = 0
    satellites      = 0
    servo1          = 1500
    servo3          = 1500
    is_armed        = False

    print("\n[HAZIR] Pixhawk'tan GUIDED modu bekleniyor...")

    current_mode = None
    waiting_message_printed = False

    while True:
        # ── Mod Bilgisi Al ──────────────────────────────────────────
        msg = master.recv_match(type='HEARTBEAT', blocking=False)
        if msg and msg.get_srcSystem() == master.target_system:
            custom_mode  = msg.custom_mode
            mode_map     = master.mode_mapping()
            prev_mode    = current_mode
            for name, mode_id in mode_map.items():
                if mode_id == custom_mode:
                    current_mode = name
                    break
            # Mod değiştiyse hemen bildir
            if current_mode != prev_mode and current_mode != "GUIDED":
                print(f"[MOD DEĞİŞTİ] {prev_mode} → {current_mode} | GUIDED bekleniyor...")
                waiting_message_printed = False

        if current_mode != "GUIDED":
            if not waiting_message_printed:
                print(f"[BEKLEMEDE] Mevcut Mod: {current_mode} | Otonom görev için Pixhawk'tan GUIDED modu bekleniyor...")
                waiting_message_printed = True
            time.sleep(0.5)
            continue

        # GUIDED mod algılandı!
        waiting_message_printed = False

        if not mission_active:
            print("\n" + "=" * 55)
            print("  [BAŞLADI] Yeni Görev Başlatılıyor...")
            print("=" * 55)
            global_planner.current_wp_index = 0
            grid_map     = Map2D()
            mission_map  = MissionMap(waypoints=GPS_WAYPOINTS)
            mission_ctrl = MissionController()
            mission_active = True
        else:
            print("\n" + "=" * 55)
            print("  [DEVAM] Görev Kaldığı Yerden Devam Ediyor...")
            print("=" * 55)

        script_set_mode = "GUIDED"
        last_mode_change_time = time.time()

        # GUIDED modunu aktif olarak teyit et (sadece algılayıp güvenmek yetmez)
        print("  → GUIDED modu aktif olarak gönderiliyor...")
        set_mode(master, "GUIDED")
        time.sleep(0.5)

        # ARM et
        arm_vehicle(master, True)
        time.sleep(1.0)

        # ARM sonrası mod kaymasına karşı tekrar GUIDED teyidi
        set_mode(master, "GUIDED")
        time.sleep(0.3)

        # Waypoint Döngüsü
        user_override            = False
        _guided_recovery_count   = 0       # Kaç kez GUIDED'a geri çekildik
        _guided_lost_since       = None    # GUIDED kaybedildiğinde bunu kaydet

        while global_planner.current_wp_index < len(GPS_WAYPOINTS):
            wp_idx = global_planner.current_wp_index
            sp_lat, sp_lon = GPS_WAYPOINTS[wp_idx]
            print(f"\n[HEDEF {wp_idx + 1}] → ({sp_lat:.7f}, {sp_lon:.7f})")
            send_global_setpoint(master, sp_lat, sp_lon)

            state              = STATE_NAVIGATE
            state_since        = time.time()
            last_sp_send       = 0.0   # Tek unified setpoint timer — tüm katmanlar kullanır
            turn_dir           = 'none'
            brake_active       = False  # Proaktif fren durumu takibi
            last_sent_speed    = 1.5    # Son gönderilen hız limiti (varsayılan 1.5 m/s)
            last_global_wp_send = 0.0   # Açık alanda setpoint spamini önlemek için timer
            SP_INTERVAL        = 0.50   # 2 Hz setpoint gönderme frekansı (Pixhawk'ın yavaşlama/hızlanma rampaları için idealdir)
            aligning_to_wp     = False  # Sadece rotaya hizalanma dönüşü mü yapıyoruz?

            while True:
                now = time.time()

                # ── MAVLink Telemetri ve Mod Alımı ─────────────────────────
                while True:
                    m = master.recv_match(blocking=False)
                    if not m:
                        break
                    
                    if m.get_type() == 'GLOBAL_POSITION_INT':
                        curr_lat    = m.lat / 1e7
                        curr_lon    = m.lon / 1e7
                        speed       = math.sqrt(m.vx**2 + m.vy**2) / 100.0
                        dist        = get_distance_meters(curr_lat, curr_lon, sp_lat, sp_lon)
                        if m.hdg != 65535:
                            heading_deg = m.hdg / 100.0
                            
                    elif m.get_type() == 'HEARTBEAT' and m.get_srcSystem() == master.target_system:
                        custom_mode = m.custom_mode
                        mode_map = master.mode_mapping()
                        for name, mode_id in mode_map.items():
                            if mode_id == custom_mode:
                                current_mode = name
                                break
                        is_armed = (m.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                    elif m.get_type() == 'GPS_RAW_INT':
                        gps_fix     = m.fix_type
                        satellites  = m.satellites_visible
                    elif m.get_type() == 'SERVO_OUTPUT_RAW':
                        servo1      = m.servo1_raw
                        servo3      = m.servo3_raw
                    elif m.get_type() == 'COMMAND_ACK':
                        print(f"\n[ACK] Komut: {m.command} | Sonuç: {m.result} (0=ACCEPTED, 4=FAILED)")

                # ── Mod Koruyucu: GUIDED dışına çıkılırsa anında kontrolü kullanıcıya bırak ──
                if current_mode is not None and current_mode != "GUIDED":
                    print(f"\n[!] UYARI: Dışarıdan [{current_mode}] modu algılandı! Otonom görev durduruluyor, kontrol kullanıcıya bırakıldı.")
                    user_override = True
                    break

                # ── Hedef yönü (body-frame) ───────────────────────────────
                goal_x, goal_y = compute_goal_body_frame(
                    curr_lat, curr_lon, sp_lat, sp_lon, heading_deg
                )
                heading_error = math.degrees(math.atan2(goal_x, goal_y))

                # ── ZED Frame + Engel Analizi ─────────────────────────────
                rgb, depth = camera.get_frame()
                if rgb is not None and depth is not None:
                    info = detector.analyze(depth)
                    buoy_detections = camera.detect_colored_buoys(rgb, depth)
                    rgb_with_overlay = _build_wp_overlay(rgb, goal_x, goal_y, dist)

                    if recorder:
                        recorder.update(rgb_with_overlay, depth, info,
                                        buoy_detections=buoy_detections,
                                        vehicle_pos=(curr_lat, curr_lon),
                                        grid_map=grid_map)

                    camera.get_vio()   # iç state güncellenir
                    grid_map.update(depth, curr_lat, curr_lon, heading_deg, buoy_detections=buoy_detections)
                    mission_map.log(curr_lat, curr_lon, heading_deg, info)

                obstacle  = info['obstacle']   if info else False
                direction = info['direction']  if info else 'none'

                # ── GlobalPlanner ────────────────────────────────────────
                global_planner.check_waypoint_reached(dist)

                # ── LocalPlanner ─────────────────────────────────────────
                occupancy_grid, *_ = grid_map.get_grid()
                lin_vel, ang_vel   = local_planner.compute_velocity_command(
                    occupancy_grid,
                    (goal_x, goal_y),
                    buoy_detections=buoy_detections,
                )

                # ── MissionController ─────────────────────────────────────
                is_docking_wp = (global_planner.current_wp_index + 1) >= DOCKING_WAYPOINT_NUM
                mc_state = mission_ctrl.update(
                    waypoint_reached  = (dist < SAFETY_DISTANCE and state == STATE_NAVIGATE),
                    obstacle_detected = obstacle,
                    target_detected   = is_docking_wp and bool(
                        buoy_detections.get('black') or
                        buoy_detections.get('red')   or
                        buoy_detections.get('green')
                    ),
                )

                # ── Koridor Tespiti ───────────────────────────────────────
                orange_all = buoy_detections.get('orange', [])
                orange_nearby = [b for b in orange_all if b[2] < ORANGE_ALIGN_DIST]
                in_corridor     = len(orange_nearby) >= 1
                orange_min_dist = min((b[2] for b in orange_nearby), default=float('inf'))

                # ============================================================
                # DURUM MAKİNESİ
                # ============================================================

                if state == STATE_NAVIGATE:
                    # ── Engel füzyonu ─────────────────────────────────────────
                    obs_dist_near, obs_dir_near = fuse_obstacles(info, buoy_detections.get('yellow', []))
                    if in_corridor:
                        obs_dist_med, obs_dir_med = fuse_obstacles(None, buoy_detections.get('yellow', []))
                    else:
                        obs_dist_med, obs_dir_med = obs_dist_near, obs_dir_near

                    # ── PROAKTİF HIZ AZALTMA ─────────────────────────────────
                    brake_dist = min(obs_dist_near, obs_dist_med, orange_min_dist)
                    if brake_dist < OBSTACLE_MED_LIMIT:
                        if brake_dist >= 2.5:
                            t          = (brake_dist - 2.5) / (OBSTACLE_MED_LIMIT - 2.5)
                            target_spd = 0.6 + t * 0.9  # 0.6 ile 1.5 m/s arasında ölçekle
                        else:
                            target_spd = 0.6            # En düşük güvenli hız (dümen hakimiyeti için)
                        
                        # Hız komutunu ilk defa veya hız belirgin şekilde değiştiğinde (en az 0.1 m/s fark) gönder
                        if not brake_active or abs(target_spd - last_sent_speed) > 0.1:
                            send_guided_speed(master, target_spd)
                            last_sent_speed = target_spd
                            brake_active = True
                    else:
                        if brake_active:
                            send_guided_speed(master, 1.5)  # Pixhawk'ın kabul edeceği makul cruise hızı (5.0 çok yüksek)
                            last_sent_speed = 1.5
                            brake_active = False

                    # ── SETPOINT GÖNDER (2 Hz unified timer) ─────────────────
                    if now - last_sp_send >= SP_INTERVAL:
                        orange_too_close = [b for b in orange_nearby if b[2] < ORANGE_ESCAPE_DIST]

                        if orange_too_close:
                            closest_orange = min(orange_too_close, key=lambda b: b[2])
                            cx_px          = closest_orange[0]
                            od             = closest_orange[2]

                            if od < ORANGE_PANIC_DIST:
                                turn_dir = 'right' if cx_px < IMAGE_WIDTH / 2.0 else 'left'
                                print(f"\n  🚨 [TURUNCU PANİK] d={od:.1f}m → STOP/PIVOT {turn_dir}")
                                state       = STATE_STOP
                                state_since = now
                            else:
                                escape_dir = 'right' if cx_px < IMAGE_WIDTH / 2.0 else 'left'
                                if curr_lat != 0.0:
                                    c_lat, c_lon = obstacle_avoidance_carrot_setpoint(
                                        curr_lat, curr_lon, escape_dir, heading_deg, lookahead_m=3.0)
                                else:
                                    c_lat, c_lon = obstacle_avoidance_setpoint(
                                        sp_lat, sp_lon, escape_dir, heading_deg)
                                send_global_setpoint(master, c_lat, c_lon)
                                last_sp_send = now
                                print(f"\n  🟠⚡[TURUNCU YAKIN] d={od:.1f}m px={cx_px} → {escape_dir}", end="\r")

                        elif (obs_dist_near < OBSTACLE_NEAR_LIMIT or
                              (info and info.get('emergency', False))):
                            is_emg   = info.get('emergency', False) if info else False
                            # Engel merkezdeyse veya acil durumsa waypoint yönüne dön (heading_error'u azaltacak şekilde)
                            if obs_dir_near == 'center' or is_emg:
                                turn_dir = 'right' if heading_error > 0 else 'left'
                            else:
                                # Engel sol veya sağda ise zıt yöne dön
                                turn_dir = 'right' if obs_dir_near == 'left' else 'left'
                            
                            print(f"\n  🚨 [NAVIGATE→STOP] d={obs_dist_near:.1f}m yön:{obs_dir_near} kaçış:{turn_dir}")
                            aligning_to_wp = False
                            state       = STATE_STOP
                            state_since = now

                        elif OBSTACLE_NEAR_LIMIT <= obs_dist_med < OBSTACLE_MED_LIMIT:
                            if curr_lat != 0.0:
                                c_lat, c_lon = obstacle_avoidance_carrot_setpoint(
                                    curr_lat, curr_lon, obs_dir_med, heading_deg)
                            else:
                                c_lat, c_lon = obstacle_avoidance_setpoint(
                                    sp_lat, sp_lon, obs_dir_med, heading_deg)
                            send_global_setpoint(master, c_lat, c_lon)
                            last_sp_send = now
                            print(f"\n  ⚠️ [ORTA ENGEL] {obs_dir_med} zıttı | d={obs_dist_med:.1f}m", end="\r")

                        elif in_corridor:
                            if curr_lat != 0.0:
                                c_lat, c_lon, lane_off = corridor_carrot_setpoint(
                                    curr_lat, curr_lon, orange_nearby, heading_deg)
                            else:
                                c_lat, c_lon, lane_off = corridor_corrected_setpoint(
                                    sp_lat, sp_lon, orange_nearby, heading_deg)
                            if c_lat is not None:
                                send_global_setpoint(master, c_lat, c_lon)
                                last_sp_send = now
                                if abs(lane_off) >= 0.08:
                                    side = "→ sağa" if lane_off > 0 else "← sola"
                                    print(f"\n  🟠 [KORIDOR] {side} off={lane_off:+.2f} n={len(orange_nearby)}", end="\r")

                        elif not in_corridor and abs(heading_error) > 45.0 and dist > 2.0 and curr_lat != 0.0:
                            turn_dir = 'right' if heading_error > 0 else 'left'
                            print(f"\n  🔄 [ROTADAN SAPMA] Açı Sapması: {heading_error:.1f}° > 45° | Hizalanmak için duruluyor... Yön: {turn_dir}")
                            aligning_to_wp = True
                            state       = STATE_STOP
                            state_since = now

                        else:
                            if now - last_global_wp_send >= 2.0:
                                send_global_setpoint(master, sp_lat, sp_lon)
                                last_global_wp_send = now
                            last_sp_send = now

                elif state == STATE_STOP:
                    # GUIDED'da kal — sadece hız/dönüşü sıfırla (mod değişimi YOK)
                    send_guided_velocity(master, 0.0, 0.0)
                    if now - state_since >= 0.5:
                        print(f"  🔄 [STOP→TURN] {'← Sol' if turn_dir == 'left' else '→ Sağ'} | Kalan: {dist:.1f}m")
                        state       = STATE_TURN
                        state_since = now

                elif state == STATE_TURN:
                    # GUIDED'da kal — otopilotun motorları aktif kontrol edebilmesi için hafif ileri hız (0.4 m/s) verilir.
                    # (0.0 olunca otopilot dönmek yerine düz kayıyor / tepkisiz kalıyor)
                    turn_speed = 0.4
                    send_guided_velocity(master, turn_speed, _yaw_rate_for_direction(turn_dir))
                    
                    time_limit = (now - state_since) >= 5.0
                    
                    if aligning_to_wp:
                        # Rota hizalanma dönüşü: 5 derece toleransla dur
                        turn_finished = (abs(heading_error) < 5.0)
                    else:
                        # Engelden kaçınma dönüşü: Önümüz temiz olmalı ve hedefe yönlenene kadar dönmeliyiz
                        center_clear = (not info) or (info['dist_center'] >= OBSTACLE_NEAR_LIMIT)
                        if center_clear:
                            # Önümüz temizse, hedefe hizalanmaya devam et (15 derece toleransla)
                            turn_finished = (abs(heading_error) < 15.0)
                        else:
                            # Önümüz kapalıysa dönüşü kesme, dönmeye devam et
                            turn_finished = False
                            
                    if turn_finished or time_limit:
                        reason = (
                            "Hizalandı" if aligning_to_wp else 
                            ("Merkez temiz ve hizalandı" if turn_finished else "Maks. dönüş süresi")
                        )
                        print(f"  🛑 [TURN→STABILIZE] {reason} | Açı Sapması: {heading_error:.1f}° | Kalan: {dist:.1f}m")
                        state       = STATE_STABILIZE
                        state_since = now

                elif state == STATE_STABILIZE:
                    # GUIDED'da kal — hız/dönüşü sıfırla, sabitlen
                    send_guided_velocity(master, 0.0, 0.0)
                    if now - state_since >= 0.24:
                        print(f"  ✅ [STABILIZE→COOLDOWN] Rotaya devam ediyorum (GUIDED'dan hiç çıkmadım)")
                        send_global_setpoint(master, sp_lat, sp_lon)
                        send_guided_speed(master, 1.5)  # Seyir hızına geri dön (1.5 m/s)
                        last_sp_send = now
                        avoider.start_cooldown()
                        aligning_to_wp = False  # Hizalanma bitti
                        state       = STATE_COOLDOWN
                        state_since = now

                elif state == STATE_COOLDOWN:
                    if now - last_sp_send >= SP_INTERVAL:
                        send_global_setpoint(master, sp_lat, sp_lon)
                        last_sp_send = now

                    cd_near, cd_dir = fuse_obstacles(info, buoy_detections.get('yellow', []))
                    if cd_near < OBSTACLE_NEAR_LIMIT:
                        print(f"\n  🚨 [COOLDOWN→STOP] Cooldown'da engel yaklaştı! d={cd_near:.1f}m")
                        turn_dir    = 'left' if cd_dir in ('right', 'center') else 'right'
                        state       = STATE_STOP
                        state_since = now
                    elif avoider.cooldown_remaining() <= 0:
                        if obstacle:
                            print(f"\n  ⚠️ [COOLDOWN→STOP] Engel hâlâ var! | Kalan: {dist:.1f}m")
                            turn_dir    = 'left' if direction in ('right', 'center') else 'right'
                            state       = STATE_STOP
                            state_since = now
                        else:
                            print(f"  ✅ [COOLDOWN→NAVIGATE] Yol açık. | Kalan: {dist:.1f}m")
                            state       = STATE_NAVIGATE
                            state_since = now

                # ── Telemetri ─────────────────────────────────────────────
                if info:
                    engel_str = "  ⚠️ ENGEL" if obstacle else ""
                    n_or  = len(buoy_detections.get('orange', []))
                    n_ye  = len(buoy_detections.get('yellow', []))
                    n_tg  = (len(buoy_detections.get('black', [])) +
                             len(buoy_detections.get('red',   [])) +
                             len(buoy_detections.get('green', [])))
                    or_dist_str = f"{orange_min_dist:.1f}" if orange_min_dist < 99 else "--"
                    brk_str     = f"🔴{brake_dist:.1f}" if brake_active else "  "
                    
                    fix_names = {0: "No", 1: "No", 2: "2D", 3: "3D", 4: "DGPS", 5: "FloatRTK", 6: "FixedRTK"}
                    fix_str = fix_names.get(gps_fix, f"B({gps_fix})")
                    gps_status = f"{fix_str}({satellites}s)"
                    arm_str = "ARMED" if is_armed else "DISARM"
                    
                    print(
                        f"  [{state:9s}|{arm_str}|GPS:{gps_status}] "
                        f"WP:{dist:.1f}m Hız:{speed:.1f} | "
                        f"Sol:{info['dist_left']:.1f} Mrkz:{info['dist_center']:.1f} Sağ:{info['dist_right']:.1f} | "
                        f"PWM: S{servo1} T{servo3} | "
                        f"🟠{n_or}({or_dist_str}m) 🟡{n_ye}({brk_str}) | "
                        f"gx={goal_x:.1f} gy={goal_y:.1f} err={heading_error:+.1f}°"
                        + engel_str,
                        end="\r"
                    )

                # ── Waypoint varış ────────────────────────────────────────
                if dist < SAFETY_DISTANCE and state == STATE_NAVIGATE:
                    print(f"\n  ✓ Hedef {wp_idx + 1} tamamlandı! (Mesafe: {dist:.1f}m)")
                    break

                time.sleep(0.02)   # 50 Hz ana döngü

            if user_override:
                break

            print("  > 1s sonra devam...")
            time.sleep(1)

        if user_override:
            # Kullanıcı modu değiştirdiği için döngüden çıktık, tekrar mod beklemeye dönüyoruz
            continue

        # Tüm waypointler bittiğinde
        print("\n" + "=" * 55)
        print("✓ TÜM WAYPOINT'LER TAMAMLANDI!")
        print("=" * 55)
        send_guided_velocity(master, 0.0, 0.0)
        mission_active = False

        # Haritaları kaydet
        try:
            mission_map.save_final_png()
            grid_map.save_final_png()
        except Exception as e:
            print(f"  ⚠️ Görev haritaları kaydedilemedi: {e}")

except KeyboardInterrupt:
    print("\n\n[!] Durduruldu → GUIDED hız/dönüş sıfırlandı")
    if 'master' in locals():
        send_guided_velocity(master, 0.0, 0.0)

except Exception as e:
    import traceback
    print(f"\n[!] Hata: {e}")
    traceback.print_exc()
    if 'master' in locals():
        send_guided_velocity(master, 0.0, 0.0)

finally:
    if 'master' in locals():
        try:
            set_mode(master, "HOLD")
            time.sleep(0.3)
            arm_vehicle(master, False)
            time.sleep(0.5)
        except Exception as e:
            print(f"  ⚠️ Mod HOLD yapılamadı veya disarm edilemedi: {e}")
    try:
        if mission_map and grid_map:
            mission_map.save_final_png()
            grid_map.save_final_png()
    except Exception as e:
        print(f"  ⚠️ Görev haritası kaydedilemedi: {e}")
    if camera:
        try: camera.stop()
        except Exception: pass
    if recorder:
        try:
            elapsed = (time.time() - recorder_start_time) if recorder_start_time else 0
            print(f"  ℹ️  Toplam kayıt süresi: {elapsed:.1f}s — kapatılıyor...")
            recorder.stop()
        except Exception: pass
    print("Tüm modüller kapatıldı.")

