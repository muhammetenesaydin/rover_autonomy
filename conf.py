#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conf.py — Merkezi Konfigürasyon Dosyası
Tüm sistem parametreleri buradan yönetilir.
"""

import pyzed.sl as sl

# =============================================================
# MAVLINK / BAĞLANTI
# =============================================================
CONNECTION_STRING = 'udp:0.0.0.0:14550'   # Pixhawk bağlantı adresi

# =============================================================
# GPS WAYPOINT LİSTESİ  (Enlem, Boylam) — Decimal Degrees
# =============================================================
"""GPS_WAYPOINTS = [
    #jeoloji (38.6733831, 39.1864474),
     #(38.6732904, 39.1861705)   # Ondalık ayırıcı NOKTA olmalı (virgül değil!)
     (38.5197358, 39.4168968),
     (38.5197206, 39.4168632)
]"""
""" sivrice 
GPS_WAYPOINTS = [
    (38.5198108, 39.4172203), 
    (38.5198475, 39.4173427),
    (38.5198877, 39.4174570),  # Ondalık ayırıcı NOKTA olmalı (virgül değil!)
]"""
GPS_WAYPOINTS = [
    (38.6734844, 39.1867569), 
    (38.6735058, 39.1868011),
    (38.6735370, 39.1868608),  # Ondalık ayırıcı NOKTA olmalı (virgül değil!)
]


# =============================================================
# ROVER / NAVİGASYON
# =============================================================
SAFETY_DISTANCE            = 1.0    # Hedefe varış toleransı (metre)
SETPOINT_INTERVAL          = 0.2    # GPS setpoint yenileme periyodu (saniye) ← 1.0→0.2 (5Hz)
WAYPOINT_ACCEPTANCE_RADIUS = 1.0    # GlobalPlanner: waypoint kabul yarıçapı (metre, NED)
DOCKING_WAYPOINT_NUM       = 5      # Docking sadece bu waypoint numarasına gelince başlasın (1-indexed)

# =============================================================
# ZED KAMERA
# =============================================================
ZED_RESOLUTION = sl.RESOLUTION.HD720   # HD720 | HD1080 | VGA
ZED_FPS        = 30                    # Kamera FPS
ZED_DEPTH_MODE = sl.DEPTH_MODE.NEURAL_PLUS   # SDK 4.x: ULTRA deprecated, NEURAL_PLUS önerilir
ZED_MIN_DIST   = 0.2                   # Minimum derinlik ölçüm mesafesi (metre) — ZED en az 0.2m kısıtı var
MAP_MIN_DIST   = ZED_MIN_DIST          # Duba tespiti mesafe alt filtresi (detect_colored_buoys)
CAMERA_HEIGHT  = 0.25                  # Kameranın su seviyesinden yüksekliği (metre)
MIN_OBSTACLE_HEIGHT = 0.1              # Engel kabul edilmesi için su seviyesinden min. yükseklik (metre)

# =============================================================
# ENGEL ALGILAMA
# =============================================================
OBSTACLE_THRESHOLD  = 1.2   # Engel algılama tetik mesafesi (metre)
DEPTH_PERCENTILE    = 20    # Gürültü filtresi — %20'lik derinlik değeri
ROI_Y_RATIO         = (0.20, 0.65)   # Sadece dikey %10 ile %35 arasını tara

# Katmanlı Engel Sınırları (Füzyon için)
OBSTACLE_NEAR_LIMIT = 1.5   # Yakın Engel (Kesin çarpar - Acil Dur/Pivot Dön, metre)
OBSTACLE_MED_LIMIT  = 4.0   # Orta Mesafe Engel (Yolunu Değiştir/Yanal Kay, metre)

# Turuncu duba mesafe eşikleri
ORANGE_PANIC_DIST   = 1.0   # Bu altında → STOP/PIVOT (duvara çarpar)
ORANGE_ESCAPE_DIST  = 2.5   # Bu altında → acil kaçış carrot (hız düşür)
ORANGE_ALIGN_DIST   = 8.0   # Bu altında → koridor ortalama aktif


# Koridor x aralıkları (0.0–1.0 oranı)
LEFT_X_RATIO   = (0.00, 0.33)
CENTER_X_RATIO = (0.33, 0.67)
RIGHT_X_RATIO  = (0.67, 1.00)

# =============================================================
# KAÇINMA (ARC / YAY HAREKETİ)
# =============================================================
AVOIDANCE_THROTTLE = 300      # Yay kaçınma için ileri gaz
AVOIDANCE_STEERING = 700      # Yönlendirme kuvveti (r değeri) — sola (+700), sağa (-700)
GUIDED_AVOID_SPEED = 0.6    # GUIDED modda ileri hız (m/s)
GUIDED_AVOID_YAW_RATE = 0.5  # GUIDED modda dönüş hızı (rad/s)
AVOIDANCE_DURATION = 2.0    # Tek kaçınma adımı süresi (saniye)
AVOIDANCE_COOLDOWN = 2.0    # Aynı engeli tekrar algılamadan önce bekleme (saniye)

# =============================================================
# KORIDOR TAKİP (turuncu duba şerit merkezleme)
# =============================================================
CORRIDOR_MIN_DIST   = 0.8    # Duba min. mesafe — bu altı yoksay (çok yakın, m)
CORRIDOR_MAX_DIST   = 12.0   # Duba max. mesafe — bu ötesi yoksay (m)
CORRIDOR_CORRECT_M  = 1.0    # Setpoint yanal düzeltme kuvveti (metre)
CORRIDOR_INTERVAL   = 0.05   # Koridor setpoint gönderme periyodu (saniye) ← 0.2→0.05 (20Hz)
CORRIDOR_MIN_WIDTH  = 0.08   # Koridor geçerlilik: piksel genişliği en az %8
CORRIDOR_MIN_WIDTH  = 0.08   # Koridor geçerlilik: piksel genişliği en az %8 olmalı (0.0-1.0)
IMAGE_WIDTH         = 1280   # ZED HD720 kamera genişliği (piksel)

# =============================================================
# VİDEO KAYIT
# =============================================================
VIDEO_OUTPUT_DIR = "recordings"   # Kayıt klasörü (otomatik oluşturulur)
VIDEO_FPS        = 10             # Kayıt FPS'i
VIDEO_FRAME_W    = 1280           # Toplam frame genişliği (RGB 640 + Depth 640)
VIDEO_FRAME_H    = 360            # Frame yüksekliği

# =============================================================
# 2D ENGEL HARİTASI (map2d.py)
# Ekstra ZED SDK çağrısı YOK — mevcut depth array kullanılır.
# =============================================================
MAP2D_WORLD_SIZE_M  = 40.0   # Harita toplam kenar boyutu (metre, 40x40m grid)
MAP2D_MAX_DEPTH_M   = 6.0    # Yazılacak maksimum engel mesafesi (metre)
MAP2D_RESOLUTION    = 0.10   # Hücre boyutu (metre) → 400x400 grid
MAP2D_COL_STEP      = 8      # Her kaçıncı sütun örneklenir (1280/8=160 sütun)
MAP2D_SAVE_INTERVAL = 0      # Kaç saniyede bir PNG kaydedilsin (0 = kaydedilmesin)
MAP2D_MERGE_DIST    = 0.8    # Aynı duba kabul mesafesi (metre) — bu altında birleştir
MAP2D_EMA_ALPHA     = 0.2    # Konum güncelleme ağırlığı (0.2 = %20 yeni, %80 eski)
MAP2D_CONFIRM_FRAMES = 3     # Aday dubanın kalıcı olması için gereken ardışık kare sayısı
MAP2D_CAMERA_FOV_H  = 110    # ZED 2 yatay görüş açısı (derece)
MAP2D_INFLATE_M     = 0.5    # Engel şişirme yarıçapı (metre) — araç gövde yarısı + güvenlik marjı
                              # Örnek: USV genişliği 0.6m → yarıçap 0.30 + 0.20 marj = 0.50m
                              # 0.0 = şişirme yok (tek piksel)
