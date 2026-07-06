#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zed_obstacle.py — Thread-safe ZED Kamera + Engel Algılama + VIO Odometri

VIO (Visual-Inertial Odometry):
  ZED2/ZED2i/ZED Mini içindeki IMU, görsel odometri ile füzyonlanır.
  get_vio() → (x_m, z_m, vio_yaw_deg, imu_yaw_deg)
    x_m        : araç sağ eksenindeki yer değişimi (metre, + sağ)
    z_m        : araç ileri eksenindeki yer değişimi (metre, + ileri)
    vio_yaw_deg: ZED Euler yaw — görsel+IMU füzyonu (0=başlangıç yönü)
    imu_yaw_deg: Ham IMU yaw — drift'li ama anlık (0=başlangıç yönü)
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import List, Optional
import cv2
import numpy as np
import pyzed.sl as sl
from conf import (
    ZED_RESOLUTION, ZED_FPS, ZED_DEPTH_MODE, ZED_MIN_DIST,
    OBSTACLE_THRESHOLD, DEPTH_PERCENTILE,
    ROI_Y_RATIO, LEFT_X_RATIO, CENTER_X_RATIO, RIGHT_X_RATIO,
    MAP_MIN_DIST, CAMERA_HEIGHT, MIN_OBSTACLE_HEIGHT,
)


# ══════════════════════════════════════════════════════════════════════════════
# CIELAB Turuncu/Sarı Duba Blob Dedektörü
# ══════════════════════════════════════════════════════════════════════════════
#
# CIELAB (OpenCV 0-255 ölçeği):
#   L : 0-255  (parlaklık)
#   a : 0-255  (128 = nötr | >128 = kırmızı/turuncu | <128 = yeşil)
#   b : 0-255  (128 = nötr | >128 = sarı            | <128 = mavi)
#
#   Turuncu : a* > orange_a_min  VE  b* > orange_b_min  VE  L* > l_min
#   Sarı    : a* < yellow_a_max  VE  b* > yellow_b_min  VE  L* > yellow_l_min
#
# Kalibrasyon kaynakları (gerçek duba görüntülerinden ölçüldü):
#   Turuncu → orange_a_min=103 (medyan a*-1.5σ),  l_min=20  (medyan L-2σ)
#   Sarı    → yellow_a_max=100 (a* < turuncu alt sınırı),
#             yellow_b_min=160 (medyan b*+1.5σ=203 → güvenli alt sınır),
#             yellow_l_min=99  (medyan L-2σ)

@dataclass
class BlobDetection:
    """ZED karesinde tespit edilen tek bir renk blobu."""

    blob_id: int              # kare-kare tutarlı ID
    cx: float                 # blob merkez x [px]
    cy: float                 # blob merkez y [px]
    x1: int                   # sınırlayıcı kutu sol
    y1: int                   # sınırlayıcı kutu üst
    x2: int                   # sınırlayıcı kutu sağ
    y2: int                   # sınırlayıcı kutu alt
    area: float               # kontur alanı [px²]
    depth_m: Optional[float]  # ROI medyan derinliği [m]
    color: str                # "orange" | "yellow" | "unknown"


class ZedBlobDetector:
    """CIELAB tabanlı turuncu/sarı duba blob dedektörü (ZED kamerası için).

    Tüm eşikler OpenCV LAB 0-255 ölçeğindedir
    (L: 0-255, a: 0-255 (128=nötr), b: 0-255 (128=nötr)).
    """

    def __init__(
        self,
        *,
        # ── Turuncu: L*, a*, b* Alt ve Üst Sınırları (Hassas Kalibrasyon) ──
        orange_l_min: float = 88,    # Turuncu parlaklık alt sınırı
        orange_l_max: float = 177,   # Turuncu parlaklık üst sınırı
        orange_a_min: float = 156,   # Turuncu a* alt sınırı
        orange_a_max: float = 165,   # Turuncu a* üst sınırı
        orange_b_min: float = 152,   # Turuncu b* alt sınırı
        orange_b_max: float = 170,   # Turuncu b* üst sınırı
        # ── Sarı: L*, a*, b* Alt ve Üst Sınırları (Hassas Kalibrasyon) ──        
        yellow_l_min: float = 201,   # Sarı parlaklık alt sınırı
        yellow_l_max: float = 255,   # Sarı parlaklık üst sınırı
        yellow_a_min: float = 107,   # Sarı a* alt sınırı
        yellow_a_max: float = 111,   # Sarı a* üst sınırı
        yellow_b_min: float = 178,   # Sarı b* alt sınırı
        yellow_b_max: float = 191,   # Sarı b* üst sınırı
        # ── Ortak ────────────────────────────────────────────────────
        min_area: float = 80,        # Minimum kontur alanı [px²]
        max_area: float = 50000,     # Maksimum kontur alanı [px²]
        morph_kernel: int = 5,       # Morfoloji çekirdek boyutu
        cx_match_thresh: float = 100,  # ID eşleştirme için max cx farkı [px]
        # ── Doygunluk reddi (aşırı parlak piksel filtresi) ────────────
        reject_saturated: bool = True,
        saturation_v_min: int = 245,
        saturation_s_max: int = 45,
        max_saturated_ratio: float = 0.20,
        # ── Morfoloji ────────────────────────────────────────────────
        use_morph_open: bool = True,
    ) -> None:
        # Turuncu eşikleri
        self._orange_l_min = float(orange_l_min)
        self._orange_l_max = float(orange_l_max)
        self._orange_a_min = float(orange_a_min)
        self._orange_a_max = float(orange_a_max)
        self._orange_b_min = float(orange_b_min)
        self._orange_b_max = float(orange_b_max)
        # Sarı eşikleri
        self._yellow_l_min = float(yellow_l_min)
        self._yellow_l_max = float(yellow_l_max)
        self._yellow_a_min = float(yellow_a_min)
        self._yellow_a_max = float(yellow_a_max)
        self._yellow_b_min = float(yellow_b_min)
        self._yellow_b_max = float(yellow_b_max)
        # Ortak
        self._min_area = float(min_area)
        self._max_area = float(max_area)
        self._morph_kernel = max(1, int(morph_kernel))
        self._cx_match_thresh = float(cx_match_thresh)
        # Doygunluk reddi
        self._reject_saturated = reject_saturated
        self._saturation_v_min = int(saturation_v_min)
        self._saturation_s_max = int(saturation_s_max)
        self._max_saturated_ratio = float(max_saturated_ratio)
        # Morfoloji
        self._use_morph_open = use_morph_open

        k = self._morph_kernel
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        # Kare-kare ID durumu
        self._prev_blobs: List[BlobDetection] = []
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Genel API
    # ------------------------------------------------------------------

    def detect(
        self,
        bgr_frame: np.ndarray,
        depth_mat: Optional[np.ndarray],
    ) -> List[BlobDetection]:
        """Tek bir BGR karesi üzerinde blob tespiti yapar.

        Kare-kare tutarlı ``blob_id`` değerlerine sahip ``BlobDetection``
        listesi döner.
        """
        # ── CIELAB dönüşümü ──────────────────────────────────────────
        lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2Lab)
        l_ch = lab[:, :, 0].astype(np.float32)
        a_ch = lab[:, :, 1].astype(np.float32)
        b_ch = lab[:, :, 2].astype(np.float32)

        # ── Turuncu maskesi: L*, a*, b* Alt ve Üst Sınır Aralığı ────────
        orange_mask = (
            (l_ch >= self._orange_l_min) & (l_ch <= self._orange_l_max)
            & (a_ch >= self._orange_a_min) & (a_ch <= self._orange_a_max)
            & (b_ch >= self._orange_b_min) & (b_ch <= self._orange_b_max)
        ).astype(np.uint8)

        # ── Sarı maskesi: L*, a*, b* Alt ve Üst Sınır Aralığı ──────────
        yellow_mask = (
            (l_ch >= self._yellow_l_min) & (l_ch <= self._yellow_l_max)
            & (a_ch >= self._yellow_a_min) & (a_ch <= self._yellow_a_max)
            & (b_ch >= self._yellow_b_min) & (b_ch <= self._yellow_b_max)
        ).astype(np.uint8)

        # Turuncu ve sarı bölgelerini birleştir
        mask_u8 = cv2.bitwise_or(orange_mask, yellow_mask)

        # ── Doygunluk reddi (yansıma / aşırı parlak piksel filtresi) ─
        saturated_mask: Optional[np.ndarray] = None
        if self._reject_saturated:
            hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
            saturated_mask = (
                (hsv[:, :, 2] > self._saturation_v_min)
                & (hsv[:, :, 1] < self._saturation_s_max)
            ).astype(np.uint8)

        # ── Morfoloji (gürültü temizleme) ────────────────────────────
        if self._use_morph_open:
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, self._kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, self._kernel)

        # ── Kontur tespiti ────────────────────────────────────────────
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates: List[BlobDetection] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self._min_area or area > self._max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

            # Ağırlık merkezi
            m = cv2.moments(cnt)
            if m["m00"] > 0:
                cx = float(m["m10"] / m["m00"])
                cy = float(m["m01"] / m["m00"])
            else:
                cx = float(x + w / 2.0)
                cy = float(y + h / 2.0)

            # Kontur iç maskesi (bounding box koordinatlarında)
            cnt_crop = cnt - (x, y)
            cnt_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt_crop], -1, 255, -1)

            # Doygunluk reddi kontrolü
            if saturated_mask is not None:
                sat_crop = saturated_mask[y1:y2, x1:x2]
                total_px = max(float(cv2.countNonZero(cnt_mask)), 1.0)
                sat_px = float(
                    cv2.countNonZero(cv2.bitwise_and(cnt_mask, cnt_mask, mask=sat_crop))
                )
                if sat_px / total_px > self._max_saturated_ratio:
                    continue

            # Renk sınıflandırması (blob içindeki dominant renk)
            o_crop = orange_mask[y1:y2, x1:x2]
            y_crop = yellow_mask[y1:y2, x1:x2]
            color = self._blob_color(cnt_mask, o_crop, y_crop)

            # Derinlik örneklemesi
            depth_crop = (
                depth_mat[y1:y2, x1:x2]
                if depth_mat is not None and depth_mat.ndim >= 2
                else None
            )
            depth_m = self._blob_depth_crop(cnt_mask, depth_crop, w, h)

            candidates.append(
                BlobDetection(
                    blob_id=-1,
                    cx=cx,
                    cy=cy,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    area=area,
                    depth_m=depth_m,
                    color=color,
                )
            )

        candidates.sort(key=lambda b: b.cx)
        return self._assign_ids(candidates)

    # ------------------------------------------------------------------
    # İç yardımcı metodlar
    # ------------------------------------------------------------------

    def _blob_color(
        self,
        cnt_mask: np.ndarray,
        orange_crop: np.ndarray,
        yellow_crop: np.ndarray,
    ) -> str:
        """Blob içindeki piksel çoğunluğuna göre renk belirler."""
        orange_cnt = cv2.countNonZero(
            cv2.bitwise_and(cnt_mask, cnt_mask, mask=orange_crop)
        )
        yellow_cnt = cv2.countNonZero(
            cv2.bitwise_and(cnt_mask, cnt_mask, mask=yellow_crop)
        )
        if orange_cnt > yellow_cnt:
            return "orange"
        elif yellow_cnt > orange_cnt:
            return "yellow"
        return "unknown"

    def _blob_depth_crop(
        self,
        cnt_mask: np.ndarray,
        depth_crop: Optional[np.ndarray],
        crop_w: int,
        crop_h: int,
    ) -> Optional[float]:
        """Kontur maskesinden medyan derinlik örnekler; iç-üçte-bir ROI yedek."""
        if depth_crop is None or cnt_mask is None or not cnt_mask.any():
            return None
        vals = depth_crop[cnt_mask != 0].ravel().astype(float)
        valid = vals[np.isfinite(vals) & (vals > 0.0)]
        if valid.size > 0:
            return float(np.median(valid))
        # Yedek: bounding box'ın iç üçte birlik bölgesi
        mx1, mx2 = crop_w // 3, crop_w - crop_w // 3
        my1, my2 = crop_h // 3, crop_h - crop_h // 3
        if mx2 <= mx1 or my2 <= my1:
            return None
        roi = depth_crop[my1:my2, mx1:mx2].ravel().astype(float)
        valid = roi[np.isfinite(roi) & (roi > 0.0)]
        return float(np.median(valid)) if valid.size > 0 else None

    def _assign_ids(self, candidates: List[BlobDetection]) -> List[BlobDetection]:
        """Kare-kare tutarlı blob_id değerleri atar."""
        if not self._prev_blobs:
            for b in candidates:
                b.blob_id = self._next_id
                self._next_id += 1
            self._prev_blobs = list(candidates)
            return candidates

        assigned: List[BlobDetection] = []
        prev_sorted = sorted(self._prev_blobs, key=lambda b: b.cx)
        cur_sorted = sorted(candidates, key=lambda b: b.cx)

        if len(cur_sorted) == len(prev_sorted):
            # Sayı aynıysa sıralı eşleştir (en hızlı yol)
            for pb, cb in zip(prev_sorted, cur_sorted):
                cb.blob_id = pb.blob_id
                assigned.append(cb)
        else:
            # Greedy en yakın cx eşleştirmesi
            used_prev: set = set()
            for cb in cur_sorted:
                best_i, best_d = None, float("inf")
                for i, pb in enumerate(prev_sorted):
                    if i in used_prev:
                        continue
                    d = abs(cb.cx - pb.cx)
                    if d < best_d:
                        best_d, best_i = d, i
                if best_i is not None and best_d < self._cx_match_thresh:
                    cb.blob_id = prev_sorted[best_i].blob_id
                    used_prev.add(best_i)
                else:
                    cb.blob_id = self._next_id
                    self._next_id += 1
                assigned.append(cb)

        self._prev_blobs = list(assigned)
        return assigned




class ZEDCamera:
    """
    Thread-safe ZED kamera sınıfı.
    Arka planda sürekli frame çeker; get_frame() ile en güncel veriyi verir.
    """

    def __init__(self, resolution=ZED_RESOLUTION, fps=ZED_FPS):
        self.zed = sl.Camera()

        params = sl.InitParameters()
        params.camera_resolution       = resolution
        params.camera_fps              = fps
        params.depth_mode              = ZED_DEPTH_MODE
        params.coordinate_units        = sl.UNIT.METER
        params.depth_minimum_distance  = ZED_MIN_DIST
        # IMU + Görsel Odometri füzyonu için koordinat sistemi:
        # IMAGE  → kamera eksenleri (x=sağ, y=aşağı, z=ileri)
        params.coordinate_system       = sl.COORDINATE_SYSTEM.IMAGE

        if self.zed.open(params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED kamera açılamadı!")

        # Kamera kalibrasyon parametrelerini al (fx, fy, cx, cy)
        calibration_parameters = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        self.fx = calibration_parameters.fx
        self.fy = calibration_parameters.fy
        self.cx = calibration_parameters.cx
        self.cy = calibration_parameters.cy

        self._rt      = sl.RuntimeParameters()
        self._rt.enable_fill_mode = True  # SDK 4.x: Derinlik haritasındaki boşlukları ve delikleri otomatik doldurur

        self._img_buf   = sl.Mat()
        self._dep_buf   = sl.Mat()
        self._pose_buf  = sl.Pose()       # VIO pozisyon tamponu
        self._sens_buf  = sl.SensorsData()  # IMU veri tamponu

        self._lock      = threading.Lock()
        self._rgb       = None
        self._depth     = None
        self._running   = False
        self._thread    = None

        # CIELAB tabanlı turuncu/sarı duba dedektörü
        self._blob_detector = ZedBlobDetector()

        # VIO çıktısı (thread-safe)
        self._vio_x   = 0.0   # sağ eksen (metre)
        self._vio_z   = 0.0   # ileri eksen (metre)
        self._vio_yaw = 0.0   # ZED Euler yaw (derece, görsel+IMU)
        self._imu_yaw = 0.0   # Ham IMU yaw (derece)
        self._vio_ok  = False  # VIO verisi geçerli mi?

    def start(self):
        # 1. Positional Tracking — IMU destekli VIO modunda başlat
        track_params = sl.PositionalTrackingParameters()
        track_params.enable_imu_fusion = True   # IMU + Görsel füzyon (VIO)
        err = self.zed.enable_positional_tracking(track_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"  ⚠️ Positional Tracking (VIO) başlatılamadı: {err}")
        else:
            print("✓ ZED VIO (Görsel-Atalet Odometrisi) aktif — IMU füzyonu açık.")

        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("✓ ZED thread başlatıldı.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self.zed.disable_positional_tracking()
        self.zed.close()
        print("✓ ZED kamera kapatıldı.")


    def _loop(self):
        while self._running:
            if self.zed.grab(self._rt) != sl.ERROR_CODE.SUCCESS:
                continue

            # ── Görüntü + Derinlik ──────────────────────────────────────
            self.zed.retrieve_image(self._img_buf, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self._dep_buf, sl.MEASURE.DEPTH)
            rgb       = self._img_buf.get_data().copy()
            raw_depth = self._dep_buf.get_data().copy()

            # 3D Yükseklik Filtrelemesi (Su Yüzeyi/Zemin Filtresi)
            h, w = raw_depth.shape[:2]
            v_indices = np.arange(h).reshape(-1, 1)
            # Y koordinatı (aşağı yönlü pozitif)
            Y = ((v_indices - self.cy) * raw_depth) / self.fy
            # Su yüzeyinden yükseklik (kamera CAMERA_HEIGHT yüksekliğinde, su Y = CAMERA_HEIGHT seviyesinde)
            height_above_water = CAMERA_HEIGHT - Y
            # Sadece su yüzeyinden en az MIN_OBSTACLE_HEIGHT yüksek olan pikselleri kabul et
            valid_mask = (height_above_water >= MIN_OBSTACLE_HEIGHT) & (raw_depth > ZED_MIN_DIST) & (raw_depth < 30.0)
            depth = np.where(valid_mask, raw_depth, np.nan)

            # ── VIO Pozisyon (görsel + IMU füzyonu) ────────────────────
            state = self.zed.get_position(self._pose_buf, sl.REFERENCE_FRAME.WORLD)
            vio_ok  = (state == sl.POSITIONAL_TRACKING_STATE.OK)
            vio_x = vio_z = vio_yaw = 0.0
            if vio_ok:
                t = self._pose_buf.get_translation().get()
                # IMAGE koordinat sistemi: t[0]=sağ, t[1]=aşağı, t[2]=ileri
                vio_x = float(t[0])   # metre, + sağ
                vio_z = float(t[2])   # metre, + ileri
                # Euler açıları (rx, ry, rz) — ry = yaw (IMAGE sisteminde)
                rot = self._pose_buf.get_euler_angles()
                vio_yaw = math.degrees(float(rot[1]))  # derece

            # ── Ham IMU Yaw ────────────────────────────────────────────
            imu_yaw = 0.0
            if self.zed.get_sensors_data(self._sens_buf,
                                          sl.TIME_REFERENCE.IMAGE) == sl.ERROR_CODE.SUCCESS:
                imu = self._sens_buf.get_imu_data()
                # IMU quaternion → Euler yaw
                q = imu.get_pose().get_orientation().get()  # [qx, qy, qz, qw]
                qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
                # Yaw (z ekseni, ZED IMAGE frame'de ry'ye karşılık gelir)
                imu_yaw = math.degrees(
                    math.atan2(2.0*(qw*qy + qx*qz),
                               1.0 - 2.0*(qy*qy + qz*qz))
                )

            with self._lock:
                self._rgb     = rgb
                self._depth   = depth
                self._vio_x   = vio_x
                self._vio_z   = vio_z
                self._vio_yaw = vio_yaw
                self._imu_yaw = imu_yaw
                self._vio_ok  = vio_ok

    def get_frame(self):
        """(rgb_bgra, depth_m) döner. Henüz veri yoksa (None, None).

        NOT: Kopya döndürülmez — tüketiciler (analyze, recorder) sadece
        okuma yaptığı için referans vermek güvenlidir. _loop() ZED buffer'ı
        zaten bir kez kopyaladığından ikinci kopya gereksizdir.
        """
        with self._lock:
            if self._rgb is None:
                return None, None
            return self._rgb, self._depth  # referans — kopya YOK (210 MB/s tasarruf)

    def get_vio(self) -> tuple:
        """Thread-safe VIO + IMU verisi döner.

        Returns:
            (x_m, z_m, vio_yaw_deg, imu_yaw_deg, vio_ok)
              x_m        : sağ eksen yer değişimi (metre, + sağ)
              z_m        : ileri eksen yer değişimi (metre, + ileri)
              vio_yaw_deg: Görsel+IMU füzyon yaw (derece, başlangıç=0)
              imu_yaw_deg: Ham IMU yaw (derece, başlangıç=0)
              vio_ok     : Tracking durumu geçerliyse True
        """
        with self._lock:
            return (self._vio_x, self._vio_z,
                    self._vio_yaw, self._imu_yaw, self._vio_ok)

    def detect_colored_buoys(self, rgb_image, depth_image):
        """
        Turuncu (şerit) ve sarı (engel) dubaları CIELAB renk uzayında tespit eder.

        ZedBlobDetector kullanır:
          • CIELAB maskeleri (kalibre edilmiş a*, b*, L* eşikleri)
          • Morfolojik gürültü temizleme
          • Kontur başına medyan derinlik örneklemesi
          • Kare-kare tutarlı blob_id takibi

        Döndürür:
            {'orange': [(cx, cy, dist_m), ...],
             'yellow': [(cx, cy, dist_m), ...]}
        """
        if rgb_image is None or depth_image is None:
            return {'orange': [], 'yellow': []}

        # BGRA → BGR dönüşümü (ZED 4-kanal çıktısı)
        if rgb_image.ndim == 3 and rgb_image.shape[2] == 4:
            bgr = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2BGR)
        else:
            bgr = rgb_image

        blobs = self._blob_detector.detect(bgr, depth_image)

        # ZedBlobDetector çıktısını mevcut format {'orange':[], 'yellow':[]}'a çevir
        buoys = {'orange': [], 'yellow': []}
        for b in blobs:
            if b.color in buoys and b.depth_m is not None:
                buoys[b.color].append((b.cx, b.cy, b.depth_m))

        return buoys



class ObstacleDetector:
    """
    Derinlik haritasını SOL / MERKEZ / SAĞ koridora böler.
    Merkez mesafesi OBSTACLE_THRESHOLD altındaysa engel algılanır.
    Hangi taraf daha açıksa oraya kaçınma yönü belirlenir.
    """

    def __init__(self, threshold=OBSTACLE_THRESHOLD, percentile=DEPTH_PERCENTILE):
        self.threshold  = threshold
        self.percentile = percentile
        # Önceki geçerli ölçümler (inf geçişi için)
        self._last_valid = {'left': float('inf'), 'center': float('inf'), 'right': float('inf')}

    def analyze(self, depth: np.ndarray) -> dict:
        """
        Returns:
            obstacle   : bool
            emergency  : bool  — tüm bölgeler inf (çok yakın, dur!)
            direction  : 'left' | 'right' | 'none'
            dist_left  : float (m)
            dist_center: float (m)
            dist_right : float (m)
        """
        h, w = depth.shape[:2]
        y0, y1 = int(h * ROI_Y_RATIO[0]), int(h * ROI_Y_RATIO[1])
        roi = depth[y0:y1, :]
        roi = np.where(np.isfinite(roi), roi, np.nan)

        def min_dist(col_slice):
            vals = col_slice[np.isfinite(col_slice)]
            return float(np.percentile(vals, self.percentile)) if vals.size else float('inf')

        lx0, lx1 = int(w * LEFT_X_RATIO[0]),   int(w * LEFT_X_RATIO[1])
        cx0, cx1 = int(w * CENTER_X_RATIO[0]),  int(w * CENTER_X_RATIO[1])
        rx0, rx1 = int(w * RIGHT_X_RATIO[0]),   int(w * RIGHT_X_RATIO[1])

        dl = min_dist(roi[:, lx0:lx1])
        dc = min_dist(roi[:, cx0:cx1])
        dr = min_dist(roi[:, rx0:rx1])

        # --- inf = çok yakın (ZED min_dist altı) kontrolü ---
        # Eğer merkez veya her iki yan inf ise ve önceki ölçüm yakındıysa → acil dur
        def _resolve(val, key):
            """inf gelirse önceki geçerli değeri kullan (en kötü durum)."""
            if val == float('inf') and self._last_valid[key] < self.threshold * 2:
                return self._last_valid[key]  # engel hâlâ orada sayılır
            if val != float('inf'):
                self._last_valid[key] = val
            return val

        dl = _resolve(dl, 'left')
        dc = _resolve(dc, 'center')
        dr = _resolve(dr, 'right')

        # Acil durum: tüm bölgeler hâlâ inf → çok yakın, anında dur
        emergency = (dl == float('inf') and dc == float('inf') and dr == float('inf'))

        # Merkez 1.2m'den yakınsa VEYA sol/sağ 1.0m'den yakınsa engel kabul et (genişlik güvenliği)
        obstacle  = (dc < self.threshold) or (dl < self.threshold * 0.8) or (dr < self.threshold * 0.8) or emergency
        direction = ('left' if dl > dr else 'right') if obstacle else 'none'

        return {
            'obstacle':     obstacle,
            'emergency':    emergency,
            'direction':    direction,
            'dist_left':    dl,
            'dist_center':  dc,
            'dist_right':   dr,
        }



