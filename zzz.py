"""
detector/zed_blob_detector.py — ZED CIELAB renk-blob dedektörü (Turuncu / Sarı Duba).

Kırmızı/mavi balon tespiti yerine turuncu (şerit) ve sarı (engel) duba tespiti
yapar.  CIELAB renk uzayında çalışır; morfolojik temizleme ve blob başına derinlik
örneklemesi uygular.  Kare-kare tutarlı blob_id değerleri üretir.

CIELAB (OpenCV 0-255 ölçeği) mantığı:
  • Turuncu : a* yüksek (kırmızı bileşen)  VE  b* yüksek (sarı bileşen)
              a > orange_a_min  AND  b > orange_b_min  AND  L > l_min
  • Sarı    : a* orta/nötr        VE  b* çok yüksek
              a < yellow_a_max  AND  b > yellow_b_min  AND  L > yellow_l_min
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


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

    Parametreler yapılandırmadan enjekte edilir; tüm eşikler OpenCV LAB
    0-255 ölçeğindedir (L: 0-255, a: 0-255 (128=nötr), b: 0-255 (128=nötr)).
    """

    def __init__(
        self,
        *,
        # ── Turuncu: yüksek a* VE yüksek b* ──────────────────────────
        # Kalibrasyon: medyan a* - 1.5*std = 103
        orange_a_min: float = 103,   # a* alt eşiği  (kalibre edildi)
        # Kalibrasyon: medyan b* + 1.5*std = 182 → min ~140 güvenli
        orange_b_min: float = 140,   # b* alt eşiği  (>128 = sarı yön)
        # ── Sarı: düşük a* VE çok yüksek b* ──────────────────────────
        # Kalibrasyon: sarı a* medyan-1.5*std=66 → turuncu alt sınırı (103) altında
        yellow_a_max: float = 100,   # a* üst eşiği  (turuncu a_min=103 altında kalır)
        # Kalibrasyon: medyan b* + 1.5*std = 203 → min ~160 güvenli
        yellow_b_min: float = 160,   # b* alt eşiği  (çok yüksek sarılık)
        # Kalibrasyon: medyan L - 2*std = 99
        yellow_l_min: float = 99,    # Sarı için min parlaklık (kalibre edildi)
        # ── Ortak ────────────────────────────────────────────────────
        # Kalibrasyon: turuncu L_MIN = 20
        l_min: float = 20,           # Genel min L* (kalibre edildi — turuncu için)
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
        self._orange_a_min = float(orange_a_min)
        self._orange_b_min = float(orange_b_min)
        # Sarı eşikleri
        self._yellow_a_max = float(yellow_a_max)
        self._yellow_b_min = float(yellow_b_min)
        self._yellow_l_min = float(yellow_l_min)
        # Ortak
        self._l_min = float(l_min)
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

        # ── Turuncu maskesi: yüksek a* VE yüksek b* ──────────────────
        orange_mask = (
            (a_ch > self._orange_a_min)
            & (b_ch > self._orange_b_min)
            & (l_ch > self._l_min)
        ).astype(np.uint8)

        # ── Sarı maskesi: orta a* VE çok yüksek b* ───────────────────
        yellow_mask = (
            (a_ch < self._yellow_a_max)
            & (b_ch > self._yellow_b_min)
            & (l_ch > self._yellow_l_min)
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
        """Blob içindeki piksel çoğunluğuna göre renk belirler.

        Parlak renklere (turuncu > sarı) öncelik verilir; beraberlikte
        'unknown' döner.
        """
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