from abc import ABC, abstractmethod
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# Konfigürasyonu yükle
import config

class SensorInterface(ABC):
    """
    Kamera modelinden bağımsız RGB ve Derinlik haritası erişim arayüzü (SOLID - ISP).
    """
    @abstractmethod
    def get_rgb_image(self):
        pass

    @abstractmethod
    def get_depth_image(self):
        pass

class ZedCameraAdapter(SensorInterface):
    """
    ROS2 ZED Kamera düğümü konularına abone olan ve duba algılama yapan adaptör.
    """
    def __init__(self, node: Node):
        self.node = node
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None

        # Konfigürasyondan alınan topic isimleri ile abonelik başlatılır
        self.rgb_sub = self.node.create_subscription(
            Image,
            config.TOPIC_RGB_IMAGE,
            self.rgb_callback,
            10
        )
        self.depth_sub = self.node.create_subscription(
            Image,
            config.TOPIC_DEPTH_IMAGE,
            self.depth_callback,
            10
        )
        self.node.get_logger().info(f"ZED Kamera Adaptörü Başlatıldı. Abone Olunan Konular: RGB={config.TOPIC_RGB_IMAGE}, Depth={config.TOPIC_DEPTH_IMAGE}")

    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.node.get_logger().error(f"RGB görüntüsü OpenCV matrisine dönüştürülemedi: {e}")

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            self.node.get_logger().error(f"Derinlik görüntüsü OpenCV matrisine dönüştürülemedi: {e}")
    
    def get_rgb_image(self):
        return self.rgb_image

    def get_depth_image(self):
        return self.depth_image

    def detect_colored_buoys(self, rgb_image, depth_image):
        """
        Görüntü üzerindeki turuncu (şerit), sarı (engel), kırmızı/yeşil/siyah (hedef) dubaları algılar.
        """
        if rgb_image is None or depth_image is None:
            return {'orange': [], 'yellow': [], 'black': [], 'red': [], 'green': []}
        
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        # HSV Renk Filtreleri
        lower_orange = np.array([0, 120, 80])
        upper_orange = np.array([18, 255, 255])
        
        lower_yellow = np.array([22, 80, 120])
        upper_yellow = np.array([35, 255, 255])
        
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        
        lower_red1 = np.array([0, 40, 40])
        upper_red1 = np.array([20, 255, 255])
        lower_red2 = np.array([155, 40, 40])
        upper_red2 = np.array([180, 255, 255])
        
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([90, 255, 255])
        
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Gürültü Filtreleme (Kapatma ve Açma Morfolojisi)
        kernel = np.ones((5, 5), np.uint8)
        mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel)
        mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        
        mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
        mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        
        # Duba konumlarını piksel ve metre olarak çıkar (min_area)
        orange_detections = self._extract_buoy_positions(mask_orange, depth_image, min_area=300)
        yellow_detections = self._extract_buoy_positions(mask_yellow, depth_image, min_area=300)
        
        black_detections = self._extract_buoy_positions(mask_black, depth_image, min_area=80)
        red_detections = self._extract_buoy_positions(mask_red, depth_image, min_area=80)
        green_detections = self._extract_buoy_positions(mask_green, depth_image, min_area=80)
        
        return {
            'orange': orange_detections,
            'yellow': yellow_detections,
            'black': black_detections,
            'red': red_detections,
            'green': green_detections
        }
    
    def _extract_buoy_positions(self, mask, depth_image, min_area=300):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Merkez etrafındaki 5x5 bölgenin derinlik ortalamasını al
            y1, y2 = max(0, center_y-2), min(depth_image.shape[0], center_y+3)
            x1, x2 = max(0, center_x-2), min(depth_image.shape[1], center_x+3)
            depth_patch = depth_image[y1:y2, x1:x2]
            
            valid_depths = depth_patch[np.isfinite(depth_patch)]
            if len(valid_depths) > 0:
                distance = float(np.mean(valid_depths))
            else:
                distance = -1.0
            
            # Mesafe sınır kontrolü (config'den min_dist alır)
            if config.MAP_MIN_DIST < distance < 30.0:
                detections.append((center_x, center_y, distance))
        
        return detections
