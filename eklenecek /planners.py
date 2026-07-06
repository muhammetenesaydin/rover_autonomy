import numpy as np
import config

class GlobalPlanner:
    """
    Yüksek seviyeli waypoint rotalarını yönetir.
    """
    def __init__(self):
        # Konfigürasyon dosyasından waypoints ve kabul yarıçapı yüklenir
        self.waypoints = config.GLOBAL_WAYPOINTS
        self.current_wp_index = 0
        self.acceptance_radius = config.WAYPOINT_ACCEPTANCE_RADIUS

    def get_current_waypoint(self):
        if self.current_wp_index < len(self.waypoints):
            return self.waypoints[self.current_wp_index]
        return None
 
    def check_waypoint_reached(self, x, y):
        if self.current_wp_index >= len(self.waypoints):
            return True
            
        target = self.waypoints[self.current_wp_index]
        dist = np.sqrt((x - target[0])**2 + (y - target[1])**2)
        
        if dist < self.acceptance_radius:
            self.current_wp_index += 1
            return True
        return False

class LocalPlanner:
    """
    Lokal doluluk haritasını, duba algılamalarını ve hedef waypoint konumunu inceleyerek
    anlık hız (linear_vel) ve gövdeye göre hedef açı (target_yaw) kararlarını üretir.
    """
    def __init__(self):
        pass

    def compute_velocity_command(self, occupancy_grid, goal_local_coords, buoy_detections=None, image_width=1280):
        """
        Girdi parametrelerine göre güvenli hız ve yönelim hedeflerini döner.
        """
        if buoy_detections is None:
            buoy_detections = {'orange': [], 'yellow': []}
        
        orange_buoys = buoy_detections.get('orange', [])
        yellow_buoys = buoy_detections.get('yellow', [])
        
        goal_x, goal_y = goal_local_coords
        waypoint_distance = np.sqrt(goal_x**2 + goal_y**2)
        
        # FAZ 1: Hedefe yakın olma durumu (< 3m) -> Sadece hedefe yönel
        if waypoint_distance < 3.0:
            heading_error = np.arctan2(goal_y, goal_x)
            
            # Hata 30 dereceden büyükse yerinde dönüş komutu üret
            if abs(heading_error) > 0.52:
                angular_vel = np.sign(heading_error) * 0.8
                return 0.5, angular_vel
            else:
                return 2.0, heading_error * 0.5
        
        # FAZ 2: Hedefe uzak olma durumu (>= 3m) -> Engellerden kaçma ve şerit koruma
        lane_center_x = None
        lane_offset = 0.0
        
        # Sadece yakındaki (max 10m) turuncu sınır dubalarını baz al
        max_lane_distance = 10.0
        nearby_orange = [buoy for buoy in orange_buoys if buoy[2] < max_lane_distance]
        
        if len(nearby_orange) >= 2:
            orange_x_coords = [buoy[0] for buoy in nearby_orange]
            left_boundary = min(orange_x_coords)
            right_boundary = max(orange_x_coords)
            
            lane_width_pixels = right_boundary - left_boundary
            min_reasonable_lane_width = image_width * 0.1
            max_reasonable_lane_width = image_width * 0.8
            
            if min_reasonable_lane_width < lane_width_pixels < max_reasonable_lane_width:
                potential_lane_center_x = (left_boundary + right_boundary) / 2
                robot_x = image_width / 2
                
                waypoint_direction_x = goal_y
                lane_direction_x = potential_lane_center_x - robot_x
                
                # Şerit yönelimi ile waypoint uyuşuyorsa veya düz gidiyorsak
                if (waypoint_direction_x * lane_direction_x >= 0) or abs(waypoint_direction_x) < 1.0:
                    lane_center_x = potential_lane_center_x
                    lane_width = right_boundary - left_boundary
                    if lane_width > 0:
                        lane_offset = (robot_x - lane_center_x) / (lane_width / 2)
                        lane_offset = np.clip(lane_offset, -1.0, 1.0)
                else:
                    lane_center_x = None
                    lane_offset = 0.0
        
        image_center_x = image_width / 2

        # ÖNCELİK 0: Acil Parkur Sınırı Koruma (Turuncu Dubaya Çok Yakınlaşma)
        orange_danger_dist = 5.0
        orange_danger_width = image_width * 0.8
        
        close_orange = []
        for buoy in orange_buoys:
            o_x, o_y, o_dist = buoy
            if o_dist < orange_danger_dist and abs(o_x - image_center_x) < orange_danger_width:
                close_orange.append(buoy)
                
        if close_orange:
            closest_orange = min(close_orange, key=lambda b: b[2])
            o_x, o_y, o_dist = closest_orange
            if o_x < image_center_x:
                return 0.0, -1.8 # Sağa sert kaç
            else:
                return 0.0, 1.8  # Sola sert kaç

        # ÖNCELİK 1: Sarı Dubalardan (Engellerden) Kaçma
        danger_zone_distance = 12.0
        danger_zone_width = image_width * 0.7
        
        close_obstacles = []
        for buoy in yellow_buoys:
            x, y, distance = buoy
            if distance < danger_zone_distance:
                if abs(x - image_center_x) < danger_zone_width:
                    close_obstacles.append(buoy)
        
        if close_obstacles:
            closest = min(close_obstacles, key=lambda b: b[2])
            obs_x, obs_y, obs_dist = closest
            
            # Dinamik Fren Sistemi (5m altında hızı tamamen kes)
            dynamic_speed = max(0.0, (obs_dist - 5.0) * 0.3)
            
            # Engel mesafesine göre kaçış açısı sertliği
            if obs_dist > 8.0:
                turn_power = 1.5
            elif obs_dist > 5.0:
                turn_power = 2.5
            else:
                turn_power = 3.5
            
            if obs_x < image_center_x:
                if lane_offset < 0.8:
                    return dynamic_speed, -turn_power
                else:
                    return 0.5, -0.2
            else:
                if lane_offset > -0.8:
                    return dynamic_speed, turn_power
                else:
                    return 0.5, 0.2
        
        # ÖNCELİK 2: Şerit Koruma (Şeritten sapma çok yüksekse)
        if lane_center_x is not None and abs(lane_offset) > 0.4:
            correction_gain = 1.6
            angular_correction = lane_offset * correction_gain
            return 1.2, angular_correction
        
        # ÖNCELİK 3: Doluluk Izgarası (Lokal Harita) Üzerinden Navigasyon
        grid_h, grid_w = occupancy_grid.shape
        center_col = grid_w // 2
        
        corridor_width = 10
        lookahead = 100

        front_view = occupancy_grid[grid_h-lookahead:grid_h, center_col-corridor_width:center_col+corridor_width]
        
        if np.any(front_view > 128):
            heading_error = np.arctan2(goal_y, goal_x)
            left_view = occupancy_grid[grid_h-lookahead:grid_h, 0:center_col]
            right_view = occupancy_grid[grid_h-lookahead:grid_h, center_col:]
            
            left_sum = np.sum(left_view)
            right_sum = np.sum(right_view)
            bias_weight = (grid_h * grid_w * 255) * 0.2
            
            left_score = left_sum
            right_score = right_sum
            
            if heading_error > 0:
                right_score += bias_weight
            else:
                left_score += bias_weight
            
            if left_score < right_score:
                if lane_offset > -0.7:
                    return 2.0, 0.5
                else:
                    return 1.5, 0.2
            else:
                if lane_offset < 0.7:
                    return 2.0, -0.5
                else:
                    return 1.5, -0.2
        else:
            # Engel yoksa hedef waypoint yönüne git
            heading_error = np.arctan2(goal_y, goal_x)
            k_p = 0.8
            angular_vel = k_p * heading_error
            
            if lane_center_x is not None:
                lane_correction = lane_offset * 0.4
                angular_vel += lane_correction
                if abs(lane_offset) > 0.3:
                    angular_vel = np.clip(angular_vel, -1.0, 1.0)
                    return 2.5, angular_vel

            angular_vel = np.clip(angular_vel, -1.0, 1.0)
            return 5.0, angular_vel
