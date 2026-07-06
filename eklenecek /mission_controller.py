from enum import Enum, auto

class MissionState(Enum):
    """ Görev modlarını tutan durum yapısı """
    IDLE = auto()
    NAVIGATING = auto()
    AVOIDING_OBSTACLE = auto()
    DOCKING = auto()
    COMPLETED = auto()
    TURNING = auto()  # Keskin Yerinde Dönüş Durumu

class MissionController:
    """
    Görev durum makinesini (State Machine) yönetir.
    """
    def __init__(self):
        self.state = MissionState.IDLE
        
    def update(self, waypoint_reached, obstacle_detected, target_detected, heading_aligned=False):
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
            pass
            
        return self.state
