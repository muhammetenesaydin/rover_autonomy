import time
from pymavlink import mavutil

connection_string = 'udp:127.0.0.1:14550'
print(f"Connecting to {connection_string}...")
master = mavutil.mavlink_connection(connection_string)
master.wait_heartbeat()
print("Heartbeat received!")

def get_parameter(name):
    master.mav.param_request_read_send(
        master.target_system, master.target_component,
        name.encode('utf-8'), -1
    )
    while True:
        msg = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=5)
        if msg is None:
            return None
        if msg.param_id == name:
            return msg.param_value

print("Reading SERVO1_FUNCTION...")
s1 = get_parameter("SERVO1_FUNCTION")
print(f"SERVO1_FUNCTION: {s1}")

print("Reading SERVO3_FUNCTION...")
s3 = get_parameter("SERVO3_FUNCTION")
print(f"SERVO3_FUNCTION: {s3}")
