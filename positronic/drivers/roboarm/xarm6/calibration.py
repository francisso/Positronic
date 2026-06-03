import socket
import struct


def get_xarm_params_from_arm(arm_ip: str) -> list[dict]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((arm_ip, 502))
    sock.send(bytes([0x00, 0x01, 0x00, 0x02, 0x00, 0x01, 0x08]))
    recv_data = sock.recv(179)
    sock.close()

    if len(recv_data) != 179 or not recv_data[8]:
        raise ValueError(f'Error getting xArm parameters, code: {recv_data[0] if len(recv_data) > 0 else "N/A"}')

    robot_dof = recv_data[9]
    params = struct.unpack('<42f', recv_data[11:])
    kinematics = []
    for i in range(robot_dof):
        kinematics.append({
            'x': params[i * 6],
            'y': params[i * 6 + 1],
            'z': params[i * 6 + 2],
            'roll': params[i * 6 + 3],
            'pitch': params[i * 6 + 4],
            'yaw': params[i * 6 + 5],
        })
    return kinematics
