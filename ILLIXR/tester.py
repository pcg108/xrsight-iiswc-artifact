import socket
import csv
import time
import struct
import numpy as np

HOST_SOCKET = "/tmp/illixr-host"
POSE_FILE = '/scratch/prashanth/ILLIXR/poses.csv'

sample_conn = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
sample_conn.connect(HOST_SOCKET)

fovea_positions = [[60, 40], [180, 40], [180, 120], [60, 120]] # 4 corners

fixed_pose = [-0.905151, 1.01911, -0.839719, -0.99481, -0.0283108, -0.0737443, 0.0641368]

foveation_level = 2

while True:
    for fovea_position in fovea_positions:
        
        # render
        float_values = np.concatenate((np.array([0, 0], dtype=np.float32), np.array(fixed_pose, dtype=np.float32), np.array(fovea_position, dtype=np.float32), np.array([foveation_level], dtype=np.float32)))
        # print(float_values)
        data = float_values.tobytes()

        sample_conn.sendall(data)
        data = sample_conn.recv(8)
        d1 = struct.unpack('d', data)
        print("Received time:", d1)

        # timewarp
        float_values = np.concatenate((np.array([1, 0], dtype=np.float32), np.array(fixed_pose, dtype=np.float32), np.array(fovea_position, dtype=np.float32), np.array([foveation_level], dtype=np.float32)))
        # print(float_values)
        data = float_values.tobytes()

        sample_conn.sendall(data)
        data = sample_conn.recv(8)
        d1 = struct.unpack('d', data)
        print("Received time, bytes:", d1)

# with open(POSE_FILE, newline='') as csvfile:
#     csv_reader = csv.reader(csvfile)
#     for i, row in enumerate(csv_reader):

#         float_values = np.array(row, dtype=np.float32)

#         if (i%60==0):
#             float_values = np.concatenate((np.array([0, 0], dtype=np.float32), float_values, np.array([0, 0], dtype=np.float32)))
#             print(float_values)
#             data = float_values.tobytes()

#             sample_conn.sendall(data)

#             data = sample_conn.recv(16)
#             d1, d2 = struct.unpack('dd', data)
#             print("Received time, bytes:", d1, d2)

#             # time.sleep(1)

#         if (i%60==1):
#             float_values = np.concatenate((np.array([1, 0], dtype=np.float32), float_values, np.array([0, 0], dtype=np.float32)))
#             print(float_values)
#             data = float_values.tobytes()

#             sample_conn.sendall(data)

#             data = sample_conn.recv(16)
#             d1, d2 = struct.unpack('dd', data)
#             print("Received time, bytes:", d1, d2)
            
            # time.sleep(1)
