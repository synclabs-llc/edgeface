import socket
import cv2
import pickle
import struct
import numpy as np
import threading

# Receiver configuration
host_ip = '0.0.0.0'
port = 8485

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host_ip, port))
server_socket.listen(5)
print(f"🟢 Listening on {host_ip}:{port}...")

conn, addr = server_socket.accept()
print(f"✅ Connected by {addr}")

data = b""
payload_size = struct.calcsize("Q")
streaming = False  # Initial state

def control_input():
    global streaming
    while True:
        cmd = input("▶️ 'r'=resume, ⏸ 'p'=pause: ").strip().lower()
        if cmd == 'r' and not streaming:
            conn.sendall(b'READY')
            print("📩 Sent READY")
            streaming = True
        elif cmd == 'p' and streaming:
            conn.sendall(b'PAUSE')
            print("📩 Sent PAUSE")
            streaming = False

# Start control thread
threading.Thread(target=control_input, daemon=True).start()

try:
    while True:
        while len(data) < payload_size:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet

        if len(data) < payload_size:
            continue

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += conn.recv(4096)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        if streaming and frame is not None:
            cv2.imshow("📷 Live Stream", frame)

        if cv2.waitKey(1) == ord('q'):
            break

except Exception as e:
    print("❌ Error:", e)
finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
    print("🔌 Connection closed.")
