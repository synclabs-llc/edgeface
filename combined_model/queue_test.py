import cv2
import time
import threading
import queue
import psutil
from PIL import Image
from track_face_no_face import PersonTracker  # replace with your actual class path

# === CONFIGURATION ===
CAM_SOURCE = 0
VIDEO_PATH = "../video/test.mp4"
SWITCH_INTERVAL_SEC = 10
CPU_CORES = [0, 1]
QUEUE_SIZE = 20

# === CPU AFFINITY ===
def set_cpu_affinity(cores):
    p = psutil.Process()
    p.cpu_affinity(cores)

# === SHARED QUEUE ===
frame_queue = queue.Queue(maxsize=QUEUE_SIZE)

# === ALTERNATING FRAME FEEDER ===
def alternating_feeder(cam_src, video_path, queue, switch_interval=10):
    cap_cam = cv2.VideoCapture(cam_src)
    cap_vid = cv2.VideoCapture(video_path)

    last_switch = time.time()
    use_camera = True

    while True:
        now = time.time()
        if now - last_switch > switch_interval:
            use_camera = not use_camera
            last_switch = now
            print(f"[Feeder] Switching to {'camera' if use_camera else 'video'}")

        cap = cap_cam if use_camera else cap_vid
        source = 'camera' if use_camera else 'video'

        ret, frame = cap.read()
        if not ret:
            if not use_camera:
                cap_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
                continue
            else:
                print("[Feeder] Camera feed ended.")
                break

        try:
            queue.put((source, frame), timeout=1)
        except queue.Full:
            print("[Feeder] Queue full, dropping frame.")

# === WORKER FUNCTION ===
def frame_worker(name, tracker, queue, lock):
    set_cpu_affinity(CPU_CORES)
    while True:
        try:
            source, frame = queue.get(timeout=3)
        except queue.Empty:
            print(f"[{name}] Queue empty, exiting.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        with lock:
            embeddings, bboxes, idxes = tracker.get_embeddings(pil_image)

        print(f"[{name}] From {source}: {len(bboxes)} bboxes detected")

# === MAIN ===
if __name__ == "__main__":
    set_cpu_affinity(CPU_CORES)

    # Initialize your tracker (CPU only)
    tracker = PersonTracker(
        face_model_name="edgeface_xs_gamma_06",
        yolo_detection_path="models/yolo11n.pt",
        refrence_image_path="faces/jim.jpeg",
        device='cpu'
    )

    lock = threading.Lock()

    # Start alternating feeder
    feeder_thread = threading.Thread(
        target=alternating_feeder,
        args=(CAM_SOURCE, VIDEO_PATH, frame_queue, SWITCH_INTERVAL_SEC)
    )
    feeder_thread.start()

    # Start two worker threads
    worker1 = threading.Thread(target=frame_worker, args=("Worker-1", tracker, frame_queue, lock))
    worker2 = threading.Thread(target=frame_worker, args=("Worker-2", tracker, frame_queue, lock))
    worker1.start()
    worker2.start()

    # Wait for all threads to finish
    feeder_thread.join()
    worker1.join()
    worker2.join()

    print("All done.")
