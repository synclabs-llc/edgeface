import numpy as np
import cv2
from PIL import Image
import torch
import onnxruntime as ort
import torch.nn.functional as F

from face_alignment import align


class PersonTrackerONNX:
    def __init__(self, face_model_path, yolo_model_path, reference_image_path):
        # Load ONNX runtime sessions
        self.face_session = ort.InferenceSession(face_model_path, providers=['CPUExecutionProvider'])
        self.yolo_session = ort.InferenceSession(yolo_model_path, providers=['CPUExecutionProvider'])

        # Get input/output names
        self.face_input = self.face_session.get_inputs()[0].name
        self.face_output = self.face_session.get_outputs()[0].name
        self.yolo_input = self.yolo_session.get_inputs()[0].name
        self.yolo_output = self.yolo_session.get_outputs()[0].name

        self.align = align.get_aligned_face_from_image

        ref_img = Image.open(reference_image_path).convert("RGB")
        aligned_ref = self.align(ref_img)
        self.reference_embedding = self.get_face_embedding(aligned_ref)

        self.cap = None
        self.last_frame = None
        self.last_hist = None

    def preprocess_image(self, image, size=(112, 112)):
        img = image.resize(size)
        img_np = np.asarray(img).astype(np.uint8)
        img_np = np.transpose(img_np, (2, 0, 1))  # CHW
        return img_np[np.newaxis, :]  # 1, 3, H, W

    def get_face_embedding(self, face_img):
        input_tensor = self.preprocess_image(face_img)
        embedding = self.face_session.run([self.face_output], {self.face_input: input_tensor})[0]
        return embedding.squeeze()

    def detect_persons(self, image):
        img_resized = image.resize((640, 640))
        img_np = np.array(img_resized).astype(np.uint8)
        img_np = np.transpose(img_np, (2, 0, 1))[np.newaxis, :]
        outputs = self.yolo_session.run(None, {self.yolo_input: img_np})
        return outputs[0]  # depends on YOLO ONNX output structure

    def compute_histogram_batch(self, imgs, bins=256):
        N, C, H, W = imgs.shape
        flat = imgs.reshape(N, C, -1)
        hist = torch.zeros(N, C, bins)
        for b in range(bins):
            hist[:, :, b] = (flat == b).sum(dim=2)
        hist = hist / hist.sum(dim=2, keepdim=True)
        return hist.view(N, -1)

    def cosine_score(self, ref, embeddings):
        ref_norm = ref / np.linalg.norm(ref)
        emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return np.dot(emb_norm, ref_norm)

    def crop_image(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        return image.crop((x1, y1, x2, y2))

    def start_tracking(self, cam=0):
        self.cap = cv2.VideoCapture(cam)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            detections = self.detect_persons(pil_image)
            # Note: You must parse YOLO output appropriately here

            # For demonstration, just show frame
            cv2.imshow("ONNX Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
