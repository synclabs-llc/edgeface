import numpy as np
import matplotlib.pyplot as plt
import torch
import torch
from torchvision import transforms
from face_alignment import align
from backbones import get_model
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
from PIL import Image
import cv2
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torchmetrics.functional import structural_similarity_index_measure as ssim


class PersonTracker:
    def __init__(self, face_model_name, yolo_detection_path, refrence_image_path, device='cuda'):
        self.face_model = get_model(face_model_name)
        self.detection_model = YOLO(yolo_detection_path)
        self.device = torch.device(device)
        self.face_model.to(self.device)
        self.detection_model.to(self.device)
        self.detection_model.eval()
        self.face_model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        image = Image.open(refrence_image_path)
        self.align = align.get_aligned_face_from_image
        transformed_input = self.transform(self.align(image))
        self.refrence_embedding = self.face_model(transformed_input.unsqueeze(0).to(device))
        self.cap = None
        self.last_frame = None ## last frame tensor for histogram
        self.last_hist = None ## histogram of last frame

    def get_bboxes(self, image,  person_class = 0):
        results = self.detection_model(image)
        idxes = torch.where(results[0].boxes.cls == person_class)
        bboxes = results[0].boxes.xyxy[idxes]
        return bboxes
    
    def crop_image_from_bbox(self, image, bbox):
        """
        Crop a region from a PIL image using a bounding box.
        bbox: [x1, y1, x2, y2] in pixel coordinates
        """
        x1, y1, x2, y2 = map(int, bbox)
        return image.crop((x1, y1, x2, y2))

    def ssim(self, image):
        if self.last_frame is None:
            self.last_frame = image
            return 1
        score = ssim(self.last_frame, image)
        return score
    
    ### get the image embeddings and 
    def get_embeddings(self, image):
        
        bboxes = self.get_bboxes(image)
        # Example: crop the first detected person
        transformed_faces = []
        idexes = []
        for idx, bbox in enumerate(bboxes):
            cropped_person = self.crop_image_from_bbox(image, bbox)
            aligned_face = self.align(cropped_person)
            try:
                transformed_face = self.transform(aligned_face)
                # print(f"Transformed face size: {transformed_face.size()}")
                transformed_faces.append(transformed_face)
                idexes.append(idx)
            except Exception as e:
                # print("Error transforming face, skipping this one.")
                continue

        if transformed_faces:
            transformed = torch.stack(transformed_faces)
            
            transformed = transformed.to(self.device)
            embeddings = self.face_model(transformed)# Move to GPU if available
        else:

            embeddings = torch.empty(0, 512).to(self.device)
        idxes = torch.tensor(idexes).to(self.device)
        return embeddings, bboxes, idxes
    
    def compute_histogram_batch(self, imgs, bins=256, value_range=(-1, 1)):
        """
        Compute normalized RGB histograms for a batch of images.

        Args:
            imgs (Tensor): (N, 3, H, W) image batch, values in `value_range`
            bins (int): Number of histogram bins
            value_range (tuple): Min and max pixel values (e.g., (-1, 1) after normalization)

        Returns:
            Tensor: (N, 3 * bins), concatenated R+G+B histograms per image
        """
        N, C, H, W = imgs.shape
        assert C == 3, "Expected 3 channels (RGB)"
        device = imgs.device
        min_val, max_val = value_range

        # Flatten to (N, C, H*W)
        flat = imgs.view(N, C, -1)

        # Scale to [0, bins - 1]
        scaled = ((flat - min_val) / (max_val - min_val) * (bins - 1)).long()
        scaled = torch.clamp(scaled, 0, bins - 1)

        # Create histogram tensor
        histograms = torch.zeros(N, C, bins, device=device)

        # Compute histogram for each channel
        for b in range(bins):
            histograms[:, :, b] = (scaled == b).sum(dim=2)

        # Normalize histograms (per image, per channel)
        histograms = histograms / histograms.sum(dim=2, keepdim=True)  # shape (N, 3, bins)

        # Flatten to (N, 3 * bins)
        return histograms.view(N, -1)

    def start_tracking(self, camera_index =0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Convert frame to PIL for consistency
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Inference
            try:
                emb, bboxes, idxes = self.get_embeddings(pil_image)

                if emb.numel() > 0:
                   
                    cos_sim = torch.nn.functional.cosine_similarity(emb, self.refrence_embedding)
                    best_match_idx = torch.argmax(cos_sim)
                    best_box = bboxes[idxes[best_match_idx.item()].item()].detach().cpu().numpy()

                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, best_box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Sim: {cos_sim[best_match_idx].item():.2f}",
                                (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                    last_img = self.crop_image_from_bbox(pil_image, best_box)
                    self.last_frame = self.transform(last_img).to(self.device)
                else:
                    if self.last_frame is None:
                        print("Need a face in frame once to start tracking")
                        ## should restart the loop
                    else : 
                        print("moving to tracker-with-no-face")
                       
                        transformed_curr = []
                        for idx, bbox in enumerate(bboxes):
                            cropped_person = self.crop_image_from_bbox(pil_image, bbox)
                            transformed_event = self.transform(cropped_person)
                            transformed_curr.append(transformed_event)
                        transformed_curr = torch.stack(transformed_curr).to(self.device)
                        hist = self.compute_histogram_batch(transformed_curr)
                        hist_score = self.get_distribution_score(hist)
                        best_match_idx = torch.argmax(hist_score)
                        best_box = bboxes[best_match_idx.item()].detach().cpu().numpy()
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, best_box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Distributed score: {hist_score[best_match_idx].item():.2f}",
                                    (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 2)
                        self.last_frame = transformed_curr[best_match_idx].detach()
                        self.last_hist = hist[best_match_idx].detach()
                # Show the frame
                cv2.imshow("Tracking", frame)

            except Exception as e:
                print(f"Error during inference: {e}")

            # Clear unused memory on GPU
            torch.cuda.empty_cache()

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
    
    def get_distribution_score(self, hist):
    
        if self.last_hist is None:
            self.last_hist = self.compute_histogram_batch(self.last_frame.unsqueeze(0))
        """
        Compare two histograms using cosine similarity.
        Returns 1.0 if identical, 0.0 if orthogonal.
        """
        return F.cosine_similarity(self.last_hist, hist, dim =1)
    

    