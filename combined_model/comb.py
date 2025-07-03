# %%
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load a COCO-pretrained YOLOv8n model
yolo_model = YOLO("yolo11n.pt")
yolo_model = yolo_model.to(device)  # Move the model to GPU if available
## the other model
face_model_name="edgeface_xs_gamma_06" # or edgeface_xs_gamma_06
face_model=get_model(face_model_name)
checkpoint_path=f'checkpoints/{face_model_name}.pt'
face_model.load_state_dict(torch.load(checkpoint_path, map_location='cuda'))
face_model.to(device)  # Move the model to GPU if available
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])


# Check the device for yolo_model
yolo_device = next(yolo_model.model.parameters()).device
print(f"YOLO model is on device: {yolo_device}")
# Check the device for face_model
face_device = next(face_model.parameters()).device
print(f"Face model is on device: {face_device}")



def get_bboxes(image,  person_class = 0):
    results = yolo_model(image)
    idxes = torch.where(results[0].boxes.cls == person_class)
    bboxes = results[0].boxes.xyxy[idxes]
    return bboxes
    
def crop_image_from_bbox(image, bbox):
    """
    Crop a region from a PIL image using a bounding box.
    bbox: [x1, y1, x2, y2] in pixel coordinates
    """
    x1, y1, x2, y2 = map(int, bbox)
    return image.crop((x1, y1, x2, y2))


def get_embeddings(image):
    bboxes = get_bboxes(image)
    # Example: crop the first detected person
    transformed_faces = []
    idexes = []
    for idx, bbox in enumerate(bboxes):
        cropped_person = crop_image_from_bbox(image, bbox)
        aligned_face = align.get_aligned_face_from_image(cropped_person)
        try:
            transformed_face = transform(aligned_face)
            # print(f"Transformed face size: {transformed_face.size()}")
            transformed_faces.append(transformed_face)
            idexes.append(idx)
        except Exception as e:
            # print("Error transforming face, skipping this one.")
            continue

    if transformed_faces:
        transformed = torch.stack(transformed_faces)
        idxes = torch.tensor(idexes)
        transformed = transformed.to(device) # Move to GPU if available
    else:
        transformed = torch.empty(0, 3, 112, 112)
    embeddings = face_model(transformed)
    return embeddings, bboxes, idxes


def get_single_emb(image_path):
    aligned = align.get_aligned_face(image_path) # align face
    transformed_input = transform(aligned)
    transformed_input = transformed_input.to(device)
    emb = face_model(transformed_input.unsqueeze(0))
    return emb


source_embeddings = get_single_emb("new_test/steve.JPG")



emb, bboxes, idxes = get_embeddings(image)



# Compute cosine similarity between emb and source_embeddings
cos_sim = torch.nn.functional.cosine_similarity(emb, source_embeddings)
# Find the index of the maximum similarity
best_match_idx = torch.argmax(cos_sim)
print("Cosine similarity:", cos_sim)
print("Best match index:", best_match_idx.item())



best_box = bboxes[idxes[best_match_idx.item()]]
final_image = crop_image_from_bbox(image, best_box)


def find_person(image_path, person_image_path):
    image = Image.open(image_path)
    source_embeddings = get_single_emb(person_image_path)
    emb, bboxes, idxes = get_embeddings(image)
    
    if emb.numel() == 0:
        print("No faces detected in the image.")
        return None
    
    cos_sim = torch.nn.functional.cosine_similarity(emb, source_embeddings)
    best_match_idx = torch.argmax(cos_sim)
    # print("Cosine similarity:", cos_sim)
    # print("Best match index:", best_match_idx.item())
    
    best_box = bboxes[idxes[best_match_idx.item()]]
    final_image = crop_image_from_bbox(image, best_box)
    
    # plt.imshow(final_image)
    # plt.axis('off')
    # plt.show()
    
    return final_image

def run_video_tracking(source_image_path="new_test/steve.JPG", camera_index=0):
    source_embeddings = get_single_emb(source_image_path).to(device)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert frame to PIL for consistency
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Inference
        try:
            emb, bboxes, idxes = get_embeddings(pil_image)

            if emb.numel() > 0:
                emb = emb.to(device)
                cos_sim = torch.nn.functional.cosine_similarity(emb, source_embeddings)
                best_match_idx = torch.argmax(cos_sim)
                best_box = bboxes[idxes[best_match_idx.item()].item()].detach().cpu().numpy()

                # Draw bounding box
                x1, y1, x2, y2 = map(int, best_box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Sim: {cos_sim[best_match_idx].item():.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

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
    cap.release()
    cv2.destroyAllWindows()


