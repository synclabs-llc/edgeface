import mtcnn
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as patches
from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from torchvision import transforms
from face_alignment import align
from backbones import get_model

detector = mtcnn.MTCNN(device='cpu')
yolo_model = YOLO('models/yolo11n.pt')


def filter_inside_by_y1_difference(face_boxes, body_boxes):
    """
    Keeps only the best body match per face (lowest y1 difference), from the inside matrix.
    
    Parameters:
        face_boxes: (n, 4) - [x1, y1, w, h]
        body_boxes: (N, 4) - [x1, y1, w, h]
    
    Returns:
        filtered_inside: (n, N) - boolean matrix with only one True per row (or all False)
    """
    face_boxes = np.array(face_boxes, dtype=np.float32)
    body_boxes = np.array(body_boxes, dtype=np.float32)

    n, N = face_boxes.shape[0], body_boxes.shape[0]
    if N == 0:
        return np.zeros((n, 0), dtype=bool)

    # Compute inside matrix
    face_x1y1 = face_boxes[:, :2]
    face_x2y2 = face_boxes[:, :2] + face_boxes[:, 2:]
    body_x1y1 = body_boxes[:, :2]
    body_x2y2 = body_boxes[:, :2] + body_boxes[:, 2:]

    inside = (
        (face_x1y1[:, None, 0] >= body_x1y1[None, :, 0]) &
        (face_x1y1[:, None, 1] >= body_x1y1[None, :, 1]) &
        (face_x2y2[:, None, 0] <= body_x2y2[None, :, 0]) &
        (face_x2y2[:, None, 1] <= body_x2y2[None, :, 1])
    )  # shape (n, N)

    face_y1 = face_x1y1[:, 1][:, None]  # (n, 1)
    body_y1 = body_x1y1[None, :, 1]     # (1, N)
    delta_y1 = face_y1 - body_y1        # (n, N)
    delta_y1[~inside] = np.inf          # ignore invalid matches

    # Find body with min y1 diff per face
    best_body_idx = np.argmin(delta_y1, axis=1)     # (n,)
    min_vals = np.min(delta_y1, axis=1)             # (n,)

    # Create filtered matrix
    filtered_inside = np.zeros_like(inside)
    for i in range(n):
        if min_vals[i] != np.inf:
            filtered_inside[i, best_body_idx[i]] = True

    offset = np.where(np.sum(filtered_inside, axis=1) > 0, 0, -1)
    idxes = np.argmax(filtered_inside, axis=1) + offset
    return idxes



def center_to_topleft(boxes):
    # boxes: (N, 4) -> [x_c, y_c, w, h]
    boxes = boxes.cpu().numpy()
    x1y1 = boxes[:, :2] - boxes[:, 2:] / 2  # x1 = x_c - w/2, y1 = y_c - h/2
    wh = boxes[:, 2:]
    return np.hstack([x1y1, wh])  # (N, 4) -> [x1, y1, w, h]

def assign_colors(num_colors):
    return np.random.rand(num_colors, 3)
    

model_name = 'edgeface_xs_gamma_06'
checkpoint_folder = 'checkpoints'
model = get_model(model_name)
checkpoint_path = os.path.join(checkpoint_folder, f"{model_name}.pt")
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.eval()

def draw_boxes(image, results: list, boxes):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    face_boxes = []
    for result in results:
        x, y, w, h = result['box']
        face_boxes.append([x, y, w, h])
    face_boxes = np.array(face_boxes, dtype=np.float32)
    n = len(face_boxes)

    # Convert YOLO boxes to top-left format
    if boxes is not None and len(boxes) > 0:
        body_boxes = center_to_topleft(boxes)
        idxes = filter_inside_by_y1_difference(face_boxes, body_boxes)
        print("Matched face→body:", idxes)

        # Build reverse mapping: body → face
        body_to_face = {}
        matched_faces = []
        for face_idx, body_idx in enumerate(idxes):
            if body_idx != -1:
                body_to_face[body_idx] = face_idx
                matched_faces.append(face_idx)

        # Assign colors only to matched faces
        colors = assign_colors(len(matched_faces))
        face_to_color = {face_idx: colors[i] for i, face_idx in enumerate(matched_faces)}

    else:
        body_boxes = np.empty((0, 4), dtype=np.float32)
        idxes = np.full(n, -1, dtype=int)
        body_to_face = {}
        face_to_color = {}

    # Draw faces
    for i, (x, y, w, h) in enumerate(face_boxes):
        color = face_to_color.get(i, 'black')
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    # Draw body boxes
    for i, (x, y, w, h) in enumerate(body_boxes):
        face_idx = body_to_face.get(i, None)
        color = face_to_color.get(face_idx, 'black') if face_idx is not None else 'black'
        rect = patches.Rectangle((x, y), w, h, linewidth=1, linestyle='dashed', edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    plt.show()
    return idxes

def get_face_body_matches(results: list, boxes):
    """
    Extract face/body boxes and match face to body by vertical alignment and containment.

    Parameters:
        results: list of face detection dicts with 'box': [x1, y1, w, h]
        boxes: tensor or array of YOLO-style boxes [x_c, y_c, w, h]

    Returns:
        face_boxes: (n, 4) np.array of [x1, y1, w, h]
        body_boxes: (N, 4) np.array of [x1, y1, w, h]
        face_to_body_idx: (n,) np.array of body indices matched to each face (or -1)
    """
    # Convert face boxes
    face_boxes = np.array([result['box'] for result in results], dtype=np.float32)

    # Convert YOLO boxes to top-left format
    if boxes is not None and len(boxes) > 0:
        body_boxes = center_to_topleft(boxes)
        face_to_body_idx = filter_inside_by_y1_difference(face_boxes, body_boxes)
        face_boxes, face_to_body_idx = filter_matched_faces(face_boxes, face_to_body_idx)
    else:
        body_boxes = np.empty((0, 4), dtype=np.float32)
        face_to_body_idx = np.full(len(face_boxes), -1, dtype=int)

    return face_boxes, body_boxes, face_to_body_idx


def get_bboxes(image,  person_class = 0):
    results = yolo_model(image)
    idxes = torch.where(results[0].boxes.cls == person_class)
    bboxes = results[0].boxes.xywh[idxes]
    return bboxes

def filter_matched_faces(face_boxes, idxes):
    """
    Remove unmatched face boxes (where idx == -1)

    Parameters:
        face_boxes: (n, 4) np.array
        idxes: (n,) np.array of int (face → body index), with -1 for unmatched

    Returns:
        matched_face_boxes: (n', 4)
        matched_idxes: (n',)
    """
    face_boxes = np.array(face_boxes)
    idxes = np.array(idxes)

    mask = idxes != -1  # keep only matched faces
    matched_face_boxes = face_boxes[mask]
    matched_idxes = idxes[mask]

    return matched_face_boxes, matched_idxes
import os
import torch
from torchvision import transforms
from PIL import Image
from face_alignment import align
from backbones import get_model

def generate_face_embeddings(folder_path, model_name="edgeface_xs_gamma_06", checkpoint_folder="checkpoints"):
    # Load model
    model = get_model(model_name)
    checkpoint_path = os.path.join(checkpoint_folder, f"{model_name}.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    # Preprocessing transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    embeddings = []
    names = []

    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(file)[0]
            image_path = os.path.join(folder_path, file)

            try:
                # Align face
                aligned = align.get_aligned_face(image_path)
                if aligned is None:
                    print(f"Warning: Could not align {file}")
                    continue
                # Show aligned face
                plt.imshow(aligned)
                plt.title(f"Aligned: {name}")
                plt.axis('off')
                plt.show()

                # Preprocess and extract embedding
                input_tensor = transform(aligned).unsqueeze(0)  # Add batch dim
                with torch.no_grad():
                    embedding = model(input_tensor).squeeze().cpu().numpy()
                
                embeddings.append(embedding)
                names.append(name)
            except Exception as e:
                print(f"Error processing {file}: {e}")
    embeddings = np.array(embeddings)

    return embeddings, names

from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import torch

def get_face_embeddings_from_image(image, face_boxes, model, device='cpu'):
    """
    Crop, resize, transform, and embed each face in the image.
    
    Args:
        image: PIL.Image or np.array (H, W, 3)
        face_boxes: (n, 4) - [x1, y1, w, h]
        model: embedding model
        transform: preprocessing function
        device: 'cpu' or 'cuda'
    
    Returns:
        face_embeddings: (n, 512) tensor
    """

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    face_tensors = []
    for x, y, w, h in face_boxes:
        face = image.crop((x, y, x + w, y + h))
        face_tensor = transform(face)
        face_tensors.append(face_tensor)

    face_batch = torch.stack(face_tensors).to(device)

    with torch.no_grad():
        embeddings = model(face_batch)  # shape: (n, 512)

    return embeddings
def recognize_faces_from_aligned(
    img_path: str,
    model,
    transform,
    source_embeddings: np.ndarray,
    names: list,
    align,
    threshold: float = 0.5
):
    """
    Recognize and label faces in an image using aligned face crops.

    Args:
        img_path: Path to group image
        model: PyTorch embedding model
        transform: Transform to preprocess 112x112 face crops
        source_embeddings: (k, 512) numpy array of known face embeddings
        names: list of length k, names of known identities
        align: object with get_aligned_faces() method (returns bboxes, [aligned PIL faces])
        threshold: cosine similarity threshold to consider a match

    Returns:
        PIL.Image with annotated face boxes and names
    """
    # Load original image
    image = Image.open(img_path).convert("RGB")
    bboxes, aligned_faces = align.get_aligned_faces(img_path)

    if not aligned_faces:
        print("No faces detected.")
        return image

    model.eval()
    embeddings = []

    for face in aligned_faces:
        tensor = transform(face).unsqueeze(0)  # [1, 3, 112, 112]
        with torch.no_grad():
            emb = model(tensor).squeeze(0).cpu()
            embeddings.append(emb)

    embeddings = torch.stack(embeddings)                      # (n, 512)
    src_embeddings = torch.tensor(source_embeddings)          # (k, 512)

    # Compute cosine similarity matrix: (n, k)
    cos_sim = F.cosine_similarity(embeddings[:, None, :], src_embeddings[None, :, :], dim=-1)

    # Get best match per face
    best_sim, best_idx = torch.max(cos_sim, dim=1)  # (n,)
    is_match = best_sim >= threshold                # (n,) mask

    # Draw boxes and labels
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for i, box in enumerate(bboxes):
        x, y, x2, y2, _ = map(int, box)

        if is_match[i]:
            name = names[best_idx[i]]
            label = f"{name} ({best_sim[i]:.2f})"
            color = "lime"
        else:
            label = "Unknown"
            color = "red"

        draw.rectangle([x, y, x2, y2], outline=color, width=2)
        draw.text((x, y - 10), label, fill="white", font=font)

    # Show result
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title("Face Recognition Results")
    plt.show()

    return image