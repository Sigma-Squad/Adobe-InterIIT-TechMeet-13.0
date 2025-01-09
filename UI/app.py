import os
import zipfile
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
import base64
from mistralai import Mistral
import re
import json
import cv2
from torch.nn import Module
from torch.autograd import profiler
import matplotlib.pyplot as plt
import time
import statistics
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from torchvision.transforms import functional as F
from torch import Tensor
import tarfile
app = Flask(__name__)
import numpy as np
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'zip', 'tar.gz'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS or
        filename.lower().endswith('.tar.gz')
    )

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
def extract_images_from_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return [os.path.join(extract_to, file) for file in os.listdir(extract_to) if allowed_image_file(file)]
def extract_images_from_tar(tar_path, output_folder):
    """Extract images from a tar.gz file."""
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall(output_folder)

    # Check for a subfolder and move its contents to the parent folder
    for item in os.listdir(output_folder):
        item_path = os.path.join(output_folder, item)
        if os.path.isdir(item_path):
            # Move files from subfolder to the parent folder
            for sub_item in os.listdir(item_path):
                sub_item_path = os.path.join(item_path, sub_item)
                os.rename(sub_item_path, os.path.join(output_folder, sub_item))
            os.rmdir(item_path) 

def load_model(model_path):
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        device = torch.device('cpu')
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
        model.classifier = torch.nn.Linear(model.text_projection.in_features, 2).to(device)
        model.load_state_dict(state_dict, strict=False)
    except:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        device = torch.device('cpu')
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        model.classifier = torch.nn.Linear(model.text_projection.in_features, 2).to(device)
        model.load_state_dict(state_dict, strict=False)
    # print(model.eval())
    return model
def preprocess_mixed(image_path):
    image = Image.open(image_path).convert("RGB")
    try:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    except:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    inputs = processor(images=image, return_tensors="pt")
def preprocess_image(image_path):
    #  transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
    #                          std=[0.26862954, 0.26130258, 0.27577711]),
    # ])
    transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)
def predict_class(image_path, model):
    device = torch.device('cpu')
    image_ten = preprocess_image(image_path).to(device)
    with torch.no_grad():

        image_features = model.get_image_features(image_ten)
        # classifier_head = torch.nn.Linear(model.config.projection_dim, 2)
        logits = model.classifier(image_features)

        probs = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1)

        print(f"Predicted class: {predicted_class.item()}")
        print(f"Class probabilities: {probs}")
        return predicted_class.item()
#task 2
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
class CLIPVisionWrapper(Module):
    """CLIP Vision Wrapper to use with Grad-CAM."""

    def __init__(self, clip_model: CLIPModel):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        outputs = self.clip_model.vision_model(x)
        return outputs.pooler_output

    @property
    def target_layers(self):

      layers = [
          self.clip_model.vision_model.encoder.layers[-5].self_attn.q_proj,
          self.clip_model.vision_model.encoder.layers[-4].self_attn.q_proj,
          self.clip_model.vision_model.encoder.layers[-3].self_attn.q_proj,
          self.clip_model.vision_model.encoder.layers[-2].self_attn.q_proj,
          self.clip_model.vision_model.encoder.layers[-1].self_attn.q_proj
      ]
      return layers    
def grad_cam_clip(images: Tensor, clip_model: CLIPModel) -> Tensor:  #Performs Grad-CAM on a batch of images using CLIP's vision transformer
    clip_model.eval()
    clip_wrapper = CLIPVisionWrapper(clip_model)
    cam = GradCAM(
        model=clip_wrapper,
        target_layers=clip_wrapper.target_layers,
        reshape_transform=_reshape_transform
    )
    grayscale_cam = cam(
        input_tensor=images,
        targets=None,
        eigen_smooth=True,
        aug_smooth=True,
    )

    original_size = images.shape[2:]
    grayscale_cam = torch.tensor(grayscale_cam).unsqueeze(1)

    resized_cam = F.interpolate(
        grayscale_cam,
        size=original_size,
        mode='bilinear',
        align_corners=False
    )

    return resized_cam.squeeze(1)
def _reshape_transform(tensor, height=14, width=14):    #Reshapes the output tensor to fit Grad-CAM's expected format
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

#Xero shot
def mask_gradcam(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path="best_clip_super_res_e30.pth"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    clip_model.classifier = torch.nn.Linear(clip_model.text_projection.in_features, 2).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    img = Image.open(image_path)
    resize_transform = transforms.Resize((224, 224))
    img_resized = resize_transform(img)

    inputs = processor(images=img_resized, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    cam_output = grad_cam_clip(inputs["pixel_values"], clip_model)
    plt.imshow(cam_output[0].cpu().numpy(), cmap='jet', alpha=0.5)
    plt.imshow(img_resized, alpha=0.5)
    plt.axis('off')

    file_name = img.filename.split('/')[-1].split('.')[0]
    # os.makedirs(os.path.dirname("gradcam_images"), exist_ok=True)
    output_path = f"gradcam_images/{file_name}_gradcam.jpg"
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    plt.close()
    return output_path

def jsonify_manual(content):
    explanations = []
    # Regular expression patterns for different formats
    patterns = [
        r"-\s*([A-Za-z0-9\s]+?)\s*-\s*(.*)",
        r"\d+\)\s*([A-Za-z0-9\s]+?)\s*-\s*(.*)",
        r"\*\*\*([A-Za-z0-9\s]+?)\*\*\*\s*-\s*(.*)",
    ]

    lines = content.splitlines()

    for line in lines:
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                artifact = match.group(1).strip()
                explanation = match.group(2).strip()
                explanations.append({artifact: explanation})
                break

    return explanations
api_key = "IjkIRpnmqIyvbUrHticqxJhqjOEopYB5"
def jsonify(explanations):
    api_key = "IjkIRpnmqIyvbUrHticqxJhqjOEopYB5"
    model = "mistral-large-latest"
    client = Mistral(api_key=api_key)
    
    prompt = f"""Convert the following explanations into a JSON object with artifact types as keys and their explanations as values. Format strictly as a valid JSON object.

Content to convert:
{explanations}

Example output format:
{{
    "blurred_edges": "Description of blurred edges",
    "lighting_artifacts": "Description of lighting issues"
}}

Rules:
- Use underscores instead of spaces in keys
- Include only the JSON object, no additional text
- Ensure all quotes are properly escaped
- Each key-value pair should be an artifact and its explanation"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": prompt
                }
            ]
        }
    ]

    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )

    response_text = chat_response.choices[0].message.content.strip()
    
    # Clean up common formatting issues
    response_text = response_text.replace('```json\n', '').replace('\n```', '')
    response_text = response_text.strip()
    
    try:
        # First attempt: direct JSON parsing
        return json.loads(response_text)
    except json.JSONDecodeError:
        try:
            # Second attempt: Clean up potential formatting issues
            cleaned_text = response_text.replace('\n', ' ').replace('\\n', ' ')
            cleaned_text = ' '.join(cleaned_text.split())  # Normalize whitespace
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON. Raw response:\n{response_text}")
            # Return a basic JSON object to avoid breaking the pipeline
            return {"parsing_error": "Failed to generate valid JSON from model response"}
def gen_explanations(base64_image, model_output, max_retries=3, retry_delay=5):
    api_keys = ["PdaIxI19UicKA0n9EfEOhqjtVx9Nx8bm", "LmFCfHzeXhUzsDV4r2FHne8n1PPGus8o"]
    class_name = "AI Generated image" if model_output == 1 else "Real image and not AI generated"
    model = "pixtral-12b-2409"

    artifact_list = f"""- Inconsistent object boundaries
- Discontinuous surfaces
- Non-manifold geometries in rigid structures
- Floating or disconnected components
- Asymmetric features in naturally symmetric objects
- Misaligned bilateral elements in animal faces
- Irregular proportions in mechanical components
- Texture bleeding between adjacent regions
- Texture repetition patterns
- Over-smoothing of natural textures
- Artificial noise patterns in uniform surfaces
- Unrealistic specular highlights
- Inconsistent material properties
- Metallic surface artifacts
- Dental anomalies in mammals
- Anatomically incorrect paw structures
- Improper fur direction flows
- Unrealistic eye reflections
- Misshapen ears or appendages
- Impossible mechanical connections
- Inconsistent scale of mechanical parts
- Physically impossible structural elements
- Inconsistent shadow directions
- Multiple light source conflicts
- Missing ambient occlusion
- Incorrect reflection mapping
- Incorrect perspective rendering
- Scale inconsistencies within single objects
- Spatial relationship errors
- Depth perception anomalies
- Over-sharpening artifacts
- Aliasing along high-contrast edges
- Blurred boundaries in fine details
- Jagged edges in curved structures
- Random noise patterns in detailed areas
- Loss of fine detail in complex structures
- Artificial enhancement artifacts
- Incorrect wheel geometry
- Implausible aerodynamic structures
- Misaligned body panels
- Impossible mechanical joints
- Distorted window reflections
- Anatomically impossible joint configurations
- Unnatural pose artifacts
- Biological asymmetry errors
- Regular grid-like artifacts in textures
- Repeated element patterns
- Systematic color distribution anomalies
- Frequency domain signatures
- Color coherence breaks
- Unnatural color transitions
- Resolution inconsistencies within regions
- Unnatural Lighting Gradients
- Incorrect Skin Tones
- Fake depth of field
- Abruptly cut off objects
- Glow or light bleed around object boundaries
- Ghosting effects: Semi-transparent duplicates of elements
- Cinematization Effects
- Excessive sharpness in certain image regions
- Artificial smoothness
- Movie-poster like composition of ordinary scenes
- Dramatic lighting that defies natural physics
- Artificial depth of field in object presentation
- Unnaturally glossy surfaces
- Synthetic material appearance
- Multiple inconsistent shadow sources
- Exaggerated characteristic features
- Impossible foreshortening in animal bodies
- Scale inconsistencies within the same object class """
    
    prompt = f"""Image said to be an {class_name}.  Explain on the following artifacts only (which are applicable) as to why it might be classified to be so(against or for).
Do not output anything other than the explanations to the artifacts which are applicable. For each artifact, limit the explanations to 50 words.
Do not format the text(no bold).
Artifacts to be considered: 
{artifact_list} """
    
    for attempt in range(max_retries):
        try:
            api_key = api_keys[attempt % len(api_keys)]
            client = Mistral(api_key=api_key)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ]
            
            chat_response = client.chat.complete(
                model=model,
                messages=messages
            )
            
            return chat_response.choices[0].message.content
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return f"Failed to generate explanation after {max_retries} attempts: {str(e)}"
# model_path = "best_clip_super_res_e30.pth"
model_path = "Adobe_quantized_model_e30_task1.pth"
model = load_model(model_path)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_zip():
    if 'zipfile' not in request.files:
        return jsonify({'error': 'No file found'}), 400

    uploaded_file = request.files['zipfile']

    if not allowed_file(uploaded_file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # Ensure the upload directory exists
    upload_folder = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Save the uploaded file
    file_path = os.path.join(upload_folder, secure_filename(uploaded_file.filename))
    uploaded_file.save(file_path)

    # Extract files
    extracted_folder = os.path.join(upload_folder, "extracted/demo_test")
    os.makedirs(extracted_folder, exist_ok=True)

    if uploaded_file.filename.lower().endswith('.zip'):
        extract_images_from_zip(file_path, extracted_folder)
    elif uploaded_file.filename.lower().endswith('.tar.gz'):
        extract_images_from_tar(file_path, extracted_folder)
    # Process each image
    results_task1 = []
    results_task2 = []
    results_task1_d = []
    results_task2_d = []
    for image_file in sorted(os.listdir(extracted_folder)):
        if image_file.endswith(".png") or image_file.endswith(".jpg"):
            image_path = os.path.join(extracted_folder, image_file)
            base64_image = encode_image(image_path)
            model_output = predict_class(image_path, model)
            explanation = gen_explanations(base64_image, model_output)
            json_exp = jsonify(explanation)
            if model_output == 0:
                class_name = "real"
            else:
                class_name = "fake"

            # Append to Task 1 results
            results_task1.append({
                "index": int(image_file.split('.')[0]),
                "prediction": class_name,
                
            })
            results_task1_d.append({
                "index": int(image_file.split('.')[0]),
                "prediction": class_name,
                "image" : base64_image,
                
            })

            # Append to Task 2 results
            results_task2.append({
                "index": int(image_file.split('.')[0]),
                "explanation" : explanation
            })
            print(json_exp)
            results_task2_d.append({
                "index": int(image_file.split('.')[0]),
                "explanation" : json_exp
            })
            print(f"Processed {image_file}: {class_name}")

    # Save results to JSON files
    output_file_task1 = os.path.join(upload_folder, "results_task1.json")
    output_file_task2 = os.path.join(upload_folder, "results_task2_d.json")
    
    with open(output_file_task1, "w") as json_file:
        json.dump(results_task1, json_file, indent=2)
    print(f"Results saved to {output_file_task1}")

    with open(output_file_task2, "w") as json_file:
        json.dump(results_task2_d, json_file, indent=2)
    print(f"Results saved to {output_file_task2}")

    # Return response
    print("rj")
    return {
       
        "results_task1": results_task1_d,
        "results_task2": results_task2,
        "task1_file": output_file_task1,
        "task2_file": output_file_task2
    }

if __name__ == '__main__':
    app.run(debug=True)

