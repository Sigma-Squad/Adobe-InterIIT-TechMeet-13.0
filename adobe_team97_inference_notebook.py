# -*- coding: utf-8 -*-
"""" # Installation of libraries and imports """

# import statements
import torch
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import base64
import requests
import os
import tarfile
from mistralai import Mistral
import json
import zipfile
from torch.autograd import profiler
import time
import statistics
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from torch import Tensor
from torch.nn import Module
import matplotlib.pyplot as plt
import torch.nn.functional as F

"""# Input Model Path and Input Zip folder"""

# Upload the path to the model and the input zip file

model_path = "Adobe_model_CIFAKE_e50.pth"
input_folder = "test-interiit.tar.gz"

""" # Task 1 - Classification of Image """

def load_model(model_path):                    # Load Pretrained CLIP Model and Weights of Finetuned model
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


def preprocess_mixed(image_path):             # Get and process image from image path
    image = Image.open(image_path).convert("RGB")
    try:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    except:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    inputs = processor(images=image, return_tensors="pt")


def preprocess_image(image_path):              # Get and preprocess image from image path (improvised)
    # transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
    #                      std=[0.26862954, 0.26130258, 0.27577711]),
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


def predict_class(image_path, model):       # Wrapper Class for Task 1 to take input of image path and model and output whether it is Real or Fake
    device = torch.device('cpu')
    image_ten = preprocess_image(image_path).to(device)

    with torch.no_grad():
        image_features = model.get_image_features(image_ten)
        # classifier_head = torch.nn.Linear(model.config.projection_dim, 2)
        logits = model.classifier(image_features)

        probs = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1)

        # print(f"Predicted class: {predicted_class.item()}")
        # print(f"Class probabilities: {probs}")
        return predicted_class.item()


"""# Task 2 - Explanability of Image Classification"""


def encode_image(image_path):        # Encode Image to base64 to pass as an parameter to a Large Language Model
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Grad-Cam Mask of Image input

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

def mask_gradcam(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path="Adobe_model_CIFAKE_e50.pth"
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

    file_name = img.filename.split('\\')[-1].split('.')[0]

    # if not os.path.exists("gradcam_images"): 
    #     os.makedirs("gradcam_images")

    output_path = f"gradcam_images\{file_name}_gradcam.jpg"
    print(output_path)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    plt.close()
    return output_path


def gen_explanations(base64_image, model_output, max_retries=3, retry_delay=5):
    api_keys = ["CjetUPdd9OZj42bO6ZnfCibWTMG9M90A", "sGLBkpRIs5LWLH9B4Dw3o0sXSlaghtvn"]
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

    prompt =  f"""Image is said to be an {class_name}. Explain on the following artifacts only(which are applicable) as to why it might be classified to be so(against or for).
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
                            "text": f"Image is said to be an {class_name}. Explain why based on visible artifacts."
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

def jsonify(explanations):
    api_key = "MtZjdXzS1FPebXt18Y2BDQ9fhZQG3FDH"
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
            return response_text

def run_pipeline(image_file, model, extracted_folder):
    try:
        image_path = os.path.join(extracted_folder, image_file)
        model_output = predict_class(image_path, model)
        masked_image_path = mask_gradcam(image_path)
        base64_image = encode_image(masked_image_path)
        
        explanation = gen_explanations(base64_image, model_output)
        if "Failed to generate explanation" in explanation:
            json_exp = {"error": explanation}
        else:
            json_exp = jsonify(explanation)
        
        class_name = "real" if model_output == 0 else "fake"
        index = int(image_file.split('.')[0])
        
        return {
            "task1": {"index": index, "prediction": class_name},
            "task2": {"index": index, "explanation": json_exp}
        }
        
    except Exception as e:
        print(f"Error processing {image_file}: {str(e)}")
        return None

"""# Inference"""

def get_inference_time(model, image_path):  # Inference Time Calculator for Task 1 process

    with profiler.profile(record_shapes=True) as prof:
        with profiler.record_function("model_inference"):
            output = predict_class(image_path, model)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print()


    for _ in range(10):
        _ = predict_class(image_path, model)
    num_iterations = 100
    inference_times = []
    for _ in range(num_iterations):
        start_time = time.time()
        output = predict_class(image_path, model)
        end_time = time.time()
        inference_times.append(end_time - start_time)
    average_inference_time = statistics.mean(inference_times)
    print(f"Average inference time: {average_inference_time:.4f} seconds")

# MAIN

output_file_task1 = "97_task1.json"
output_file_task2 = "97_task2.json"

extracted_folder = input_folder.split(".")[0]
extenstion = input_folder.split(".")[-1]

if (extenstion == "gz"):
    if not os.path.exists(extracted_folder):
        with tarfile.open(input_folder, 'r:gz') as tar_ref:
            tar_ref.extractall(extracted_folder)
else:
    extracted_folder = input_folder.split(".")[0]
    if not os.path.exists(extracted_folder):
        with zipfile.ZipFile(input_folder, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder)


model = load_model(model_path)

# Inference Time calculation
#get_inference_time(model, extracted_folder + "/demo_test/6.png")
print()

# Run of Task 1 + Task 2
results_task1 = []
results_task2 = []

subfolders = [f.path for f in os.scandir(extracted_folder) if f.is_dir()]
if subfolders:
    extracted_folder = subfolders[0]

for image_file in os.listdir(extracted_folder):
    if image_file.endswith((".png", ".JPEG")):
        result = run_pipeline(image_file, model, extracted_folder)
        
        if result:
            results_task1.append(result["task1"])
            results_task2.append(result["task2"])
            print(f"Processed {image_file}:{result['task1']['prediction']}")
        
        time.sleep(2)  # Rate limiting

with open(output_file_task1, "w") as json_file:
    json.dump(results_task1, json_file, indent=2)
print(f"Results saved to {output_file_task1}")
with open(output_file_task2, "w") as json_file:
    json.dump(results_task2, json_file, indent=2)
print(f"Results saved to {output_file_task2}")