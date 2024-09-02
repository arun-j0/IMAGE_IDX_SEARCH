import torch
from transformers import CLIPProcessor, CLIPModel
from app.config import API_KEY, INDEX_NAME, NAMESPACE
from pinecone import Pinecone

# Initialize Pinecone
pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)

# Initialize CLIP model and processor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def image_embed(image_to_embed):
    with torch.no_grad():
        image_embedding = model.get_image_features(**image_to_embed)
    image_embedding = image_embedding / image_embedding.norm(p=2, dim=-1, keepdim=True)
    return image_embedding

def send_img_to_embed(input_img):
    image_to_embed = processor(images=input_img, return_tensors="pt").to(device)
    return image_embed(image_to_embed)

def text_embed(text_to_embed):
    with torch.no_grad():
        text_embedding = model.get_text_features(**text_to_embed)
    text_embedding = text_embedding / text_embedding.norm(p=2, dim=-1, keepdim=True)
    return text_embedding

def send_text_to_embed(input_text):
    text_to_embed = processor(text=input_text, return_tensors="pt").to(device)
    return text_embed(text_to_embed)
