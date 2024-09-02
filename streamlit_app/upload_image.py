import streamlit as st
from app.embeddings import send_img_to_embed
from app.image_utils import save_image, ensure_images_dir_exists
from app.config import INDEX_NAME, NAMESPACE
from pinecone import Pinecone
from PIL import Image
import os

# Initialize Pinecone
api_key = "1ef7ca53-5a38-4e7f-aaf2-84cd42924530"
pc = Pinecone(api_key=api_key)
index = pc.Index(INDEX_NAME)

ensure_images_dir_exists()

st.title("Upload and Store Image Embedding")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_image:
    try:
        img = Image.open(uploaded_image)
        embedding = send_img_to_embed(img)

        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_path = os.path.join("images", uploaded_image.name)
        save_image(img, img_path)

        if st.button("Store Image Embedding"):
            embedding_list = embedding.cpu().numpy().tolist()[0]
            index.upsert(
                vectors=[
                    {
                        "id": uploaded_image.name,
                        "values": embedding_list,
                        "metadata": {"description": "Uploaded Image", "image_path": img_path}
                    }
                ],
                namespace=NAMESPACE
            )
            st.write("Image embedding stored in Pinecone.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
