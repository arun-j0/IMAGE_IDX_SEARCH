import streamlit as st
from app.embeddings import send_img_to_embed
from app.similarity import calculate_similarity
from app.config import INDEX_NAME, NAMESPACE
from pinecone import Pinecone
from PIL import Image

# Initialize Pinecone
api_key = "1ef7ca53-5a38-4e7f-aaf2-84cd42924530"
pc = Pinecone(api_key=api_key)
index = pc.Index(INDEX_NAME)

st.title("Find Similar Images")

img1 = st.file_uploader("Upload an image to find similar ones", type=["jpg", "png"])

if img1:
    try:

        image1 = Image.open(img1)
        embedding1 = send_img_to_embed(image1)
        similarity_order = []

        emd_id = []

        for ids in index.list(namespace="example-namespace"):
            print(ids)
            emd_id = ids
        
        all_data = index.fetch(emd_id ,namespace="example-namespace")



            # Create a new dictionary with id, image_path, and values
        result = {}

        for key, item in all_data['vectors'].items():
            result[item['id']] = {
            'image_path': item['metadata']['image_path'],
            'values': item['values'],
                }
                
        similarity_results = []

        for i in result:
            embedding2 = result[i]["values"]
            res = calculate_similarity(embedding1,embedding2)

            res = float(res)*100
            similarity_results.append({'id': i ,'similarity_score':res,'path':result[i]['image_path']})
            sorted_similarity_results = sorted(similarity_results, key=lambda x: x['similarity_score'], reverse=True)

        for result in sorted_similarity_results:
            img = result["path"]
            st.image(img)

    except Exception as e:
        st.error(f"An error occurred: {e}")
