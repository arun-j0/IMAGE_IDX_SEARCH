import streamlit as st
from app.embeddings import send_text_to_embed
from app.similarity import calculate_similarity
from app.config import INDEX_NAME, NAMESPACE
from pinecone import Pinecone

# Initialize Pinecone
api_key = "1ef7ca53-5a38-4e7f-aaf2-84cd42924530"
pc = Pinecone(api_key=api_key)
index = pc.Index(INDEX_NAME)

st.title("Find Images by Text")

# Text input for search query
search_query = st.text_input("Enter a description to find similar images")

if search_query:
    try:
        # Convert the search query to an embedding
        embedding1 = send_text_to_embed(search_query)
        similarity_order = []
        emd_id = []

        # Fetch all vectors in the namespace
        for ids in index.list(namespace="example-namespace"):
            emd_id = ids
        
        all_data = index.fetch(emd_id, namespace="example-namespace")

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
            res = calculate_similarity(embedding1, embedding2)
            res = float(res) * 100
            similarity_results.append({'id': i, 'similarity_score': res, 'path': result[i]['image_path']})
        
        # Sort the results based on similarity score in descending order
        sorted_similarity_results = sorted(similarity_results, key=lambda x: x['similarity_score'], reverse=True)

        # Display the images based on similarity
        for result in sorted_similarity_results:
            img = result["path"]
            st.image(img)

    except Exception as e:
        st.error(f"An error occurred: {e}")
