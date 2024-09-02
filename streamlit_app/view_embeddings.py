import streamlit as st
from app.config import INDEX_NAME, NAMESPACE
from pinecone import Pinecone

# Initialize Pinecone
api_key = "1ef7ca53-5a38-4e7f-aaf2-84cd42924530"
pc = Pinecone(api_key=api_key)
index = pc.Index(INDEX_NAME)

st.title("View Stored Image Embeddings")

try:
    ids = index.list(namespace='example-namespace')
    st.write("All IDs:", ids)

    for ids in index.list(namespace="example-namespace"):
        emd_id = ids
            
    col1,col2 = st.columns(2)

    for i in emd_id:
        embedding_info = index.fetch(ids=[i], namespace="example-namespace")
        embedding = embedding_info['vectors'][i]['values']
        metadata = embedding_info['vectors'][i]['metadata']
    
                # Creating columns
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"Image ID: {i}")
            st.image(metadata['image_path'], use_column_width=True)

        with col2:    
            st.write(f"Embedding: {embedding[:100]}")
                
except Exception as e:
    st.error(f"An error occurred: {e}")