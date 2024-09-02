import streamlit as st

# Define the pages
PAGES = {
    "Upload Image": "upload_image.py",
    "View Embeddings": "view_embeddings.py",
    "Find Similar Images": "find_similar_images.py",
    "Search Images Using Text": "text_search.py"
}

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    
    # Load the selected page
    page = PAGES[selection]
    
    with open(f'streamlit_app/{page}') as f:
        code = f.read()
    
    exec(code, globals())

if __name__ == "__main__":
    main()
