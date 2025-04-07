import streamlit as st
import requests
import json
from pathlib import Path
import base64
from typing import Dict, List
import markdown

# Configuration
API_BASE_URL = "http://localhost:8000"

def display_image(image_path: str):
    """Display an image from the local filesystem."""
    try:
        image_path = Path(image_path)
        if image_path.exists():
            st.image(str(image_path))
        else:
            st.warning(f"Image not found: {image_path}")
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")

def display_search_results(results: Dict):
    """Display search results with images and LLM response."""
    # Display LLM response
    st.markdown("### AI-Generated Response")
    
    # Process the response to handle image references
    response_text = results["llm_response"]
    parts = response_text.split("[IMAGE:")
    
    # Display the first part (text before any images)
    st.markdown(parts[0], unsafe_allow_html=True)
    
    # Process remaining parts (if any contain images)
    for part in parts[1:]:
        if "]" in part:
            # Split into image path and remaining text
            image_path, remaining_text = part.split("]", 1)
            image_path = image_path.strip()
            
            # Display the image
            try:
                image_path = Path(image_path)
                if image_path.exists():
                    st.image(str(image_path))
                else:
                    st.warning(f"Image not found: {image_path}")
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")
            
            # Display the remaining text
            st.markdown(remaining_text, unsafe_allow_html=True)
    
    # Only display detailed results if they exist and show_details is True
    if "search_results" in results and st.session_state.get("show_details", False):
        st.markdown("### Detailed Results")
        for group in results["search_results"]:
            # Display context (high similarity match)
            if group.get("context"):
                context = group["context"]
                st.markdown(f"##### Page {context['metadata']['page_number']} (High Relevance)")
                st.markdown(context["content"])
                
                # Display images
                if context["images"]:
                    st.markdown("**Related Images:**")
                    for img in context["images"]:
                        display_image(img["path"])
            
            # Display related pages
            if group.get("pages"):
                st.markdown("##### Related Pages")
                for page in group["pages"]:
                    st.markdown(f"**Page {page['metadata']['page_number']}:**")
                    st.markdown(page["content"])
                    
                    # Display images
                    if page["images"]:
                        st.markdown("**Related Images:**")
                        for img in page["images"]:
                            display_image(img["path"])

def main():
    st.title("Technical Documentation RAG System")
    
    # Initialize session state for show_details
    if "show_details" not in st.session_state:
        st.session_state.show_details = False
    
    # Sidebar for document management
    with st.sidebar:
        st.header("Document Management")
        
        # Document upload
        uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
        if uploaded_file:
            files = {"file": uploaded_file}
            response = requests.post(f"{API_BASE_URL}/documents/upload", files=files)
            if response.status_code == 200:
                st.success("Document uploaded successfully!")
                document_id = response.json()["id"]
                
                # Process document
                if st.button("Process Document"):
                    process_response = requests.post(
                        f"{API_BASE_URL}/documents/{document_id}/process"
                    )
                    if process_response.status_code == 200:
                        st.success("Document processed successfully!")
                    else:
                        st.error(f"Error processing document: {process_response.text}")
            else:
                st.error(f"Error uploading document: {response.text}")
    
    # Main content area
    st.header("Search Technical Documentation")
    
    # Search form
    with st.form("search_form"):
        query = st.text_input("Enter your query")
        document_id = st.number_input("Document ID (optional)", min_value=1, step=1)
        threshold = st.slider("Similarity Threshold", 0.1, 1.0, 0.3)
        limit = st.slider("Result Limit", 1, 20, 5)
        context_pages = st.slider("Context Pages", 1, 10, 3)
        include_detailed_results = st.checkbox("Show Detailed Results", value=False)
        
        submitted = st.form_submit_button("Search")
        
        if submitted and query:
            # Update session state
            st.session_state.show_details = include_detailed_results
            
            # Perform enhanced search
            params = {
                "query": query,
                "threshold": threshold,
                "limit": limit,
                "context_pages": context_pages,
                "include_detailed_results": include_detailed_results
            }
            if document_id:
                params["document_id"] = document_id
                
            response = requests.get(f"{API_BASE_URL}/search/enhanced", params=params)
            
            if response.status_code == 200:
                results = response.json()
                display_search_results(results)
            else:
                st.error(f"Error performing search: {response.text}")

if __name__ == "__main__":
    main() 