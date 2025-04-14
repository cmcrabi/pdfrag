import streamlit as st
import requests
import json
from pathlib import Path
import base64
from typing import Dict, List, Tuple, Optional
import markdown
import logging

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration
API_BASE_URL = "http://localhost:8000"

def get_products() -> Dict[str, int]:
    """Fetches products from the API and returns a name -> id mapping."""
    try:
        response = requests.get(f"{API_BASE_URL}/products/")
        response.raise_for_status()  # Raise an exception for bad status codes
        products = response.json()
        # Create a dictionary mapping name to id
        product_map = {product['name']: product['id'] for product in products}
        logger.info(f"Fetched {len(product_map)} products.")
        return product_map
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching products: {e}")
        logger.error(f"Error fetching products: {e}", exc_info=True)
        return {}
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching products: {e}")
        logger.error(f"Unexpected error fetching products: {e}", exc_info=True)
        return {}

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
    # Optional: Display Product ID searched
    if "product_id" in results:
        st.markdown(f"*(Searched within Product ID: {results['product_id']})*")

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
    st.set_page_config(layout="wide")  # Use wider layout
    st.title("Technical Documentation RAG System")
    
    # Fetch products at the beginning
    products_map = get_products()
    product_names = list(products_map.keys())
    
    # Initialize session state for show_details
    if "show_details" not in st.session_state:
        st.session_state.show_details = False
    
    # Sidebar for document management
    with st.sidebar:
        st.header("Document Management")
        
        # --- Product Creation ---
        st.subheader("Create Product")
        with st.form("product_form"):
            new_product_name = st.text_input("New Product Name")
            submitted_product = st.form_submit_button("Create Product")
            if submitted_product and new_product_name:
                try:
                    response = requests.post(f"{API_BASE_URL}/products/", json={"name": new_product_name})
                    if response.status_code == 200:
                        st.success(f"Product '{new_product_name}' created successfully!")
                        # Refresh products list (optional, might require rerun)
                        st.rerun()
                    elif response.status_code == 400:
                        st.error(f"Error: {response.json().get('detail', response.text)}")
                    else:
                        response.raise_for_status()  # Raise for other errors
                except requests.exceptions.RequestException as e:
                    st.error(f"Error creating product: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

        # --- Document Upload ---
        st.subheader("Upload Document")
        if not product_names:
            st.warning("Please create a product first before uploading documents.")
        else:
            selected_product_name_upload = st.selectbox(
                "Select Product for Upload",
                options=product_names,
                index=None, # Default to no selection
                placeholder="Choose a product..."
            )
            
            # Add inputs for version and title
            doc_version = st.text_input("Version (Optional)", key="upload_version")
            doc_title = st.text_input("Title (Optional)", key="upload_title")

            uploaded_file = st.file_uploader(
                "Upload PDF Document", 
                type=["pdf"], 
                disabled=(not selected_product_name_upload), 
                key="pdf_uploader"
            )

            # Use a button to trigger the upload explicitly
            upload_button_disabled = not (uploaded_file and selected_product_name_upload)
            if st.button("Upload and Process Document", disabled=upload_button_disabled, key="upload_button"):
                # This block now runs only when the button is clicked
                selected_product_id_upload = products_map.get(selected_product_name_upload)
                if selected_product_id_upload:
                    # Prepare form data including product_id, version, title
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    data = {"product_id": selected_product_id_upload}
                    if doc_version:
                        data['version'] = doc_version
                    if doc_title:
                        data['title'] = doc_title
                    # Always send content_type from the uploaded file object
                    if uploaded_file.type:
                         data['content_type'] = uploaded_file.type

                    # Use triple quotes for the f-string to handle internal quotes easily
                    st.info(f"""Uploading '{uploaded_file.name}' (Version: {doc_version or 'N/A'}, Title: {doc_title or 'N/A'}) 
                            for product '{selected_product_name_upload}' (ID: {selected_product_id_upload})...""")
                    try:
                        # Step 1: Upload the document
                        response = requests.post(f"{API_BASE_URL}/documents/upload", files=files, data=data)

                        if response.status_code == 200:
                            st.success("Document uploaded successfully! Now processing...")
                            document_info = response.json()
                            document_id = document_info["id"]
                            
                            # Step 2: Process the document (re-added)
                            with st.spinner(f"Processing document ID {document_id}..."):
                                process_response = requests.post(
                                    f"{API_BASE_URL}/documents/{document_id}/process"
                                )
                                if process_response.status_code == 200:
                                    st.success("Document uploaded and processed successfully!")
                                    # Rerun to clear state after successful processing
                                    st.rerun()
                                else:
                                    # Log the processing error but maybe don't fail the whole upload
                                    logger.error(f"Error processing document {document_id} after upload: {process_response.status_code} - {process_response.text}")
                                    st.error(f"Document uploaded, but processing failed: {process_response.text}")

                        elif response.status_code == 404: # Product not found during upload
                            st.error(f"Upload Error: {response.json().get('detail', response.text)}")
                        elif response.status_code == 409: # Duplicate hash
                            st.warning(f"Upload Warning: {response.json().get('detail', response.text)}")
                        elif response.status_code == 400: # Invalid product ID or other bad request
                            st.error(f"Upload Error: {response.json().get('detail', response.text)}")
                        else:
                            response.raise_for_status() # Raise for other errors

                    except requests.exceptions.RequestException as e:
                        st.error(f"Error uploading document: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during upload: {e}")
                else:
                    st.error("Selected product not found in map. Please refresh.")

    # Main content area
    st.header("Search Technical Documentation")
    
    if not product_names:
        st.warning("No products found. Please create a product and upload documents first.")
    else:
        # Search form
        with st.form("search_form"):
            query = st.text_input("Enter your query")
            selected_product_name_search = st.selectbox(
                "Select Product to Search",
                options=product_names,
                index=0,  # Default to first product
                # placeholder="Choose a product..." # Not needed if default is set
            )
            # Remove: document_id = st.number_input("Document ID (optional)", min_value=1, step=1)
            threshold = st.slider("Similarity Threshold", 0.1, 1.0, 0.7)  # Default 0.7
            limit = st.slider("Result Limit", 1, 20, 7)  # Default 7
            context_pages = st.slider("Context Pages (for search retrieval)", 1, 10, 5)  # Default 5
            include_detailed_results = st.checkbox("Show Detailed Results", value=st.session_state.show_details)

            submitted = st.form_submit_button("Search")

            if submitted and query and selected_product_name_search:
                # Update session state for checkbox persistence
                st.session_state.show_details = include_detailed_results

                selected_product_id_search = products_map.get(selected_product_name_search)
                if not selected_product_id_search:
                    st.error("Selected product not found. Please refresh.")
                else:
                    st.info(f"Searching for '{query}' in product '{selected_product_name_search}' (ID: {selected_product_id_search})...")
                    # Perform enhanced search
                    params = {
                        "query": query,
                        "product_id": selected_product_id_search,  # Use product_id
                        "threshold": threshold,
                        "limit": limit,
                        "context_pages": context_pages,
                        "include_detailed_results": include_detailed_results
                    }
                    # Remove: if document_id:
                    # Remove:     params["document_id"] = document_id

                    try:
                        response = requests.get(f"{API_BASE_URL}/search/enhanced", params=params)
                        response.raise_for_status()  # Raise for bad status codes

                        results = response.json()
                        display_search_results(results)

                    except requests.exceptions.RequestException as e:
                        st.error(f"Error performing search: {e}")
                        logger.error(f"Search API Error: Status {response.status_code}, Body: {response.text}", exc_info=True)
                    except Exception as e:
                        st.error(f"An unexpected error occurred during search: {e}")
                        logger.error(f"Search Unexpected Error: {e}", exc_info=True)

if __name__ == "__main__":
    main() 