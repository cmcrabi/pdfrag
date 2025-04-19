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

# Placeholder for fetching documents - needs API endpoint
def get_documents_for_product(product_id: int) -> List[Dict]:
    """Fetches documents for a specific product ID from the API."""
    if not product_id:
        logger.warning("get_documents_for_product called with no product_id")
        return []
        
    api_url = f"{API_BASE_URL}/products/{product_id}/documents/"
    try:
        response = requests.get(api_url)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        documents = response.json()
        logger.info(f"Fetched {len(documents)} documents for product {product_id}.")
        return documents
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
             st.warning(f"Product ID {product_id} not found on the server.") # User-friendly warning for 404
             logger.warning(f"Product ID {product_id} not found when fetching documents.")
        else:
            st.error(f"HTTP error fetching documents for product {product_id}: {http_err}")
            logger.error(f"HTTP error fetching documents: {http_err}", exc_info=True)
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API to fetch documents: {e}")
        logger.error(f"Network error fetching documents: {e}", exc_info=True)
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching documents: {e}")
        logger.error(f"Unexpected error fetching documents: {e}", exc_info=True)
        return []


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

def display_search_results_in_chat(results: Dict, show_details: bool):
    """Formats and displays search results within the chat interface."""
    # Display LLM response first
    response_text = results.get("llm_response", "No LLM response found in results.")
    parts = response_text.split("[IMAGE:")

    # Display the first part (text before any images)
    st.markdown(parts[0], unsafe_allow_html=True)

    # Process remaining parts (if any contain images)
    for part in parts[1:]:
        if "]" in part:
            image_path, remaining_text = part.split("]", 1)
            image_path = image_path.strip()
            display_image(image_path) # Display image inline
            st.markdown(remaining_text, unsafe_allow_html=True) # Display text after image

    # Display detailed results if requested and available
    if show_details and "search_results" in results:
        with st.expander("Show Detailed Search Context"):
            st.markdown("---") # Separator
            if "product_id" in results:
                 st.markdown(f"*(Searched within Product ID: {results['product_id']})*")

            for i, group in enumerate(results["search_results"]):
                 st.markdown(f"#### Context Group {i+1}")
                 # Display context (high similarity match)
                 if group.get("context"):
                     context = group["context"]
                     st.markdown(f"##### Page {context['metadata']['page_number']} (High Relevance - Score: {context.get('score', 'N/A'):.4f})")
                     st.text_area(f"Context_{i}", context["content"], height=150, key=f"context_area_{i}")

                     # Display images for context
                     if context.get("images"):
                         st.markdown("**Related Images (Context):**")
                         cols = st.columns(3) # Adjust columns as needed
                         for idx, img in enumerate(context["images"]):
                             with cols[idx % 3]:
                                 display_image(img["path"])

                 # Display related pages
                 if group.get("pages"):
                     st.markdown("##### Related Pages (Lower Relevance)")
                     for page in group["pages"]:
                         st.markdown(f"**Page {page['metadata']['page_number']} (Score: {page.get('score', 'N/A'):.4f}):**")
                         st.text_area(f"Related_{i}_{page['metadata']['page_number']}", page["content"], height=100, key=f"related_area_{i}_{page['metadata']['page_number']}")

                         # Display images for related pages
                         if page.get("images"):
                             st.markdown("**Related Images (Page):**")
                             cols_rel = st.columns(3) # Adjust columns as needed
                             for idx_rel, img_rel in enumerate(page["images"]):
                                  with cols_rel[idx_rel % 3]:
                                       display_image(img_rel["path"])
                 st.markdown("---") # Separator between groups

def main():
    st.set_page_config(layout="wide", page_title="Doc RAG")
    st.title("📄 Technical Documentation RAG System")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Fetch products for dropdowns
    products_map = get_products()
    product_names = list(products_map.keys())

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")

        with st.expander("⚙️ Search Tuning", expanded=True):
            if not product_names:
                st.warning("Create a product below before searching.")
                selected_product_name_search = None
                st.session_state.search_product_id = None
            else:
                # Use session state to preserve selection across reruns
                if 'selected_product_name_search' not in st.session_state:
                    st.session_state.selected_product_name_search = product_names[0] if product_names else None

                selected_product_name_search = st.selectbox(
                    "Select Product to Search",
                    options=product_names,
                    key='selected_product_name_search',
                    index=product_names.index(st.session_state.selected_product_name_search) if st.session_state.selected_product_name_search in product_names else 0
                )
                st.session_state.search_product_id = products_map.get(selected_product_name_search)

            st.session_state.similarity_threshold = st.slider("Similarity Threshold", 0.1, 1.0, 0.7, key='threshold_slider')
            st.session_state.result_limit = st.slider("Result Limit", 1, 20, 7, key='limit_slider')
            st.session_state.context_pages = st.slider("Context Pages", 1, 10, 5, key='context_slider')
            st.session_state.show_details = st.checkbox("Show Detailed Results", value=st.session_state.get("show_details", False), key='details_checkbox')

        with st.expander("📦 Document Management"):

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
                            st.rerun() # Rerun to refresh product list
                        elif response.status_code == 400:
                             st.error(f"Error: {response.json().get('detail', response.text)}")
                        else:
                            response.raise_for_status()
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error creating product: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

            # --- Document Listing (Commented Out) ---
            # st.subheader("Existing Documents")
            # if st.session_state.get('search_product_id') and selected_product_name_search:
            #     # Display info about the selected product
            #     st.caption(f"Showing documents for: **{selected_product_name_search}** (ID: {st.session_state.search_product_id})")
                
            #     # Fetch documents for the selected product
            #     product_documents = get_documents_for_product(st.session_state.search_product_id)
                
            #     if product_documents:
            #          # Display documents in the new format
            #          for doc in product_documents:
            #              doc_id = doc.get('id')
            #              title = doc.get('title', doc.get('original_filename', f"Document {doc_id}")) # Fallback title
            #              version = doc.get('version', 'N/A')
            #              download_uri = doc.get('download_uri') # Get the URI from the response
                         
            #              if download_uri:
            #                  # Construct the full download URL using the base API URL
            #                  full_download_url = f"{API_BASE_URL}{download_uri}"
            #                  # Display as hyperlink: Title:Version
            #                  st.markdown(f"[{title}]({full_download_url}): {version}", unsafe_allow_html=True)
            #              else:
            #                  # Display text if no URI (error case)
            #                  st.markdown(f"{title}: {version} (Download link unavailable)")
                         
            #              # TODO: Add delete button maybe?
            #     elif product_documents == []: # Explicitly check for empty list vs None/Error
            #          st.info("No documents found for this product.")
            #     # else: # Error case handled within get_documents_for_product
            #     #     st.error("Failed to fetch documents.")
            # else:
            #     st.info("Select a product in 'Search Tuning' to see its documents.")


            # --- Document Upload ---
            st.subheader("Upload New Document")
            if not product_names:
                st.warning("Please create a product first before uploading documents.")
                upload_product_selected = None
            else:
                 # Allow selecting product for upload separately
                 upload_product_selected = st.selectbox(
                     "Select Product for Upload",
                     options=product_names,
                     index=None,
                     placeholder="Choose a product...",
                     key="upload_product_select"
                 )

            doc_version = st.text_input("Version (Optional)", key="upload_version")
            doc_title = st.text_input("Title (Optional)", key="upload_title")
            uploaded_file = st.file_uploader(
                "Upload PDF Document",
                type=["pdf"],
                disabled=(not upload_product_selected),
                key="pdf_uploader"
            )

            upload_button_disabled = not (uploaded_file and upload_product_selected)
            if st.button("Upload and Process Document", disabled=upload_button_disabled, key="upload_button"):
                selected_product_id_upload = products_map.get(upload_product_selected)
                if selected_product_id_upload and uploaded_file:
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    data = {"product_id": selected_product_id_upload}
                    if doc_version: data['version'] = doc_version
                    if doc_title: data['title'] = doc_title
                    if uploaded_file.type: data['content_type'] = uploaded_file.type

                    st.info(f"""Uploading '{uploaded_file.name}' (Version: {doc_version or 'N/A'}, Title: {doc_title or 'N/A'})
                              for product '{upload_product_selected}' (ID: {selected_product_id_upload})...""")
                    try:
                        # Step 1: Upload
                        response = requests.post(f"{API_BASE_URL}/documents/upload", files=files, data=data)

                        if response.status_code == 200:
                            st.success("Document uploaded! Processing...")
                            document_info = response.json()
                            document_id = document_info["id"]

                            # Step 2: Process
                            with st.spinner(f"Processing document ID {document_id}..."):
                                process_response = requests.post(f"{API_BASE_URL}/documents/{document_id}/process")
                                if process_response.status_code == 200:
                                    st.success("Document processed successfully!")
                                    # Clear uploader and potentially rerun to update lists
                                    st.rerun()
                                else:
                                     logger.error(f"Error processing document {document_id}: {process_response.status_code} - {process_response.text}")
                                     st.error(f"Processing failed: {process_response.text}")

                        elif response.status_code in [400, 404, 409]: # Handle specific client errors
                             st.error(f"Upload Error: {response.json().get('detail', response.text)}")
                        else:
                            response.raise_for_status() # Raise for other server errors

                    except requests.exceptions.RequestException as e:
                        st.error(f"Network error during upload/processing: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                else:
                     st.error("Missing file or selected product for upload.")


    # --- Main Chat Area ---

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Check if content is a dict (search results) or string (user query)
            if isinstance(message["content"], dict):
                # Display search results using the dedicated function
                display_search_results_in_chat(message["content"], message.get("show_details", False))
            else:
                # Display simple text (user query)
                st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is your question?"):
        # Check if a product is selected for search
        if not st.session_state.get('search_product_id'):
            st.warning("Please select a product in the 'Search Tuning' sidebar section before asking a question.")
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Prepare and display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")

                # Prepare search request payload
                search_payload = {
                    "query": prompt,
                    "product_id": st.session_state.search_product_id,
                    "threshold": st.session_state.similarity_threshold,
                    "limit": st.session_state.result_limit,
                    "context_pages": st.session_state.context_pages
                }
                logger.info(f"Sending search request: {search_payload}")

                try:
                    # Call the search API - Reverted to /search/enhanced
                    search_response = requests.get(f"{API_BASE_URL}/search/enhanced", params=search_payload) # Use GET and params, back to /enhanced
                    search_response.raise_for_status() # Raise for bad status codes
                    results = search_response.json()
                    logger.info(f"Received search results: {json.dumps(results)[:200]}...") # Log snippet

                    # Display results using the dedicated function
                    message_placeholder.empty() # Clear "Thinking..."
                    display_search_results_in_chat(results, st.session_state.show_details)

                    # Add assistant response (results dict) to chat history
                    # We store the full result dict to allow re-displaying details later if needed
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": results,
                        "show_details": st.session_state.show_details # Store show_details state with the message
                    })

                except requests.exceptions.RequestException as e:
                    logger.error(f"Search API request failed: {e}", exc_info=True)
                    message_placeholder.error(f"Search failed: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: Search failed. {e}"})
                except Exception as e:
                    logger.error(f"An unexpected error occurred during search: {e}", exc_info=True)
                    message_placeholder.error(f"An unexpected error occurred: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: An unexpected error occurred. {e}"})

if __name__ == "__main__":
    main() 